"""感知層 - Sherpa-ONNX 串流 STT 引擎（Paraformer-Bilingual，中英混切）"""

from __future__ import annotations

import json
import os
import queue
import re
import sys
import threading
import time
from enum import Enum
from pathlib import Path

import numpy as np

try:
    import sherpa_onnx  # type: ignore
    _so_import_error = ""
except Exception as _e:
    sherpa_onnx = None  # type: ignore[assignment]
    _so_import_error = str(_e)

try:
    import opencc  # type: ignore
    _s2twp = opencc.OpenCC("s2twp")
except Exception:
    _s2twp = None


def _to_traditional(text: str) -> str:
    if _s2twp is None:
        return text
    try:
        return _s2twp.convert(text)
    except Exception:
        return text


class State(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPED = "stopped"


# Sherpa-ONNX 幻覺過濾
_HALLUCINATION_PATTERNS = re.compile(
    r"(thank you for watching|字幕由|請訂閱|訂閱頻道|點讚|不吝|掌聲|♪|♫|music|ambient|"
    r"by\s+\w+\s+caption|subtitles?\s+by|"
    r"台灣繁體中文會議|包含商業術語與英文詞彙|以下是台灣|"
    r"翻譯中|翻唱中|字幕製作|暢時暢時|臺灣繁號|繁號五十七|"
    r"這首歌是以往|這首歌是)",
    re.IGNORECASE,
)

_PUNCTUATION_ONLY = re.compile(r'^[\s\W]+$')


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _resources_root() -> Path | None:
    if not _is_frozen():
        return None
    exe = Path(sys.executable).resolve()
    backend_dir = exe.parent
    if backend_dir.name.lower() == "backend":
        return backend_dir.parent
    return exe.parent


def _load_model_pack_config() -> dict:
    candidates = []
    resources = _resources_root()
    if resources:
        candidates.append(resources / "model_pack_config.json")
    candidates.append(_project_root() / "desktop" / "model_pack_config.json")
    for p in candidates:
        try:
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
    return {}


def _has_encoder(p: Path) -> bool:
    return (p / "encoder.int8.onnx").exists() or (p / "encoder.onnx").exists()


def _find_sherpa_model_dir() -> Path | None:
    """尋找本地 Sherpa-ONNX 模型目錄（含 encoder.int8.onnx 或 encoder.onnx）"""
    env_dir = os.environ.get("AMA_SHERPA_DIR", "").strip()
    if env_dir:
        p = Path(env_dir).expanduser()
        if p.is_dir() and _has_encoder(p):
            return p

    cfg = _load_model_pack_config()
    model_dir_name = str(cfg.get(
        "sherpaModelDirName",
        "sherpa-onnx-streaming-paraformer-bilingual-zh-en"
    )).strip()

    candidates: list[Path] = []
    resources = _resources_root()
    if resources:
        candidates.append(resources / "models" / "sherpa-onnx" / model_dir_name)
        candidates.append(resources / "models" / "sherpa-onnx")

    root = _project_root()
    for base in (
        root / "desktop" / "models" / "sherpa-onnx",
        root / "models" / "sherpa-onnx",
    ):
        candidates.append(base / model_dir_name)
        candidates.append(base)

    probe_roots = [Path(__file__).resolve().parent]
    try:
        probe_roots.append(Path(sys.executable).resolve().parent)
    except Exception:
        pass
    seen: set[str] = set()
    for probe in probe_roots:
        for anc in [probe, *probe.parents]:
            key = str(anc)
            if key in seen:
                continue
            seen.add(key)
            base = anc / "models" / "sherpa-onnx"
            candidates.append(base / model_dir_name)
            candidates.append(base)

    for p in candidates:
        if not p.exists():
            continue
        if p.is_dir() and _has_encoder(p):
            return p
        if p.is_dir():
            nested = sorted(
                [d for d in p.iterdir() if d.is_dir() and _has_encoder(d)]
            )
            if nested:
                return nested[0]
    return None


class STTEngine:
    SAMPLE_RATE = 16000
    SILENCE_THRESHOLD = 0.003

    def __init__(self, model_size: str = "base"):
        if sherpa_onnx is None:
            raise RuntimeError(f"sherpa_onnx 無法載入: {_so_import_error}")

        self._lock = threading.Lock()
        self._state = State.IDLE
        self._last_partial_text = ""
        self._last_confirmed_text = ""
        self._last_audio_rms = 0.0
        self._result_callback = None
        self._current_speaker = 1
        self._last_segment_end = 0.0
        self._time_offset_sec = 0.0

        self._recognizer = self._create_recognizer()
        self._stream = self._recognizer.create_stream()

        # 時間 fallback：偵測文字停止變化
        self._last_seen_text = ""
        self._last_text_change_time = 0.0
        self._TEXT_STALE_TIMEOUT = 2.0  # 文字超過 2 秒沒變化 → 強制輸出

        # 背景 worker：避免 decode() 阻塞 SocketIO 事件執行緒
        self._audio_queue: queue.Queue = queue.Queue(maxsize=300)
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="stt-worker"
        )
        self._worker_thread.start()
        print("[STT] Sherpa-ONNX Paraformer 模型載入完成", flush=True)

    def _create_recognizer(self) -> "sherpa_onnx.OnlineRecognizer":
        local_dir = _find_sherpa_model_dir()
        if not local_dir:
            raise RuntimeError(
                "找不到 Sherpa-ONNX 模型目錄，請確認模型已安裝\n"
                "（預期目錄：models/sherpa-onnx/sherpa-onnx-streaming-paraformer-bilingual-zh-en）"
            )

        encoder = str(local_dir / "encoder.int8.onnx")
        decoder = str(local_dir / "decoder.int8.onnx")
        if not Path(encoder).exists():
            encoder = str(local_dir / "encoder.onnx")
            decoder = str(local_dir / "decoder.onnx")
        tokens = str(local_dir / "tokens.txt")

        cpu_count = os.cpu_count() or 4
        num_threads = max(2, min(4, cpu_count // 2))

        print(f"[STT] 使用本地模型: {local_dir}", flush=True)
        return sherpa_onnx.OnlineRecognizer.from_paraformer(
            encoder=encoder,
            decoder=decoder,
            tokens=tokens,
            num_threads=num_threads,
            sample_rate=self.SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=20,
        )

    # ── 屬性 ──────────────────────────────────────────────

    @property
    def state(self) -> str:
        with self._lock:
            return self._state.value

    @property
    def partial_text(self) -> str:
        with self._lock:
            return self._last_partial_text

    @property
    def last_audio_rms(self) -> float:
        with self._lock:
            return self._last_audio_rms

    # ── 控制 ──────────────────────────────────────────────

    def start(self) -> str:
        with self._lock:
            if self._state != State.IDLE:
                return self._state.value
            self._stream = self._recognizer.create_stream()
            self._last_partial_text = ""
            self._last_confirmed_text = ""
            self._last_audio_rms = 0.0
            self._current_speaker = 1
            self._last_segment_end = 0.0
            self._time_offset_sec = 0.0
            self._state = State.RECORDING
            return self._state.value

    def pause(self) -> str:
        with self._lock:
            if self._state != State.RECORDING:
                return self._state.value
            self._state = State.PAUSED
            return self._state.value

    def resume(self) -> str:
        with self._lock:
            if self._state != State.PAUSED:
                return self._state.value
            self._state = State.RECORDING
            return self._state.value

    def request_stop(self) -> None:
        with self._lock:
            if self._state not in (State.RECORDING, State.PAUSED):
                return
            self._state = State.IDLE
            stream = self._stream
        # 送 tail padding 刷出最後一段
        self._flush_stream(stream)

    def stop(self) -> tuple[str, list[dict]]:
        self.request_stop()
        return "stopped", []

    def reset(self):
        with self._lock:
            self._stream = self._recognizer.create_stream()
            self._last_partial_text = ""
            self._last_confirmed_text = ""
            self._time_offset_sec = 0.0
            self._state = State.IDLE

    def set_result_callback(self, cb):
        """設定辨識結果回呼，由推論執行緒呼叫"""
        self._result_callback = cb

    # ── 音頻處理 ──────────────────────────────────────────

    def feed_audio(self, chunk: bytes) -> list[dict]:
        """非阻塞：把 chunk 丟進隊列馬上回傳，由 worker 執行緒處理"""
        with self._lock:
            if self._state != State.RECORDING:
                return []
        try:
            self._audio_queue.put_nowait(chunk)
        except queue.Full:
            print("[STT] audio queue full, dropping chunk", flush=True)
        return []

    def _worker_loop(self) -> None:
        """背景執行緒：持續從 queue 取 chunk 並執行 sherpa-onnx 推論"""
        print("[STT] worker thread started", flush=True)
        while True:
            try:
                chunk = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._process_chunk(chunk)
            except Exception as e:
                print(f"[STT] worker 例外: {e}", flush=True)

    def _process_chunk(self, chunk: bytes) -> None:
        with self._lock:
            if self._state != State.RECORDING:
                return
            stream = self._stream

        try:
            pcm16 = np.frombuffer(chunk, dtype=np.int16)
            if pcm16.size == 0:
                return
            samples = pcm16.astype(np.float32) / 32768.0
        except Exception as e:
            print(f"[STT] PCM 解析錯誤: {e}", flush=True)
            return

        rms = float(np.sqrt(np.mean(np.square(samples))))
        with self._lock:
            self._last_audio_rms = rms

        # 送入串流辨識器並解碼
        stream.accept_waveform(self.SAMPLE_RATE, samples)
        ready_count = 0
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)
            ready_count += 1
        if ready_count == 0:
            print("[STT] is_ready=False, no decode this chunk", flush=True)

        result = self._recognizer.get_result(stream)
        text = (result.text if hasattr(result, "text") else str(result)).strip()

        is_ep = self._recognizer.is_endpoint(stream)
        if text:
            print(f"[STT] partial={repr(text[:40])} endpoint={is_ep}", flush=True)

        # 時間 fallback：文字超過 2 秒沒變化就強制輸出
        now = time.monotonic()
        if text != self._last_seen_text:
            self._last_seen_text = text
            self._last_text_change_time = now
        stale = (text and not is_ep
                 and self._last_text_change_time > 0
                 and (now - self._last_text_change_time) >= self._TEXT_STALE_TIMEOUT)

        if is_ep or stale:
            reason = "endpoint" if is_ep else "stale_timeout"
            print(f"[STT] flush ({reason}), text={repr(text)}", flush=True)
            if text:
                self._emit_text(text)
            self._recognizer.reset(stream)
            self._last_seen_text = ""
            self._last_text_change_time = 0.0
            with self._lock:
                self._last_partial_text = ""
        else:
            with self._lock:
                self._last_partial_text = _to_traditional(text) if text else ""

    def transcribe_audio(self, audio: np.ndarray) -> list[dict]:
        """相容舊介面（stop 時呼叫），sherpa-onnx 串流版不需要"""
        return []

    def _flush_stream(self, stream) -> None:
        """送入 tail padding，刷出最後未送出的辨識結果"""
        tail = np.zeros(int(0.5 * self.SAMPLE_RATE), dtype=np.float32)
        stream.accept_waveform(self.SAMPLE_RATE, tail)
        while self._recognizer.is_ready(stream):
            self._recognizer.decode(stream)
        result = self._recognizer.get_result(stream)
        text = (result.text if hasattr(result, "text") else str(result)).strip()
        if text:
            self._emit_text(text)

    def _emit_text(self, text: str) -> None:
        text = _to_traditional(text)
        filtered = self._filter_repetitions(text)
        if not filtered:
            print(f"[STT] _emit_text: dropped by repetition filter, original={repr(text[:60])}", flush=True)
            return
        text = filtered
        if _HALLUCINATION_PATTERNS.search(text):
            print(f"[STT] _emit_text: dropped by hallucination filter: {repr(text[:60])}", flush=True)
            return
        if _PUNCTUATION_ONLY.match(text):
            print(f"[STT] _emit_text: dropped punctuation-only: {repr(text)}", flush=True)
            return
        if text == self._last_confirmed_text:
            print(f"[STT] _emit_text: dropped duplicate: {repr(text[:60])}", flush=True)
            return
        self._last_confirmed_text = text

        now = self._time_offset_sec
        self._time_offset_sec += 1.0
        segment = {
            "text": text,
            "start": now,
            "end": now + 1.0,
            "language": "zh",
            "speaker": self._current_speaker,
        }
        if self._result_callback:
            try:
                self._result_callback([segment])
            except Exception as e:
                print(f"[STT] callback 錯誤: {e}", flush=True)

    @staticmethod
    def _filter_repetitions(text: str) -> str:
        if len(text) > 3:
            text = re.sub(r'(.)\1{2,}', r'\1\1', text)             # 嗯嗯嗯嗯 → 嗯嗯
            text = re.sub(r'(.{2,20}?)\1{2,}', r'\1', text)        # 非貪婪：優先捕捉短重複
            text = re.sub(r'(.{2,20})\1{2,}', r'\1', text)         # 貪婪再跑一次
            for length in range(3, min(len(text) // 3 + 1, 21)):
                phrase = text[:length]
                count = text.count(phrase)
                if count >= 4:
                    remainder = text.replace(phrase, '').replace('，', '').replace(',', '')
                    if len(remainder) < length:
                        return ''
            if len(text) > 300:
                return ''
        return text.strip()
