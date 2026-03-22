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
    """偵測目錄是否含有 encoder 模型檔（支援 Paraformer 和 Zipformer 兩種命名格式）。"""
    # Paraformer 固定命名：encoder.int8.onnx / encoder.onnx
    if (p / "encoder.int8.onnx").exists() or (p / "encoder.onnx").exists():
        return True
    # Zipformer/Transducer 含版本號命名：encoder-epoch-*.onnx
    return any(p.glob("encoder-*.onnx"))


def _find_sherpa_model_dir() -> Path | None:
    """尋找本地 Sherpa-ONNX 模型目錄（支援 Paraformer 與 Zipformer/Transducer）"""
    env_dir = os.environ.get("AMA_SHERPA_DIR", "").strip()
    if env_dir:
        p = Path(env_dir).expanduser()
        if p.is_dir() and _has_encoder(p):
            return p

    cfg = _load_model_pack_config()
    model_dir_name = str(cfg.get(
        "sherpaModelDirName",
        "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
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
    SILENCE_THRESHOLD = 0.003  # VAD 靜音門檻（RMS）
    CHUNK_SAMPLES = 480         # 30ms @ 16kHz

    def __init__(self, model_size: str = "base"):
        if sherpa_onnx is None:
            raise RuntimeError(f"sherpa_onnx 無法載入: {_so_import_error}")

        self._lock = threading.Lock()
        self._state = State.IDLE
        self._pending_flush_stream = None  # 停止後等待 background flush 的 stream
        self._last_partial_text = ""
        self._last_confirmed_text = ""
        self._last_audio_rms = 0.0
        self._result_callback = None
        self._partial_callback = None
        self._current_speaker = 1
        self._last_segment_end = 0.0
        self._time_offset_sec = 0.0

        self._recognizer = self._create_recognizer()
        self._stream = self._recognizer.create_stream()

        # 時間 fallback：偵測文字停止變化
        self._last_seen_text = ""
        self._last_text_change_time = 0.0
        self._TEXT_STALE_TIMEOUT = 1.5  # 文字超過 1.5 秒沒變化 → 強制輸出

        # 20ms chunk buffer：累積 samples 後以 320 個為單位送 Sherpa
        self._sample_buffer = np.array([], dtype=np.float32)

        # 背景 worker：避免 decode() 阻塞 SocketIO 事件執行緒
        self._audio_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="stt-worker"
        )
        self._worker_thread.start()
        print("[STT] Sherpa-ONNX 串流模型載入完成", flush=True)

    def _create_recognizer(self) -> "sherpa_onnx.OnlineRecognizer":
        local_dir = _find_sherpa_model_dir()
        if not local_dir:
            raise RuntimeError(
                "找不到 Sherpa-ONNX 模型目錄，請確認模型已安裝\n"
                "（預期目錄：models/sherpa-onnx/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20）"
            )
        print(f"[STT] 使用本地模型: {local_dir}", flush=True)
        # 偵測模型類型：含 joiner 檔案 → Transducer（Zipformer）；否則 → Paraformer
        is_transducer = any(local_dir.glob("joiner*.onnx")) or any(local_dir.glob("joiner-*.onnx"))
        if is_transducer:
            return self._create_transducer_recognizer(local_dir)
        return self._create_paraformer_recognizer(local_dir)

    @staticmethod
    def _find_model_file(model_dir: Path, prefix: str) -> str:
        """在模型目錄中找指定前綴的 ONNX 檔，優先選 int8 量化版。"""
        # 先找固定命名（Paraformer 格式）
        for suffix in (".int8.onnx", ".onnx"):
            p = model_dir / f"{prefix}{suffix}"
            if p.exists():
                return str(p)
        # 再找帶版本號命名（Zipformer 格式：prefix-epoch-*.int8.onnx）
        for suffix in (".int8.onnx", ".onnx"):
            candidates = sorted(model_dir.glob(f"{prefix}-*{suffix}"))
            if candidates:
                return str(candidates[0])
        return ""

    def _create_transducer_recognizer(self, local_dir: Path) -> "sherpa_onnx.OnlineRecognizer":
        """建立 Zipformer Transducer 串流辨識器（from_transducer API）。"""
        encoder = self._find_model_file(local_dir, "encoder")
        decoder = self._find_model_file(local_dir, "decoder")
        joiner  = self._find_model_file(local_dir, "joiner")
        tokens  = str(local_dir / "tokens.txt")
        print(f"[STT] 模型類型: Zipformer Transducer", flush=True)
        print(f"[STT]   encoder={Path(encoder).name}", flush=True)
        print(f"[STT]   decoder={Path(decoder).name}", flush=True)
        print(f"[STT]   joiner ={Path(joiner).name}", flush=True)
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=4,
            sample_rate=self.SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=0.8,
            rule3_min_utterance_length=10,
        )

    def _create_paraformer_recognizer(self, local_dir: Path) -> "sherpa_onnx.OnlineRecognizer":
        """建立 Paraformer 串流辨識器（from_paraformer API）。"""
        encoder = self._find_model_file(local_dir, "encoder")
        decoder = self._find_model_file(local_dir, "decoder")
        tokens  = str(local_dir / "tokens.txt")
        print(f"[STT] 模型類型: Paraformer", flush=True)
        print(f"[STT]   encoder={Path(encoder).name}", flush=True)
        print(f"[STT]   decoder={Path(decoder).name}", flush=True)
        return sherpa_onnx.OnlineRecognizer.from_paraformer(
            encoder=encoder,
            decoder=decoder,
            tokens=tokens,
            num_threads=4,
            sample_rate=self.SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=0.8,
            rule3_min_utterance_length=10,
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
            self._last_seen_text = ""
            self._last_text_change_time = 0.0
            self._sample_buffer = np.array([], dtype=np.float32)
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
        t_rs = time.monotonic()
        with self._lock:
            if self._state not in (State.RECORDING, State.PAUSED):
                return
            self._state = State.IDLE
            stream = self._stream
            self._pending_flush_stream = stream  # 交給 background task 執行 flush
        # 把 queue 中殘留的音訊 chunk 直接送入 stream，讓 _flush_stream 能解碼最後一段語音
        # （不直接丟棄，避免錄音結束前最後幾秒的語音遺失）
        t_drain = time.monotonic()
        drained = 0
        while not self._audio_queue.empty():
            try:
                chunk = self._audio_queue.get_nowait()
                drained += 1
                try:
                    pcm16 = np.frombuffer(chunk, dtype=np.int16)
                    if pcm16.size > 0:
                        samples = pcm16.astype(np.float32) / 32768.0
                        stream.accept_waveform(self.SAMPLE_RATE, samples)
                except Exception as e:
                    print(f"[STT] request_stop drain 例外: {e}", flush=True)
            except queue.Empty:
                break
        print(f"[STT] request_stop: queue drain {drained} chunk(s)，耗時 {(time.monotonic()-t_drain)*1000:.0f}ms，total {(time.monotonic()-t_rs)*1000:.0f}ms", flush=True)
        # _flush_stream 移到 background task，此處不再阻塞 event handler

    def take_pending_flush_stream(self):
        """取走等待 flush 的 stream（每次 stop 後呼叫一次）。回傳 stream 或 None。"""
        with self._lock:
            s = self._pending_flush_stream
            self._pending_flush_stream = None
            return s

    def stop(self) -> tuple[str, list[dict]]:
        self.request_stop()
        return "stopped", []

    def reset(self):
        """清除所有內部狀態，確保新會議從乾淨狀態開始。"""
        with self._lock:
            self._stream = self._recognizer.create_stream()
            self._last_partial_text = ""
            self._last_confirmed_text = ""
            self._last_seen_text = ""
            self._last_text_change_time = 0.0
            self._last_audio_rms = 0.0
            self._current_speaker = 1
            self._time_offset_sec = 0.0
            self._sample_buffer = np.array([], dtype=np.float32)
            self._state = State.IDLE
        # 清空 audio queue（鎖外執行，避免與 worker thread 競態）
        drained = 0
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
        if drained:
            print(f"[STT] reset: drained {drained} stale chunk(s) from queue", flush=True)

    def set_result_callback(self, cb):
        """設定辨識結果回呼，由推論執行緒呼叫"""
        self._result_callback = cb

    def set_partial_callback(self, cb):
        """設定即時 partial text 回呼"""
        self._partial_callback = cb

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

        # 累積進 buffer，以 CHUNK_SAMPLES (20ms) 為單位處理
        self._sample_buffer = np.concatenate([self._sample_buffer, samples])
        while len(self._sample_buffer) >= self.CHUNK_SAMPLES:
            window = self._sample_buffer[:self.CHUNK_SAMPLES]
            self._sample_buffer = self._sample_buffer[self.CHUNK_SAMPLES:]
            self._process_window(window, stream)

    def _process_window(self, samples: np.ndarray, stream) -> None:
        """處理單一 20ms 音訊窗口：VAD → Sherpa decode → endpoint/stale 判斷"""
        rms = float(np.sqrt(np.mean(np.square(samples))))
        with self._lock:
            self._last_audio_rms = rms

        # VAD：RMS 低於門檻，靜音跳過，不送 Sherpa
        if rms < self.SILENCE_THRESHOLD:
            return

        # 送入串流辨識器並解碼
        stream.accept_waveform(self.SAMPLE_RATE, samples)
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)

        result = self._recognizer.get_result(stream)
        text = (result.text if hasattr(result, "text") else str(result)).strip()

        is_ep = self._recognizer.is_endpoint(stream)
        if text:
            print(f"[STT] partial={repr(text[:40])} endpoint={is_ep}", flush=True)

        # 時間 fallback：文字超過 stale timeout 沒變化就強制輸出
        now = time.monotonic()
        if text != self._last_seen_text:
            self._last_seen_text = text
            self._last_text_change_time = now
        stale = (text and not is_ep
                 and self._last_text_change_time > 0
                 and (now - self._last_text_change_time) >= self._TEXT_STALE_TIMEOUT)

        if is_ep or stale:
            reason = "endpoint" if is_ep else "stale_timeout"
            # 保存觸發 endpoint 的 window，reset 後回放以補救段落邊界漏字
            boundary_chunk = samples.copy()
            # 優先用 finalize_decoding 強制輸出，不需餵靜音；不支援時 fallback 0.5s padding
            try:
                self._recognizer.finalize_decoding(stream)
            except Exception:
                tail = np.zeros(int(0.08 * self.SAMPLE_RATE), dtype=np.float32)
                stream.accept_waveform(self.SAMPLE_RATE, tail)
            while self._recognizer.is_ready(stream):
                self._recognizer.decode_stream(stream)
            final_result = self._recognizer.get_result(stream)
            final_text = (final_result.text if hasattr(final_result, "text") else str(final_result)).strip()
            # 取較長的版本，避免 padding 後反而縮短
            emit_text = final_text if len(final_text) >= len(text) else text
            print(f"[STT] flush ({reason}), text={repr(emit_text)}", flush=True)
            if emit_text:
                self._emit_text(emit_text)
            # 若停止已被要求，保留 stream 讓 _flush_stream 接手，不重置也不 warmup
            with self._lock:
                stopping = self._state != State.RECORDING
            if stopping:
                self._last_seen_text = ""
                self._last_text_change_time = 0.0
                self._last_confirmed_text = ""
                with self._lock:
                    self._last_partial_text = ""
                return
            self._recognizer.reset(stream)
            self._last_seen_text = ""
            self._last_text_change_time = 0.0
            self._last_confirmed_text = ""  # 清除跨段去重記憶，避免新段開頭被誤判重複
            with self._lock:
                self._last_partial_text = ""
            # 回放 boundary chunk 讓 encoder 熱身，避免新段前幾字丟失
            stream.accept_waveform(self.SAMPLE_RATE, boundary_chunk)
            while self._recognizer.is_ready(stream):
                self._recognizer.decode_stream(stream)
            warmup_result = self._recognizer.get_result(stream)
            warmup_text = (warmup_result.text if hasattr(warmup_result, "text") else str(warmup_result)).strip()
            if warmup_text:
                self._last_seen_text = warmup_text
                self._last_text_change_time = time.monotonic()
        else:
            # 有變動才 emit partial，避免重複推送
            trad = _to_traditional(text) if text else ""
            with self._lock:
                changed = trad != self._last_partial_text
                self._last_partial_text = trad
            if changed and self._partial_callback:
                try:
                    self._partial_callback(trad)
                except Exception as e:
                    print(f"[STT] partial callback 錯誤: {e}", flush=True)

    def transcribe_audio(self, audio: np.ndarray) -> list[dict]:
        """相容舊介面（stop 時呼叫），sherpa-onnx 串流版不需要"""
        return []

    def _flush_stream(self, stream) -> None:
        """送入 tail padding，強制刷出最後未送出的辨識結果。

        優化策略（依順序嘗試，盡早結束）：
        1. finalize_decoding()：若此版本支援則立即強制輸出，幾乎不耗時
        2. 初始 decode loop：消化 finalize 後的 ready chunks
        3. 取結果：若已有文字則直接結束（避免後續 padding 耗時）
        4. 補 1.0s silence padding（僅在步驟 3 無結果時才執行）
           - Paraformer right_context ≈ 60–160ms，1.0s 已超過 6 倍以上
           - 原本 2.5s 約需 2–5s CPU 時間（83 個 30ms encoder chunk），已縮短
        """
        t_start = time.monotonic()
        try:
            # ── 步驟 1：嘗試 finalize_decoding（立即強制輸出，不需餵靜音）──
            t1 = time.monotonic()
            finalize_supported = False
            try:
                self._recognizer.finalize_decoding(stream)
                finalize_supported = True
                print(f"[STT] finalize_decoding() 支援，耗時 {(time.monotonic()-t1)*1000:.0f}ms", flush=True)
            except Exception as fe:
                print(f"[STT] finalize_decoding() 不支援 ({type(fe).__name__})，改用 padding", flush=True)

            # ── 步驟 2：decode 所有 ready chunks ──
            t2 = time.monotonic()
            n_decode = 0
            while self._recognizer.is_ready(stream):
                self._recognizer.decode_stream(stream)
                n_decode += 1
            print(f"[STT] 初始 decode {n_decode} chunk(s)，耗時 {(time.monotonic()-t2)*1000:.0f}ms", flush=True)

            # ── 步驟 3：嘗試直接取結果（若 finalize 有效則此處即可拿到文字）──
            t3 = time.monotonic()
            result = self._recognizer.get_result(stream)
            text = (result.text if hasattr(result, "text") else str(result)).strip()
            print(f"[STT] 第一次 get_result: {repr(text[:60]) if text else '(empty)'}，耗時 {(time.monotonic()-t3)*1000:.0f}ms", flush=True)

            # ── 步驟 4：若無結果，補 1.0s silence padding 再試一次 ──
            if not text:
                t4 = time.monotonic()
                # 1.0s 已足夠覆蓋 Paraformer right_context（≈160ms），不需要 2.5s
                tail = np.zeros(int(1.0 * self.SAMPLE_RATE), dtype=np.float32)
                stream.accept_waveform(self.SAMPLE_RATE, tail)
                n_pad_decode = 0
                while self._recognizer.is_ready(stream):
                    self._recognizer.decode_stream(stream)
                    n_pad_decode += 1
                print(f"[STT] 1.0s padding + decode {n_pad_decode} chunk(s)，耗時 {(time.monotonic()-t4)*1000:.0f}ms", flush=True)

                result = self._recognizer.get_result(stream)
                text = (result.text if hasattr(result, "text") else str(result)).strip()
                print(f"[STT] padding 後 get_result: {repr(text[:60]) if text else '(empty)'}", flush=True)

            total_ms = (time.monotonic() - t_start) * 1000
            print(f"[STT] _flush_stream 完成，total={total_ms:.0f}ms，text={repr(text[:60]) if text else '(empty)'}", flush=True)
            if text:
                self._emit_text(text)
        except Exception as e:
            print(f"[STT] _flush_stream 例外: {e}", flush=True)
        finally:
            try:
                self._recognizer.reset(stream)
            except Exception:
                pass

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
