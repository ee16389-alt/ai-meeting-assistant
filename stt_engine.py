"""感知層 - sherpa-onnx STT 引擎（僅使用 bundled 模型）"""

from __future__ import annotations

import json
import os
import sys
import threading
from enum import Enum
from pathlib import Path

import numpy as np
_sherpa_import_error: str = ""
try:
    import sherpa_onnx  # type: ignore
except Exception as _e:
    sherpa_onnx = None  # type: ignore[assignment]
    _sherpa_import_error = str(_e)

try:
    import opencc  # type: ignore
    _s2twp = opencc.OpenCC("s2twp")  # 簡體 → 繁體（台灣習慣用字）
except Exception:
    _s2twp = None


def _to_traditional(text: str) -> str:
    """將簡體中文轉換為繁體中文（台灣用字）；若 opencc 不可用則原文回傳。"""
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


# 說話者間隔門檻（秒）：語音段落間隙超過此值視為換人說話
SPEAKER_GAP_THRESHOLD = 1.5


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


def _find_bundled_sherpa_dir() -> Path:
    # 優先使用環境變數指定路徑（首次下載後由 Electron 設定）
    env_dir = os.environ.get("AMA_SHERPA_DIR", "").strip()
    if env_dir:
        p = Path(env_dir).expanduser()
        if p.is_dir() and (p / "tokens.txt").exists():
            return p
        # 可能指向父目錄，往下找一層
        if p.is_dir():
            nested = sorted([d for d in p.iterdir() if d.is_dir() and (d / "tokens.txt").exists()])
            if nested:
                return nested[0]

    cfg = _load_model_pack_config()
    configured_name = str(cfg.get("sherpaModelDirName", "")).strip()

    candidates: list[Path] = []
    resources = _resources_root()
    if resources:
        base = resources / "models" / "sherpa-onnx"
        if configured_name:
            candidates.append(base / configured_name)
        candidates.append(base)

    root = _project_root()
    for base in (root / "desktop" / "models" / "sherpa-onnx", root / "models" / "sherpa-onnx"):
        if configured_name:
            candidates.append(base / configured_name)
        candidates.append(base)

    # PyInstaller in Electron may run from .../Resources/backend/_internal.
    # Walk up from both __file__ and sys.executable to find the actual bundled models dir.
    probe_roots = [Path(__file__).resolve().parent]
    try:
        probe_roots.append(Path(sys.executable).resolve().parent)
    except Exception:
        pass
    seen = set()
    for probe in probe_roots:
        for anc in [probe, *probe.parents]:
            key = str(anc)
            if key in seen:
                continue
            seen.add(key)
            base = anc / "models" / "sherpa-onnx"
            if configured_name:
                candidates.append(base / configured_name)
            candidates.append(base)

    for p in candidates:
        if not p.exists():
            continue
        if p.is_dir() and (p / "tokens.txt").exists():
            return p
        if p.is_dir():
            nested = sorted(
                [d for d in p.iterdir() if d.is_dir() and (d / "tokens.txt").exists()],
                key=lambda x: x.name,
            )
            if nested:
                if configured_name:
                    for d in nested:
                        if d.name == configured_name:
                            return d
                return nested[0]
    raise FileNotFoundError("找不到 bundled sherpa-onnx 模型目錄")


def _pick_model_file(model_dir: Path, prefix: str) -> Path:
    # Prefer int8 models for CPU packaging.
    patterns = [f"{prefix}*.int8.onnx", f"{prefix}*.onnx"]
    for pattern in patterns:
        files = sorted(model_dir.glob(pattern))
        if files:
            return files[0]
    raise FileNotFoundError(f"找不到 {prefix}*.onnx ({model_dir})")


class STTEngine:
    TRANSCRIBE_INTERVAL_MS = 800
    SAMPLE_RATE = 16000

    def __init__(self, model_size="small"):
        del model_size  # API compatibility only; sherpa-onnx ignores this.
        if sherpa_onnx is None:
            raise RuntimeError(f"sherpa_onnx 無法載入: {_sherpa_import_error}")

        self._lock = threading.Lock()
        self._state = State.IDLE
        self._pcm_buffer = np.array([], dtype=np.float32)
        self._time_offset_sec = 0.0
        self._speaker_counter = 0
        self._current_speaker = 1
        self._last_segment_end = 0.0

        self._recognizer = self._create_recognizer()
        self._stream = self._recognizer.create_stream()
        self._last_partial_text = ""
        self._last_audio_rms = 0.0

        print("[STT] sherpa-onnx 模型載入完成", flush=True)

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

    def start(self) -> str:
        with self._lock:
            if self._state != State.IDLE:
                return self._state.value
            self._pcm_buffer = np.array([], dtype=np.float32)
            self._time_offset_sec = 0.0
            self._speaker_counter = 0
            self._current_speaker = 1
            self._last_segment_end = 0.0
            self._last_partial_text = ""
            self._last_audio_rms = 0.0
            self._stream = self._recognizer.create_stream()
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

    def stop(self) -> tuple[str, list[dict]]:
        """停止錄音並執行最終轉寫，回傳 (state, segments)"""
        with self._lock:
            if self._state not in (State.RECORDING, State.PAUSED):
                return self._state.value, []
            self._state = State.STOPPED
            segments = self._transcribe_remaining(finalize=True)
            self._state = State.IDLE
            return "stopped", segments

    def reset(self):
        with self._lock:
            self._pcm_buffer = np.array([], dtype=np.float32)
            self._time_offset_sec = 0.0
            self._speaker_counter = 0
            self._current_speaker = 1
            self._last_segment_end = 0.0
            self._last_partial_text = ""
            self._last_audio_rms = 0.0
            self._stream = self._recognizer.create_stream()
            self._state = State.IDLE

    def feed_audio(self, chunk: bytes) -> list[dict]:
        """接收 16kHz PCM int16 chunk，累積後觸發轉寫。回傳轉寫結果列表。"""
        with self._lock:
            if self._state != State.RECORDING:
                return []

            self._append_pcm_chunk(chunk)
            total_duration_ms = self._get_buffer_duration_ms()
            if total_duration_ms < self.TRANSCRIBE_INTERVAL_MS:
                return []
            return self._do_transcribe()

    def _create_recognizer(self):
        model_dir = _find_bundled_sherpa_dir()
        tokens = model_dir / "tokens.txt"
        encoder = _pick_model_file(model_dir, "encoder")
        decoder = _pick_model_file(model_dir, "decoder")
        joiner = _pick_model_file(model_dir, "joiner")

        print(f"[STT] 使用 bundled sherpa-onnx: {model_dir}", flush=True)
        print(
            f"[STT] encoder={encoder.name}, decoder={decoder.name}, joiner={joiner.name}",
            flush=True,
        )

        return sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(tokens),
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            num_threads=max(1, (os.cpu_count() or 4) - 1),
            sample_rate=self.SAMPLE_RATE,
            provider="cpu",
            decoding_method="modified_beam_search",
            max_active_paths=8,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=0.6,
            rule2_min_trailing_silence=0.4,
            rule3_min_utterance_length=20.0,
        )

    def _get_buffer_duration_ms(self) -> int:
        if self._pcm_buffer.size == 0:
            return 0
        return int((self._pcm_buffer.size / self.SAMPLE_RATE) * 1000)

    SILENCE_THRESHOLD = 0.005  # 靜音門檻：低於此音量不處理

    def _append_pcm_chunk(self, chunk: bytes) -> None:
        try:
            pcm16 = np.frombuffer(chunk, dtype=np.int16)
            if pcm16.size == 0:
                return
            pcm32 = pcm16.astype(np.float32) / 32768.0
            
            # 計算音量 (RMS)
            rms = float(np.sqrt(np.mean(np.square(pcm32)))) if pcm32.size else 0.0
            self._last_audio_rms = rms
            
            # 靜音過濾：如果音量太小，不加入緩衝區，避免 AI 產生幻覺
            if rms < self.SILENCE_THRESHOLD:
                return
                
            self._pcm_buffer = np.concatenate([self._pcm_buffer, pcm32])
        except Exception as e:
            print(f"[STT] PCM 解析錯誤: {e}", flush=True)

    def _do_transcribe(self) -> list[dict]:
        return self._decode_buffer(finalize=False)

    def _transcribe_remaining(self, finalize: bool = False) -> list[dict]:
        if self._pcm_buffer.size == 0 and not finalize:
            return []
        return self._decode_buffer(finalize=finalize)

    def _decode_buffer(self, finalize: bool) -> list[dict]:
        try:
            if self._pcm_buffer.size > 0:
                self._stream.accept_waveform(self.SAMPLE_RATE, self._pcm_buffer)
                self._time_offset_sec += self._pcm_buffer.size / self.SAMPLE_RATE
                self._pcm_buffer = np.array([], dtype=np.float32)

            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)

            if finalize:
                try:
                    self._stream.input_finished()
                except Exception:
                    pass
                while self._recognizer.is_ready(self._stream):
                    self._recognizer.decode_stream(self._stream)

            results = []
            current_text = _to_traditional(self._recognizer.get_result(self._stream).strip())
            should_emit = False

            if finalize:
                should_emit = bool(current_text)
            elif self._recognizer.is_endpoint(self._stream) and current_text:
                should_emit = True

            if should_emit:
                seg = self._make_segment(current_text)
                if seg:
                    results.append(seg)
                self._last_partial_text = ""
                self._recognizer.reset(self._stream)
            else:
                self._last_partial_text = current_text

            return results
        except Exception as e:
            print(f"[STT] sherpa-onnx 轉寫錯誤: {e}", flush=True)
            return []

    def _make_segment(self, text: str) -> dict | None:
        text = _to_traditional(text.strip())
        if not text:
            return None

        start = max(0.0, self._time_offset_sec)
        end = max(start, self._time_offset_sec)
        try:
            start = float(self._recognizer.start_time(self._stream))
            timestamps = self._recognizer.timestamps(self._stream)
            if timestamps:
                end = start + float(timestamps[-1])
            else:
                end = max(start, self._time_offset_sec)
        except Exception:
            # If timestamps are unavailable, keep coarse timing only.
            start = max(0.0, self._last_segment_end)
            end = max(start, self._time_offset_sec)

        gap = start - self._last_segment_end
        if self._last_segment_end > 0 and gap > SPEAKER_GAP_THRESHOLD:
            self._current_speaker += 1
        self._last_segment_end = max(self._last_segment_end, end)

        return {
            "text": text,
            "start": start,
            "end": end,
            "language": "zh",
            "speaker": self._current_speaker,
        }
