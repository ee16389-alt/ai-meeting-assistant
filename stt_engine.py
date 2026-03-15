"""感知層 - Faster-Whisper STT 引擎（medium int8，台灣腔中英混切）"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from enum import Enum
from pathlib import Path

import numpy as np

try:
    from faster_whisper import WhisperModel  # type: ignore
    _fw_import_error = ""
except Exception as _e:
    WhisperModel = None  # type: ignore[assignment]
    _fw_import_error = str(_e)

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


SPEAKER_GAP_THRESHOLD = 1.5

# Whisper 初始提示：引導輸出繁體中文台灣用字，適應中英混切
TAIWAN_PROMPT = "以下是台灣繁體中文會議紀錄，包含商業術語與英文詞彙。"

# Whisper 常見幻覺句（靜音時模型虛構的輸出，或 prompt 回音）
_HALLUCINATION_PATTERNS = re.compile(
    r"(thank you for watching|字幕由|請訂閱|訂閱頻道|點讚|不吝|掌聲|♪|♫|music|ambient|"
    r"by\s+\w+\s+caption|subtitles?\s+by|"
    r"台灣繁體中文會議紀錄|包含商業術語與英文詞彙|以下是台灣|"
    r"翻譯中|翻唱中|字幕製作)",
    re.IGNORECASE,
)


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


def _find_whisper_model_dir() -> Path | None:
    """尋找本地 Faster-Whisper 模型目錄（含 model.bin）"""
    env_dir = os.environ.get("AMA_WHISPER_DIR", "").strip()
    if env_dir:
        p = Path(env_dir).expanduser()
        if p.is_dir() and (p / "model.bin").exists():
            return p

    cfg = _load_model_pack_config()
    model_dir_name = str(cfg.get("whisperModelDirName", "faster-whisper-medium")).strip()

    candidates: list[Path] = []
    resources = _resources_root()
    if resources:
        candidates.append(resources / "models" / "whisper" / model_dir_name)
        candidates.append(resources / "models" / "whisper")

    root = _project_root()
    for base in (
        root / "desktop" / "models" / "whisper",
        root / "models" / "whisper",
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
            base = anc / "models" / "whisper"
            candidates.append(base / model_dir_name)
            candidates.append(base)

    for p in candidates:
        if not p.exists():
            continue
        if p.is_dir() and (p / "model.bin").exists():
            return p
        if p.is_dir():
            nested = sorted(
                [d for d in p.iterdir() if d.is_dir() and (d / "model.bin").exists()]
            )
            if nested:
                return nested[0]
    return None


class STTEngine:
    TRANSCRIBE_INTERVAL_MS = 4000   # 每 4 秒批次辨識（small 模型夠快，兼顧即時感與效能）
    SAMPLE_RATE = 16000
    SILENCE_THRESHOLD = 0.003

    def __init__(self, model_size: str = "medium"):
        if WhisperModel is None:
            raise RuntimeError(f"faster_whisper 無法載入: {_fw_import_error}")

        self._lock = threading.Lock()
        self._state = State.IDLE
        self._pcm_buffer = np.array([], dtype=np.float32)
        self._time_offset_sec = 0.0
        self._current_speaker = 1
        self._last_segment_end = 0.0
        self._last_partial_text = ""
        self._last_audio_rms = 0.0

        self._model = self._create_model(model_size)
        print("[STT] Faster-Whisper 模型載入完成", flush=True)

    def _create_model(self, model_size: str) -> "WhisperModel":
        local_dir = _find_whisper_model_dir()
        if local_dir:
            print(f"[STT] 使用本地模型: {local_dir}", flush=True)
            model_path = str(local_dir)
        else:
            # 開發模式：自動從 HuggingFace 下載（由 HF_HOME 控制快取位置）
            model_path = f"Systran/faster-whisper-{model_size}"
            print(f"[STT] 本地模型未找到，下載 {model_path}", flush=True)

        cpu_count = os.cpu_count() or 4
        # 保留至少 2 個核心給 Flask/SocketIO，避免暫停/停止無反應
        cpu_threads = max(2, min(6, cpu_count // 2))
        return WhisperModel(
            model_path,
            device="cpu",
            compute_type="int8",
            num_workers=1,
            cpu_threads=cpu_threads,
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
            self._pcm_buffer = np.array([], dtype=np.float32)
            self._time_offset_sec = 0.0
            self._current_speaker = 1
            self._last_segment_end = 0.0
            self._last_partial_text = ""
            self._last_audio_rms = 0.0
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

    def request_stop(self) -> np.ndarray:
        """原子操作：立即設為 IDLE 並取走剩餘 buffer，供非同步最終轉寫。"""
        with self._lock:
            if self._state not in (State.RECORDING, State.PAUSED):
                return np.array([], dtype=np.float32)
            remaining = self._pcm_buffer.copy()
            self._pcm_buffer = np.array([], dtype=np.float32)
            self._state = State.IDLE
            return remaining

    def stop(self) -> tuple[str, list[dict]]:
        """同步停止（相容舊介面）。"""
        remaining = self.request_stop()
        segments = self.transcribe_audio(remaining)
        return "stopped", segments

    def reset(self):
        with self._lock:
            self._pcm_buffer = np.array([], dtype=np.float32)
            self._time_offset_sec = 0.0
            self._current_speaker = 1
            self._last_segment_end = 0.0
            self._last_partial_text = ""
            self._last_audio_rms = 0.0
            self._state = State.IDLE

    # ── 音頻處理 ──────────────────────────────────────────

    def feed_audio(self, chunk: bytes) -> list[dict]:
        with self._lock:
            if self._state != State.RECORDING:
                return []
            self._append_pcm_chunk(chunk)
            duration_ms = self._get_buffer_duration_ms()
            if duration_ms < self.TRANSCRIBE_INTERVAL_MS:
                self._last_partial_text = "辨識中..."
                return []
            audio = self._pcm_buffer.copy()
            self._pcm_buffer = np.array([], dtype=np.float32)
            self._last_partial_text = ""

        # 鎖外執行推論，不阻塞其他 feed_audio
        return self._do_transcribe(audio)

    def transcribe_audio(self, audio: np.ndarray) -> list[dict]:
        """對外部傳入的 buffer 進行轉寫（供非同步 stop 使用）。"""
        if audio.size == 0:
            return []
        # 裁剪尾端靜音
        nz = np.nonzero(audio)[0]
        if nz.size == 0:
            return []
        keep = min(int(nz[-1]) + self.SAMPLE_RATE // 2, audio.size)
        audio = audio[:keep]
        return self._do_transcribe(audio)

    def _append_pcm_chunk(self, chunk: bytes) -> None:
        try:
            pcm16 = np.frombuffer(chunk, dtype=np.int16)
            if pcm16.size == 0:
                return
            pcm32 = pcm16.astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(np.square(pcm32)))) if pcm32.size else 0.0
            self._last_audio_rms = rms
            if rms < self.SILENCE_THRESHOLD:
                self._pcm_buffer = np.concatenate(
                    [self._pcm_buffer, np.zeros(pcm32.size, dtype=np.float32)]
                )
                return
            self._pcm_buffer = np.concatenate([self._pcm_buffer, pcm32])
        except Exception as e:
            print(f"[STT] PCM 解析錯誤: {e}", flush=True)

    def _get_buffer_duration_ms(self) -> int:
        if self._pcm_buffer.size == 0:
            return 0
        return int((self._pcm_buffer.size / self.SAMPLE_RATE) * 1000)

    def _do_transcribe(self, audio: np.ndarray) -> list[dict]:
        try:
            segments_iter, info = self._model.transcribe(
                audio,
                language="zh",
                beam_size=1,        # greedy search，速度提升 3-5x
                best_of=1,
                temperature=0.0,
                initial_prompt=TAIWAN_PROMPT,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=400,
                    speech_pad_ms=200,
                    threshold=0.5,
                ),
                condition_on_previous_text=False,  # 避免 prompt 回音
                word_timestamps=False,
            )

            results = []
            for seg in segments_iter:
                text = seg.text.strip()
                if not text or _HALLUCINATION_PATTERNS.search(text):
                    continue
                text = _to_traditional(text)
                text = self._filter_repetitions(text)
                if not text:
                    continue

                start = self._time_offset_sec + seg.start
                end = self._time_offset_sec + seg.end
                gap = start - self._last_segment_end
                if self._last_segment_end > 0 and gap > SPEAKER_GAP_THRESHOLD:
                    self._current_speaker += 1
                self._last_segment_end = max(self._last_segment_end, end)

                results.append({
                    "text": text,
                    "start": start,
                    "end": end,
                    "language": getattr(info, "language", "zh"),
                    "speaker": self._current_speaker,
                })

            self._time_offset_sec += audio.size / self.SAMPLE_RATE
            return results

        except Exception as e:
            print(f"[STT] Faster-Whisper 辨識錯誤: {e}", flush=True)
            self._time_offset_sec += audio.size / self.SAMPLE_RATE
            return []

    @staticmethod
    def _filter_repetitions(text: str) -> str:
        if len(text) > 3:
            text = re.sub(r'(.)\1{2,}', r'\1\1', text)            # 嗯嗯嗯嗯 → 嗯嗯
            text = re.sub(r'(.{2,20})\1{2,}', r'\1', text)        # 在於英文文化的大臺… → 一次
            # 若整段仍由單一短語高密度重複組成，整段丟棄
            for length in range(3, min(len(text) // 3 + 1, 21)):
                phrase = text[:length]
                count = text.count(phrase)
                if count >= 4:
                    remainder = text.replace(phrase, '').replace('，', '').replace(',', '')
                    if len(remainder) < length:
                        return ''
        return text.strip()
