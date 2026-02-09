"""感知層 - Faster-Whisper / Whisper.cpp STT 引擎，含狀態機管理與說話者分段"""

import threading
import numpy as np
from enum import Enum
from faster_whisper import WhisperModel
import os
import wave
import tempfile
import subprocess

DEFAULT_WHISPER_CPP_BIN = "/Users/minashih/tools/whisper.cpp/bin/whisper-cli"
DEFAULT_WHISPER_CPP_MODEL_DIR = "/Users/minashih/models/whisper.cpp"

class State(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPED = "stopped"


INITIAL_PROMPT = ""
PROMPT_ECHO_PHRASES = [
    "請以繁體中文為主",
    "請以繁體中文為主，保留英文專有名詞與縮寫，並使用正確標點符號",
    "請以繁體中文為主，並使用正確標點符號",
    "以下是一段中英語",
    "以下是一段中英混合的會議對話逐字稿",
]

# 說話者間隔門檻（秒）：語音段落間隙超過此值視為換人說話
SPEAKER_GAP_THRESHOLD = 1.5


class STTEngine:
    TRANSCRIBE_INTERVAL_MS = 5000  # 每累積 5 秒觸發轉寫
    PARTIAL_INTERVAL_MS = 1000     # partial 更新頻率
    PARTIAL_WINDOW_MS = 2000       # partial 轉寫視窗大小
    SAMPLE_RATE = 16000

    def __init__(self, model_size="small", engine="faster-whisper"):
        self._lock = threading.Lock()
        self._state = State.IDLE
        self._model_size = model_size
        self._engine = engine
        self._pcm_buffer = np.array([], dtype=np.float32)
        self._time_offset_sec = 0.0
        self._speaker_counter = 0
        self._current_speaker = 1
        self._last_segment_end = 0.0  # 上一段結束時間（秒）
        self._locked_language = None
        self._last_partial_emit_sec = 0.0
        self._load_model(model_size, engine)

    @property
    def state(self) -> str:
        with self._lock:
            return self._state.value

    @property
    def model_size(self) -> str:
        with self._lock:
            return self._model_size

    @property
    def engine(self) -> str:
        with self._lock:
            return self._engine

    def set_model(self, engine: str, model_size: str) -> bool:
        """切換引擎與模型大小（僅限 idle 狀態）。成功回傳 True。"""
        with self._lock:
            if self._state != State.IDLE:
                return False
            if model_size == self._model_size and engine == self._engine:
                return True
            self._model_size = model_size
            self._engine = engine
        self._load_model(model_size, engine)
        return True

    def start(self) -> str:
        with self._lock:
            if self._state != State.IDLE:
                return self._state.value
            self._pcm_buffer = np.array([], dtype=np.float32)
            self._time_offset_sec = 0.0
            self._speaker_counter = 0
            self._current_speaker = 1
            self._last_segment_end = 0.0
            self._locked_language = None
            self._last_partial_emit_sec = 0.0
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
            segments = self._transcribe_remaining()
            self._state = State.IDLE
            return "stopped", segments

    def reset(self):
        with self._lock:
            self._pcm_buffer = np.array([], dtype=np.float32)
            self._time_offset_sec = 0.0
            self._speaker_counter = 0
            self._current_speaker = 1
            self._last_segment_end = 0.0
            self._locked_language = None
            self._last_partial_emit_sec = 0.0
            self._state = State.IDLE

    def feed_audio(self, chunk: bytes) -> dict:
        """接收 16kHz PCM int16 chunk，累積後觸發轉寫。回傳 final/partial。"""
        with self._lock:
            if self._state != State.RECORDING:
                return {"final": [], "partial": ""}

            self._append_pcm_chunk(chunk)
            total_duration_ms = self._get_buffer_duration_ms()

            partial_text = ""
            if total_duration_ms >= self.PARTIAL_INTERVAL_MS:
                partial_text = self._maybe_partial()

            if total_duration_ms < self.TRANSCRIBE_INTERVAL_MS:
                return {"final": [], "partial": partial_text}

            final_segments = self._do_transcribe()
            return {"final": final_segments, "partial": ""}

    def _get_buffer_duration_ms(self) -> int:
        """估算緩衝區音頻時長（毫秒）"""
        if self._pcm_buffer.size == 0:
            return 0
        return int((self._pcm_buffer.size / self.SAMPLE_RATE) * 1000)

    def _append_pcm_chunk(self, chunk: bytes) -> None:
        """加入 PCM chunk（int16）到緩衝區"""
        try:
            pcm16 = np.frombuffer(chunk, dtype=np.int16)
            if pcm16.size == 0:
                return
            pcm32 = pcm16.astype(np.float32) / 32768.0
            self._pcm_buffer = np.concatenate([self._pcm_buffer, pcm32])
        except Exception as e:
            print(f"[STT] PCM 解析錯誤: {e}", flush=True)

    def _do_transcribe(self) -> list[dict]:
        """對累積的音頻執行轉寫"""
        try:
            if self._pcm_buffer.size == 0:
                return []

            if self._engine == "whisper.cpp":
                return self._transcribe_whisper_cpp()

            transcribe_kwargs = {
                "vad_filter": True,
                "beam_size": 5,
            }
            if self._locked_language:
                transcribe_kwargs["language"] = self._locked_language

            segments, info = self._model.transcribe(self._pcm_buffer, **transcribe_kwargs)

            results = []
            if not self._locked_language and info.language and info.language != "ko":
                self._locked_language = "zh"
                print(f"[STT] 語言已鎖定為 zh (偵測: {info.language})", flush=True)
            for seg in segments:
                seg_start = seg.start + self._time_offset_sec
                seg_end = seg.end + self._time_offset_sec
                # 偵測說話者切換：段落間隙超過門檻則視為換人
                gap = seg_start - self._last_segment_end
                if self._last_segment_end > 0 and gap > SPEAKER_GAP_THRESHOLD:
                    self._current_speaker += 1

                self._last_segment_end = seg_end

                text = seg.text.strip()
                if text:
                    if any(phrase in text for phrase in PROMPT_ECHO_PHRASES):
                        continue
                    results.append({
                        "text": text,
                        "start": seg_start,
                        "end": seg_end,
                        "language": info.language,
                        "speaker": self._current_speaker,
                    })
            self._time_offset_sec += self._pcm_buffer.size / self.SAMPLE_RATE
            self._pcm_buffer = np.array([], dtype=np.float32)
            return results

        except Exception as e:
            print(f"[STT] 轉寫錯誤: {e}")
            return []

    def _transcribe_remaining(self) -> list[dict]:
        """轉寫剩餘未處理的音頻"""
        if self._pcm_buffer.size == 0:
            return []
        return self._do_transcribe()

    def _maybe_partial(self) -> str:
        """低延遲 partial 轉寫（不清空 buffer）。"""
        if self._engine == "whisper.cpp":
            return ""
        total_sec = self._pcm_buffer.size / self.SAMPLE_RATE
        if total_sec - self._last_partial_emit_sec < (self.PARTIAL_INTERVAL_MS / 1000.0):
            return ""
        window_samples = int((self.PARTIAL_WINDOW_MS / 1000.0) * self.SAMPLE_RATE)
        if window_samples <= 0:
            return ""
        window = self._pcm_buffer[-window_samples:] if self._pcm_buffer.size > window_samples else self._pcm_buffer

        try:
            transcribe_kwargs = {
                "vad_filter": True,
                "beam_size": 3,
            }
            if self._locked_language:
                transcribe_kwargs["language"] = self._locked_language

            segments, info = self._model.transcribe(window, **transcribe_kwargs)
            text_parts = []
            if not self._locked_language and info.language and info.language != "ko":
                self._locked_language = "zh"
            for seg in segments:
                text = seg.text.strip()
                if text and not any(phrase in text for phrase in PROMPT_ECHO_PHRASES):
                    text_parts.append(text)
            self._last_partial_emit_sec = total_sec
            return " ".join(text_parts).strip()
        except Exception:
            return ""

    def _load_model(self, model_size: str, engine: str) -> None:
        if engine == "whisper.cpp":
            self._whisper_cpp_bin = os.getenv("WHISPER_CPP_BIN", "").strip() or DEFAULT_WHISPER_CPP_BIN
            self._whisper_cpp_model_dir = os.getenv("WHISPER_CPP_MODEL_DIR", "").strip() or DEFAULT_WHISPER_CPP_MODEL_DIR
            if not self._whisper_cpp_bin or not os.path.isfile(self._whisper_cpp_bin):
                raise RuntimeError("WHISPER_CPP_BIN 未設定或檔案不存在")
            if not self._whisper_cpp_model_dir or not os.path.isdir(self._whisper_cpp_model_dir):
                raise RuntimeError("WHISPER_CPP_MODEL_DIR 未設定或目錄不存在")
            self._whisper_cpp_model_path = os.path.join(
                self._whisper_cpp_model_dir, f"ggml-{model_size}.bin"
            )
            if not os.path.isfile(self._whisper_cpp_model_path):
                raise RuntimeError(f"找不到模型: {self._whisper_cpp_model_path}")
            print(f"[STT] 使用 Whisper.cpp: {self._whisper_cpp_bin}")
            print(f"[STT] 模型: {self._whisper_cpp_model_path}")
            return

        print(f"[STT] 載入 Whisper 模型: {model_size} (CPU, int8)...")
        self._model = WhisperModel(
            model_size, device="cpu", compute_type="int8"
        )
        print("[STT] 模型載入完成")

    def _transcribe_whisper_cpp(self) -> list[dict]:
        wav_path = None
        out_prefix = None
        temp_dir = None
        try:
            pcm16 = (self._pcm_buffer * 32768.0).clip(-32768, 32767).astype(np.int16)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
                wav_path = wf.name
            with wave.open(wav_path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(self.SAMPLE_RATE)
                w.writeframes(pcm16.tobytes())

            temp_dir = tempfile.TemporaryDirectory(prefix="whispercpp_")
            out_prefix = os.path.join(temp_dir.name, "out")
            threads = os.getenv("WHISPER_CPP_THREADS", "").strip()
            cmd = [
                self._whisper_cpp_bin,
                "-m", self._whisper_cpp_model_path,
                "-f", wav_path,
                "-otxt",
                "-of", out_prefix,
                "-l", "auto",
            ]
            if threads.isdigit():
                cmd.extend(["-t", threads])
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            txt_path = f"{out_prefix}.txt"
            text = ""
            if os.path.isfile(txt_path):
                with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()

            seg_start = self._time_offset_sec
            seg_end = self._time_offset_sec + (self._pcm_buffer.size / self.SAMPLE_RATE)
            self._time_offset_sec += self._pcm_buffer.size / self.SAMPLE_RATE
            self._pcm_buffer = np.array([], dtype=np.float32)

            if not text:
                return []
            return [{
                "text": text,
                "start": seg_start,
                "end": seg_end,
                "language": "",
                "speaker": self._current_speaker,
            }]
        except Exception as e:
            print(f"[STT] Whisper.cpp 轉寫錯誤: {e}")
            return []
        finally:
            for path in (wav_path, f"{out_prefix}.txt" if out_prefix else None):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
            if temp_dir:
                try:
                    temp_dir.cleanup()
                except Exception:
                    pass
