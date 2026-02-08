"""感知層 - Faster-Whisper STT 引擎，含狀態機管理與說話者分段"""

import threading
import numpy as np
from enum import Enum
from faster_whisper import WhisperModel


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
    SAMPLE_RATE = 16000

    def __init__(self, model_size="small"):
        self._lock = threading.Lock()
        self._state = State.IDLE
        self._pcm_buffer = np.array([], dtype=np.float32)
        self._time_offset_sec = 0.0
        self._speaker_counter = 0
        self._current_speaker = 1
        self._last_segment_end = 0.0  # 上一段結束時間（秒）
        self._locked_language = None
        print(f"[STT] 載入 Whisper 模型: {model_size} (CPU, int8)...")
        self._model = WhisperModel(
            model_size, device="cpu", compute_type="int8"
        )
        print("[STT] 模型載入完成")

    @property
    def state(self) -> str:
        with self._lock:
            return self._state.value

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
