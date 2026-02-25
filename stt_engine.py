"""感知層 - 只使用 Sherpa-ONNX 的 STT 引擎，含狀態機管理"""

import os
import threading
from enum import Enum

import numpy as np

DEFAULT_SHERPA_ONNX_MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "models",
    "sherpa-onnx",
    "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
)


class State(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPED = "stopped"


class STTEngine:
    SAMPLE_RATE = 16000
    WARMUP_SECONDS = float(os.getenv("SHERPA_ONNX_WARMUP_SECONDS", "0"))

    def __init__(self, model_size="zipformer-small-zh-en", engine="sherpa-onnx", model_id: str | None = None):
        self._lock = threading.Lock()
        self._state = State.IDLE
        self._model_size = model_size
        self._engine = engine
        self._model_id = model_id or model_size
        self._language_mode = "auto"

        self._last_error = ""
        self._error_lock = threading.Lock()

        self._sherpa_recognizer = None
        self._sherpa_stream = None
        self._sherpa_last_text = ""
        self._lazy_init = os.getenv("STT_LAZY_INIT", "1").strip().lower() not in {"0", "false", "no"}
        if not self._lazy_init:
            self._load_model(model_size, engine, model_id=self._model_id)

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

    @property
    def language_mode(self) -> str:
        with self._lock:
            return self._language_mode

    def set_model(self, engine: str, model_size: str, model_id: str | None = None) -> bool:
        """切換模型（僅限 idle 狀態，且僅支援 sherpa-onnx）。"""
        with self._lock:
            if self._state != State.IDLE:
                return False
            if engine != "sherpa-onnx":
                raise RuntimeError("目前僅支援 sherpa-onnx")
            if model_size == self._model_size and (model_id or model_size) == self._model_id:
                return True
            self._model_size = model_size
            self._engine = engine
            self._model_id = model_id or model_size
        self._load_model(model_size, engine, model_id=self._model_id)
        return True

    def set_language_mode(self, mode: str) -> bool:
        """切換語言模式（僅限 idle 狀態）。"""
        with self._lock:
            if self._state != State.IDLE:
                return False
            self._language_mode = mode or "auto"
        return True

    def start(self) -> str:
        needs_load = False
        with self._lock:
            if self._state != State.IDLE:
                return self._state.value
            needs_load = self._sherpa_recognizer is None

        if needs_load:
            try:
                self._load_model(self.model_size, self.engine, model_id=self._model_id)
            except Exception as e:
                self._set_error(f"STT 模型載入失敗: {e}")
                with self._lock:
                    return self._state.value

        with self._lock:
            if self._sherpa_recognizer is not None:
                self._sherpa_stream = self._sherpa_recognizer.create_stream()
                self._sherpa_last_text = ""
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
        """停止錄音並回傳最後結果。"""
        with self._lock:
            if self._state not in (State.RECORDING, State.PAUSED):
                return self._state.value, []
            self._state = State.STOPPED

            segments = []
            if self._sherpa_recognizer is not None and self._sherpa_stream is not None:
                result = self._sherpa_recognizer.get_result(self._sherpa_stream)
                text = result.text if hasattr(result, "text") else str(result)
                if text:
                    segments.append({"text": text})
                self._sherpa_recognizer.reset(self._sherpa_stream)
                self._sherpa_last_text = ""

            self._state = State.IDLE
            return "stopped", segments

    def reset(self):
        with self._lock:
            if self._sherpa_recognizer is not None:
                self._sherpa_stream = self._sherpa_recognizer.create_stream()
            self._sherpa_last_text = ""
            self._state = State.IDLE

    def feed_audio(self, chunk: bytes) -> dict:
        """接收 16kHz PCM int16 chunk。回傳 {final, partial}。"""
        with self._lock:
            if self._state != State.RECORDING:
                return {"final": [], "partial": ""}
            return self._feed_audio_sherpa(chunk)

    def get_and_clear_error(self) -> str:
        with self._error_lock:
            msg = self._last_error
            self._last_error = ""
            return msg

    def _set_error(self, message: str) -> None:
        with self._error_lock:
            self._last_error = message

    def _load_model(self, model_size: str, engine: str, model_id: str | None = None) -> None:
        if engine != "sherpa-onnx":
            raise RuntimeError("目前僅支援 sherpa-onnx")
        self._load_sherpa_onnx()

    def _load_sherpa_onnx(self) -> None:
        try:
            import sherpa_onnx  # type: ignore
        except Exception as e:
            raise RuntimeError(f"sherpa-onnx 未安裝: {e}")

        model_dir = os.getenv("SHERPA_ONNX_MODEL_DIR", "").strip() or DEFAULT_SHERPA_ONNX_MODEL_DIR
        encoder = self._pick_model_file(
            model_dir,
            [
                "encoder-epoch-99-avg-1.onnx",
                "encoder-epoch-99-avg-1.int8.onnx",
            ],
        )
        decoder = self._pick_model_file(
            model_dir,
            [
                "decoder-epoch-99-avg-1.onnx",
            ],
        )
        joiner = self._pick_model_file(
            model_dir,
            [
                "joiner-epoch-99-avg-1.onnx",
                "joiner-epoch-99-avg-1.int8.onnx",
            ],
        )
        tokens = self._pick_model_file(
            model_dir,
            [
                "tokens.txt",
            ],
        )
        bpe_vocab = self._pick_model_file(
            model_dir,
            [
                "bpe.model",
                "bpe.vocab",
            ],
        )

        num_threads = int(os.getenv("SHERPA_ONNX_THREADS", "2"))
        self._sherpa_recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            num_threads=num_threads,
            sample_rate=self.SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=0.8,
            rule2_min_trailing_silence=0.5,
            rule3_min_utterance_length=6.0,
            provider="cpu",
            modeling_unit="cjkchar+bpe",
            bpe_vocab=bpe_vocab,
        )
        self._sherpa_stream = self._sherpa_recognizer.create_stream()
        self._sherpa_last_text = ""
        self._warmup_sherpa_onnx()
        print("[STT] sherpa-onnx 模型載入完成")

    def _warmup_sherpa_onnx(self) -> None:
        """預熱一次推論，降低第一段逐字稿冷啟動延遲。"""
        if self._sherpa_recognizer is None:
            return
        try:
            stream = self._sherpa_recognizer.create_stream()
            warmup_samples = int(self.SAMPLE_RATE * self.WARMUP_SECONDS)
            if warmup_samples <= 0:
                return
            silence = np.zeros(warmup_samples, dtype=np.float32)
            stream.accept_waveform(self.SAMPLE_RATE, silence)
            while self._sherpa_recognizer.is_ready(stream):
                self._sherpa_recognizer.decode_stream(stream)
            self._sherpa_recognizer.get_result(stream)
            self._sherpa_recognizer.reset(stream)
        except Exception:
            # 預熱失敗不影響主流程
            pass

    def _pick_model_file(self, model_dir: str, candidates: list[str]) -> str:
        for name in candidates:
            path = os.path.join(model_dir, name)
            if os.path.isfile(path):
                return path
        raise RuntimeError(f"找不到模型檔: {candidates} in {model_dir}")

    def _feed_audio_sherpa(self, chunk: bytes) -> dict:
        if self._sherpa_recognizer is None or self._sherpa_stream is None:
            return {"final": [], "partial": ""}
        try:
            if not isinstance(chunk, (bytes, bytearray, memoryview)):
                return {"final": [], "partial": ""}
            if len(chunk) < 2:
                return {"final": [], "partial": ""}
            if len(chunk) % 2 != 0:
                # PCM16 must be aligned to 2 bytes; drop trailing partial byte.
                chunk = chunk[:-1]
                if not chunk:
                    return {"final": [], "partial": ""}
            pcm16 = np.frombuffer(chunk, dtype=np.int16)
            if pcm16.size == 0:
                return {"final": [], "partial": ""}
            samples = pcm16.astype(np.float32) / 32768.0
            self._sherpa_stream.accept_waveform(self.SAMPLE_RATE, samples)

            while self._sherpa_recognizer.is_ready(self._sherpa_stream):
                self._sherpa_recognizer.decode_stream(self._sherpa_stream)

            result = self._sherpa_recognizer.get_result(self._sherpa_stream)
            text = result.text if hasattr(result, "text") else str(result)

            partial = ""
            if text and text != self._sherpa_last_text:
                partial = text
                self._sherpa_last_text = text

            if self._sherpa_recognizer.is_endpoint(self._sherpa_stream):
                final_segments = []
                if text:
                    final_segments.append({"text": text})
                self._sherpa_recognizer.reset(self._sherpa_stream)
                self._sherpa_last_text = ""
                return {"final": final_segments, "partial": ""}

            return {"final": [], "partial": partial}
        except Exception as e:
            self._set_error(f"sherpa-onnx 轉寫錯誤: {e}")
            return {"final": [], "partial": ""}
