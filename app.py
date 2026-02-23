"""Flask + SocketIO 主伺服器 - 路由與事件處理"""

import os
import time
import threading
import subprocess
from queue import Queue, Empty
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from stt_engine import STTEngine
from cognition import (
    proofread_text,
    summarize_full,
    summarize_key_points,
    extract_action_items,
    check_health,
    get_current_model,
    set_current_model,
    AVAILABLE_OLLAMA_MODELS,
    DEFAULT_OLLAMA_MODEL_KEY,
)
from utils import sanitize_meeting_name

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "meeting-assistant-secret")
# Force threading to avoid invalid async_mode in packaged runtime.
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    max_http_buffer_size=10 * 1024 * 1024,
    async_mode="threading",
)

# 單一模型（sherpa-onnx）
AVAILABLE_STT_MODELS = {
    "sherpa_zipformer_small_zh_en": {
        "label": "Sherpa-ONNX Zipformer Small (中英)",
        "size": "zipformer-small-zh-en",
        "engine": "sherpa-onnx",
    },
}
DEFAULT_STT_MODEL_KEY = "sherpa_zipformer_small_zh_en"

AVAILABLE_LANG_MODES = {
    "auto": {"label": "自動偵測"},
    "zh": {"label": "中文"},
    "en": {"label": "英文"},
    "ja": {"label": "日文"},
    "ko": {"label": "韓文"},
    "th": {"label": "泰文"},
    "tl": {"label": "菲律賓文"},
    "id": {"label": "印尼文"},
    "vi": {"label": "越南文"},
}
DEFAULT_LANG_MODE = "auto"

# 全域 STT 引擎實例
stt = STTEngine(
    model_size=AVAILABLE_STT_MODELS[DEFAULT_STT_MODEL_KEY]["size"],
    engine=AVAILABLE_STT_MODELS[DEFAULT_STT_MODEL_KEY]["engine"],
    model_id=AVAILABLE_STT_MODELS[DEFAULT_STT_MODEL_KEY].get("model_id"),
)
current_stt_model_key = DEFAULT_STT_MODEL_KEY
current_lang_mode = DEFAULT_LANG_MODE

# 會議逐字稿暫存（用於摘要與匯出）
transcript_lines: list[dict] = []
proofread_index = 0  # 追蹤下一個待校對的行號
audio_chunk_count = 0
audio_save_enabled = False
audio_file_handle = None
current_meeting_name = ""
proofread_enabled = False
traditional_enabled = False
_opencc_converter = None
_opencc_available_cache = None
transcript_lock = threading.Lock()

audio_queue: Queue[tuple[int, bytes]] = Queue(maxsize=200)
audio_queue_dropped = 0
current_session_id = 0
audio_worker_started = False

DEFAULT_SHERPA_ONNX_MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "models",
    "sherpa-onnx",
    "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
)


def _resolve_meeting_name(input_name: str = "") -> str:
    base = (input_name or "").strip()
    if not base:
        base = current_meeting_name or time.strftime("meeting_%Y%m%d_%H%M%S")
    return sanitize_meeting_name(base, time.strftime("meeting_%Y%m%d_%H%M%S"))


def _meeting_export_dir(input_name: str = "") -> str:
    meeting_name = _resolve_meeting_name(input_name)
    return os.path.join(os.path.dirname(__file__), "download", meeting_name)


def _sherpa_onnx_availability() -> tuple[bool, str]:
    model_dir = os.getenv("SHERPA_ONNX_MODEL_DIR", "").strip() or DEFAULT_SHERPA_ONNX_MODEL_DIR
    found = set(os.listdir(model_dir)) if os.path.isdir(model_dir) else set()
    if not found:
        return False, f"模型目錄不存在: {model_dir}"
    has_encoder = any(x in found for x in ("encoder-epoch-99-avg-1.int8.onnx", "encoder-epoch-99-avg-1.onnx"))
    has_joiner = any(x in found for x in ("joiner-epoch-99-avg-1.int8.onnx", "joiner-epoch-99-avg-1.onnx"))
    has_decoder = "decoder-epoch-99-avg-1.onnx" in found
    has_tokens = "tokens.txt" in found
    if not (has_encoder and has_joiner and has_decoder and has_tokens):
        return False, "Sherpa-ONNX 模型檔不完整"
    return True, ""


def _model_payload() -> dict:
    sherpa_ok, sherpa_reason = _sherpa_onnx_availability()
    current = AVAILABLE_STT_MODELS.get(
        current_stt_model_key, {"label": "Unknown", "size": "zipformer-small-zh-en", "engine": "sherpa-onnx"}
    )
    return {
        "stt_model": {
            "key": current_stt_model_key,
            "label": current["label"],
            "size": current["size"],
            "engine": current.get("engine", "sherpa-onnx"),
        },
        "language_mode": current_lang_mode,
        "language_modes": [
            {
                "key": key,
                "label": item["label"],
            }
            for key, item in AVAILABLE_LANG_MODES.items()
        ],
        "stt_models": [
            {
                "key": key,
                "label": item["label"],
                "size": item["size"],
                "engine": item.get("engine", "sherpa-onnx"),
                "model_id": item.get("model_id", item["size"]),
                "available": (item.get("engine", "sherpa-onnx") != "sherpa-onnx" or sherpa_ok),
                "reason": ("" if item.get("engine", "sherpa-onnx") != "sherpa-onnx" else sherpa_reason),
            }
            for key, item in AVAILABLE_STT_MODELS.items()
        ],
        "ollama_model": {
            "key": next(
                (
                    key for key, item in AVAILABLE_OLLAMA_MODELS.items()
                    if item["model"] == get_current_model()
                ),
                DEFAULT_OLLAMA_MODEL_KEY,
            ),
            "model": get_current_model(),
        },
        "ollama_models": [
            {
                "key": key,
                "label": item["label"],
                "model": item["model"],
            }
            for key, item in AVAILABLE_OLLAMA_MODELS.items()
        ],
    }

def _opencc_available() -> bool:
    global _opencc_available_cache
    if _opencc_available_cache is not None:
        return _opencc_available_cache
    try:
        from opencc import OpenCC  # type: ignore
        _opencc_available_cache = True
    except Exception:
        _opencc_available_cache = False
    return _opencc_available_cache

def _drain_audio_queue() -> None:
    try:
        while True:
            audio_queue.get_nowait()
    except Empty:
        return

def _emit_opencc_status() -> None:
    socketio.emit("opencc_status", {
        "available": _opencc_available(),
        "enabled": traditional_enabled,
    })

def _handle_stt_result(partial_text: str, segments: list[dict]) -> None:
    if partial_text:
        if traditional_enabled:
            partial_text = _to_traditional(partial_text)
        socketio.emit("transcript_partial", {
            "text": partial_text,
            "timestamp": time.strftime("%H:%M:%S"),
        })
    if segments:
        socketio.emit("transcript_partial_clear")
    for seg in segments:
        text = seg["text"]
        if traditional_enabled:
            text = _to_traditional(text)
        with transcript_lock:
            idx = len(transcript_lines)
            line = {
                "index": idx,
                "text": text,
                "timestamp": time.strftime("%H:%M:%S"),
                "language": seg.get("language", ""),
            }
            transcript_lines.append(line)
        socketio.emit("transcript_update", line)

        if proofread_enabled:
            idx = line["index"]
            original_text = line["text"]
            socketio.start_background_task(
                _proofread_line, idx, original_text
            )

def _audio_worker() -> None:
    while True:
        try:
            session_id, chunk = audio_queue.get(timeout=0.5)
        except Empty:
            continue
        if session_id != current_session_id:
            continue
        if stt.state != "recording":
            continue
        result = stt.feed_audio(chunk)
        if isinstance(result, dict):
            partial_text = result.get("partial", "") or ""
            segments = result.get("final", []) or []
        else:
            partial_text = ""
            segments = result
        _handle_stt_result(partial_text, segments)
        err = stt.get_and_clear_error()
        if err:
            socketio.emit("error", {"message": err})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return {"ok": True}


# ── SocketIO 事件處理 ───────────────────────────────────

@socketio.on("connect")
def handle_connect():
    _ensure_audio_worker()
    ollama_ok = check_health()
    payload = {"state": stt.state, "ollama": ollama_ok}
    payload.update(_model_payload())
    payload["opencc_available"] = _opencc_available()
    payload["traditional_enabled"] = traditional_enabled
    emit("state_changed", payload)
    _emit_opencc_status()


@socketio.on("start_recording")
def handle_start(data=None):
    global transcript_lines, proofread_index, audio_save_enabled, audio_file_handle, current_meeting_name, current_lang_mode, proofread_enabled, traditional_enabled, current_session_id
    with transcript_lock:
        transcript_lines.clear()
    proofread_index = 0
    audio_save_enabled = False
    proofread_enabled = False
    traditional_enabled = False
    if audio_file_handle:
        try:
            audio_file_handle.close()
        except Exception:
            pass
        audio_file_handle = None

    meeting_name = ""
    save_audio = False
    language_mode = None
    enable_traditional = True
    if isinstance(data, dict):
        meeting_name = (data.get("meeting_name") or "").strip()
        save_audio = bool(data.get("save_audio"))
        language_mode = (data.get("language_mode") or "").strip() or None
        if "enable_traditional" in data:
            enable_traditional = bool(data.get("enable_traditional"))
    current_meeting_name = _resolve_meeting_name(meeting_name)
    proofread_enabled = False
    traditional_enabled = enable_traditional
    current_session_id += 1
    _drain_audio_queue()

    if save_audio:
        export_dir = os.path.join(os.path.dirname(__file__), "download", current_meeting_name)
        os.makedirs(export_dir, exist_ok=True)
        audio_path = os.path.join(export_dir, "audio.webm")
        audio_file_handle = open(audio_path, "wb")
        audio_save_enabled = True

    if language_mode and language_mode in AVAILABLE_LANG_MODES:
        stt.set_language_mode(language_mode)
        current_lang_mode = language_mode
    state = stt.start()
    emit("state_changed", {
        "state": state,
        "opencc_available": _opencc_available(),
        "traditional_enabled": traditional_enabled,
    })
    _emit_opencc_status()


@socketio.on("audio_chunk")
def handle_audio_chunk(data):
    """接收二進位音頻 chunk"""
    global audio_chunk_count, audio_queue_dropped
    if isinstance(data, dict):
        chunk = data.get("chunk", b"")
    else:
        chunk = data
    if isinstance(chunk, list):
        try:
            chunk = bytes(chunk)
        except Exception:
            chunk = b""
    try:
        size = len(chunk) if chunk is not None else 0
    except Exception:
        size = 0
    if size == 0:
        print("[STT] 收到空音訊 chunk", flush=True)
        return {"ok": False, "reason": "empty"}
    if size < 16:
        print(f"[STT] 收到過小 chunk: {size} bytes", flush=True)
    audio_chunk_count += 1
    print(f"[STT] 收到音訊 chunk: {size} bytes (count={audio_chunk_count})", flush=True)

    try:
        audio_queue.put_nowait((current_session_id, chunk))
    except Exception:
        audio_queue_dropped += 1
        return {"ok": False, "reason": "queue_full", "dropped": audio_queue_dropped}
    return {"ok": True, "size": size, "count": audio_chunk_count}


@socketio.on("audio_record_chunk")
def handle_audio_record_chunk(data):
    """接收 webm 音頻 chunk（用於儲存錄音檔）"""
    global audio_file_handle, audio_save_enabled
    if not audio_save_enabled or not audio_file_handle:
        return
    if isinstance(data, dict):
        chunk = data.get("chunk", b"")
    else:
        chunk = data
    if isinstance(chunk, list):
        try:
            chunk = bytes(chunk)
        except Exception:
            chunk = b""
    if not chunk:
        return
    try:
        audio_file_handle.write(chunk)
        audio_file_handle.flush()
    except Exception:
        pass


@socketio.on("audio_recording_done")
def handle_audio_recording_done():
    """錄音檔寫入完成"""
    global audio_file_handle, audio_save_enabled
    if audio_file_handle:
        try:
            audio_file_handle.close()
        except Exception:
            pass
    audio_file_handle = None
    audio_save_enabled = False


@socketio.on("pause_recording")
def handle_pause():
    state = stt.pause()
    emit("state_changed", {"state": state})


@socketio.on("resume_recording")
def handle_resume():
    state = stt.resume()
    emit("state_changed", {"state": state})


@socketio.on("stop_recording")
def handle_stop():
    global current_session_id
    handle_audio_recording_done()
    current_session_id += 1
    _drain_audio_queue()
    state, final_segments = stt.stop()
    if final_segments:
        socketio.emit("transcript_partial_clear")
    for seg in final_segments:
        text = seg["text"]
        if traditional_enabled:
            text = _to_traditional(text)
        with transcript_lock:
            idx = len(transcript_lines)
            line = {
                "index": idx,
                "text": text,
                "timestamp": time.strftime("%H:%M:%S"),
                "language": seg.get("language", ""),
            }
            transcript_lines.append(line)
        socketio.emit("transcript_update", line)

        if proofread_enabled:
            idx = line["index"]
            original_text = line["text"]
            socketio.start_background_task(
                _proofread_line, idx, original_text
            )

    socketio.emit("state_changed", {"state": "idle"})


@socketio.on("set_stt_model")
def handle_set_stt_model(data):
    global current_stt_model_key
    model_key = ""
    if isinstance(data, dict):
        model_key = (data.get("model_key") or "").strip()
    if model_key not in AVAILABLE_STT_MODELS:
        emit("error", {"message": "不支援的模型選項"})
        return
    if stt.state != "idle":
        emit("error", {"message": "錄音中無法切換模型，請先停止再切換。"})
        return
    model_size = AVAILABLE_STT_MODELS[model_key]["size"]
    model_engine = AVAILABLE_STT_MODELS[model_key].get("engine", "sherpa-onnx")
    model_id = AVAILABLE_STT_MODELS[model_key].get("model_id", model_size)
    try:
        ok = stt.set_model(model_engine, model_size, model_id=model_id)
    except Exception as e:
        emit("error", {"message": f"模型切換失敗：{e}"})
        return
    if not ok:
        emit("error", {"message": "目前狀態無法切換模型"})
        return

    current_stt_model_key = model_key
    emit("stt_model_changed", _model_payload())


@socketio.on("set_language_mode")
def handle_set_language_mode(data):
    global current_lang_mode
    mode = ""
    if isinstance(data, dict):
        mode = (data.get("language_mode") or "").strip()
    if mode not in AVAILABLE_LANG_MODES:
        emit("error", {"message": "不支援的語言模式"})
        return
    if not stt.set_language_mode(mode):
        emit("error", {"message": "錄音中無法切換語言，請先停止再切換。"})
        return
    current_lang_mode = mode
    emit("language_mode_changed", _model_payload())


@socketio.on("set_ollama_model")
def handle_set_ollama_model(data):
    model_key = ""
    if isinstance(data, dict):
        model_key = (data.get("model_key") or "").strip()
    if model_key not in AVAILABLE_OLLAMA_MODELS:
        emit("error", {"message": "不支援的 AI 摘要模型"})
        return
    model = AVAILABLE_OLLAMA_MODELS[model_key]["model"]
    set_current_model(model)
    emit(
        "ollama_model_changed",
        {
            "ollama_model": {
                "key": model_key,
                "model": model,
            },
            "ollama_models": [
                {"key": key, "label": item["label"], "model": item["model"]}
                for key, item in AVAILABLE_OLLAMA_MODELS.items()
            ],
        },
    )


@socketio.on("request_summary")
def handle_summary(data):
    mode = "full"
    transcript_override = ""
    if isinstance(data, dict):
        mode = data.get("mode", "full")
        transcript_override = (data.get("transcript_override") or "").strip()
    with transcript_lock:
        full_text = transcript_override or "\n".join(
            line.get("proofread", line["text"]) for line in transcript_lines
        )

    if not full_text.strip():
        emit("error", {"message": "尚無逐字稿內容可供摘要"})
        return

    socketio.start_background_task(_generate_summary, mode, full_text)


@socketio.on("enhance_transcript")
def handle_enhance_transcript():
    with transcript_lock:
        has_text = any((line.get("text") or "").strip() for line in transcript_lines)
    if not has_text:
        emit("error", {"message": "尚無逐字稿內容可供處理"})
        return
    socketio.start_background_task(_enhance_transcript)


@socketio.on("export_meeting")
def handle_export(data):
    meeting_name = ""
    transcript_override = ""
    if isinstance(data, dict):
        meeting_name = (data.get("meeting_name") or "").strip()
        transcript_override = (data.get("transcript_override") or "").strip()
    meeting_name = _resolve_meeting_name(meeting_name)
    socketio.start_background_task(_export_meeting, meeting_name, transcript_override)


@socketio.on("export_summary")
def handle_export_summary(data):
    meeting_name = ""
    mode = "full"
    transcript_override = ""
    if isinstance(data, dict):
        meeting_name = (data.get("meeting_name") or "").strip()
        mode = data.get("mode", "full")
        transcript_override = (data.get("transcript_override") or "").strip()
    meeting_name = _resolve_meeting_name(meeting_name)
    socketio.start_background_task(_export_summary, meeting_name, mode, transcript_override)


@socketio.on("get_storage_path")
def handle_get_storage_path(data=None):
    meeting_name = ""
    if isinstance(data, dict):
        meeting_name = (data.get("meeting_name") or "").strip()
    emit("storage_path", {"path": _meeting_export_dir(meeting_name)})


@socketio.on("open_storage_folder")
def handle_open_storage_folder(data=None):
    meeting_name = ""
    if isinstance(data, dict):
        meeting_name = (data.get("meeting_name") or "").strip()
    folder = _meeting_export_dir(meeting_name)
    os.makedirs(folder, exist_ok=True)
    try:
        subprocess.Popen(["open", folder])
        emit("folder_opened", {"path": folder})
    except Exception as e:
        emit("error", {"message": f"無法開啟資料夾: {e}"})


# ── 背景任務 ───────────────────────────────────────────

def _to_traditional(text: str) -> str:
    """簡轉繁（需 opencc）。若不可用，回傳原文並回報一次錯誤。"""
    global traditional_enabled, _opencc_converter, _opencc_available_cache
    if not text:
        return text
    if _opencc_converter is None:
        try:
            from opencc import OpenCC  # type: ignore
            _opencc_converter = OpenCC("s2t")
            _opencc_available_cache = True
        except Exception:
            traditional_enabled = False
            _opencc_available_cache = False
            socketio.emit("error", {"message": "簡轉繁需安裝 opencc，已暫時關閉轉換。"})
            return text
    try:
        return _opencc_converter.convert(text)
    except Exception:
        return text

def _proofread_line(index: int, original_text: str):
    """背景校對單行逐字稿"""
    proofread = proofread_text(original_text)
    if proofread and not proofread.startswith("[錯誤]"):
        with transcript_lock:
            if index < len(transcript_lines):
                transcript_lines[index]["proofread"] = proofread
        socketio.emit("proofread_update", {
            "index": index,
            "original": original_text,
            "proofread": proofread,
        })


def _prepare_summary_input(base_text: str) -> str:
    """摘要專用背景清理：只回傳供摘要使用的文本，不回寫逐字稿。"""
    base = (base_text or "").strip()
    if not base:
        return ""
    lines = []
    seen = set()
    for raw in base.splitlines():
        line = " ".join(raw.split()).strip()
        if not line or len(line) <= 1:
            continue
        key = "".join(ch for ch in line.lower() if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        lines.append(line)
    return "\n".join(lines).strip() or base


def _is_model_error(text: str) -> bool:
    return (text or "").strip().startswith("[錯誤]")


def _generate_summary(mode: str, full_text: str):
    """背景生成摘要"""
    summary_input = _prepare_summary_input(full_text)
    if not summary_input.strip():
        socketio.emit("error", {"message": "尚無逐字稿內容可供摘要"})
        return

    if mode == "full":
        content = summarize_full(summary_input)
        if _is_model_error(content):
            socketio.emit("error", {"message": content})
            return
    elif mode == "key_points":
        content = summarize_key_points(summary_input)
        if _is_model_error(content):
            socketio.emit("error", {"message": content})
            return
    elif mode == "action_items":
        content = extract_action_items(summary_input)
        if _is_model_error(content):
            socketio.emit("error", {"message": content})
            return
    elif mode == "all":
        summary_full = summarize_full(summary_input)
        if _is_model_error(summary_full):
            socketio.emit("error", {"message": summary_full})
            return
        summary_key = summarize_key_points(summary_input)
        if _is_model_error(summary_key):
            socketio.emit("error", {"message": summary_key})
            return
        summary_actions = extract_action_items(summary_input)
        if _is_model_error(summary_actions):
            socketio.emit("error", {"message": summary_actions})
            return
        content_lines = []
        content_lines.append("【全文摘要】")
        content_lines.append(summary_full)
        content_lines.append("")
        content_lines.append("【重點條列】")
        content_lines.append(summary_key)
        content_lines.append("")
        content_lines.append("【待辦清單】")
        content_lines.append(summary_actions)
        content_lines.append("")
        content_lines.append("【逐字稿】")
        content_lines.append(full_text)
        content = "\n".join(content_lines)
    else:
        content = summarize_full(summary_input)

    socketio.emit("summary_result", {"mode": mode, "content": content})


def _export_meeting(meeting_name: str, transcript_override: str = ""):
    """背景匯出逐字稿（僅 transcript）"""
    with transcript_lock:
        full_text = (transcript_override or "").strip() or "\n".join(
            line.get("proofread", line["text"]) for line in transcript_lines
        )
    if not full_text.strip():
        socketio.emit("error", {"message": "尚無逐字稿內容可供匯出"})
        return
    transcript_content = f"會議名稱: {meeting_name}\n\n{full_text}" if full_text else f"會議名稱: {meeting_name}"

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"{meeting_name}_transcript_{ts}.txt"

    # 僅寫入逐字稿
    export_dir = os.path.join(os.path.dirname(__file__), "download", meeting_name)
    os.makedirs(export_dir, exist_ok=True)
    transcript_path = os.path.join(export_dir, output_filename)
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript_content)

    socketio.emit("export_ready", {
        "files": [
            {
                "filename": output_filename,
                "content": transcript_content,
                "saved_path": transcript_path,
            },
        ]
    })


def _export_summary(meeting_name: str, mode: str, transcript_override: str = ""):
    """背景匯出摘要"""
    with transcript_lock:
        full_text = (transcript_override or "").strip() or "\n".join(
            line.get("proofread", line["text"]) for line in transcript_lines
        )
    if not full_text.strip():
        socketio.emit("error", {"message": "尚無逐字稿內容可供匯出摘要"})
        return
    summary_input = _prepare_summary_input(full_text)
    if not summary_input.strip():
        socketio.emit("error", {"message": "尚無可用內容可供匯出摘要"})
        return
    summary_full = ""
    summary_key = ""
    summary_actions = ""
    if summary_input.strip():
        if mode in ("full", "all"):
            summary_full = summarize_full(summary_input)
        if mode in ("key_points", "all"):
            summary_key = summarize_key_points(summary_input)
        if mode in ("action_items", "all"):
            summary_actions = extract_action_items(summary_input)
    else:
        summary_full = "[錯誤] 尚無逐字稿內容可供摘要"
        summary_key = summary_full
        summary_actions = summary_full

    summary_lines = []
    summary_lines.append(f"會議名稱: {meeting_name}")
    summary_lines.append(f"匯出時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("=" * 50)
    summary_lines.append("")
    if mode in ("full", "all"):
        summary_lines.append("【全文摘要】")
        summary_lines.append(summary_full)
        summary_lines.append("")
    if mode in ("key_points", "all"):
        summary_lines.append("【重點條列】")
        summary_lines.append(summary_key)
        summary_lines.append("")
    if mode in ("action_items", "all"):
        summary_lines.append("【待辦清單】")
        summary_lines.append(summary_actions)
    summary_content = "\n".join(summary_lines)

    mode_suffix_map = {
        "full": "full",
        "key_points": "key_points",
        "action_items": "action_items",
        "all": "all",
    }
    mode_suffix = mode_suffix_map.get(mode, "full")
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"{meeting_name}_summary_{mode_suffix}_{ts}.txt"

    export_dir = os.path.join(os.path.dirname(__file__), "download", meeting_name)
    os.makedirs(export_dir, exist_ok=True)
    summary_path = os.path.join(export_dir, output_filename)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_content)

    socketio.emit("export_ready", {
        "files": [
            {
                "filename": output_filename,
                "content": summary_content,
                "saved_path": summary_path,
            },
        ]
    })


def _enhance_transcript():
    """背景校對整份逐字稿（事後高品質處理）"""
    with transcript_lock:
        items = [
            (idx, line.get("text", ""), line.get("proofread"))
            for idx, line in enumerate(transcript_lines)
        ]

    for idx, text, existing in items:
        if not text or (existing and existing.strip()):
            continue
        proofread = proofread_text(text)
        if proofread and not proofread.startswith("[錯誤]"):
            with transcript_lock:
                if idx < len(transcript_lines):
                    transcript_lines[idx]["proofread"] = proofread
            socketio.emit("proofread_update", {
                "index": idx,
                "original": text,
                "proofread": proofread,
            })
    socketio.emit("enhance_done", {"ok": True})


def _ensure_audio_worker() -> None:
    global audio_worker_started
    if audio_worker_started:
        return
    audio_worker_started = True
    socketio.start_background_task(_audio_worker)


if __name__ == "__main__":
    print("=" * 50)
    print("  AI 會議助理 v1.1.0")
    print("  http://localhost:8000")
    print("=" * 50)
    _ensure_audio_worker()
    socketio.run(
        app,
        host="0.0.0.0",
        port=8000,
        debug=True,
        allow_unsafe_werkzeug=True,
        use_reloader=False,
    )
