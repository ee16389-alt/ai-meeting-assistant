"""Flask + SocketIO 主伺服器 - 路由與事件處理"""

import os
import time
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from stt_engine import STTEngine
from cognition import (
    proofread_text,
    summarize_full,
    summarize_key_points,
    extract_action_items,
    check_health,
)

app = Flask(__name__)
app.config["SECRET_KEY"] = "meeting-assistant-secret"
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=10 * 1024 * 1024)

# 可切換的 STT 模型
AVAILABLE_STT_MODELS = {
    "whisper_small": {"label": "Whisper Small", "size": "small", "engine": "faster-whisper"},
    "whisper_medium": {"label": "Whisper Medium", "size": "medium", "engine": "faster-whisper"},
    "whisper_large": {"label": "Whisper Large", "size": "large", "engine": "faster-whisper"},
    "faster_whisper_medium": {"label": "Faster-Whisper Medium (多語)", "size": "medium", "engine": "faster-whisper"},
    "whispercpp_small": {"label": "Whisper.cpp Small", "size": "small", "engine": "whisper.cpp", "model_id": "small"},
    "whispercpp_medium": {"label": "Whisper.cpp Medium", "size": "medium", "engine": "whisper.cpp", "model_id": "medium"},
    "whispercpp_large": {"label": "Whisper.cpp Large-v3", "size": "large-v3", "engine": "whisper.cpp", "model_id": "large-v3"},
}
DEFAULT_STT_MODEL_KEY = "whisper_small"

# 全域 STT 引擎實例
stt = STTEngine(
    model_size=AVAILABLE_STT_MODELS[DEFAULT_STT_MODEL_KEY]["size"],
    engine=AVAILABLE_STT_MODELS[DEFAULT_STT_MODEL_KEY]["engine"],
    model_id=AVAILABLE_STT_MODELS[DEFAULT_STT_MODEL_KEY].get("model_id"),
)
current_stt_model_key = DEFAULT_STT_MODEL_KEY

# 會議逐字稿暫存（用於摘要與匯出）
transcript_lines: list[dict] = []
proofread_index = 0  # 追蹤下一個待校對的行號
audio_chunk_count = 0
audio_save_enabled = False
audio_file_handle = None
current_meeting_name = ""

DEFAULT_WHISPER_CPP_BIN = "/Users/minashih/tools/whisper.cpp/bin/whisper-cli"
DEFAULT_WHISPER_CPP_MODEL_DIR = "/Users/minashih/models/whisper.cpp"


def _whisper_cpp_availability() -> tuple[bool, str]:
    bin_path = os.getenv("WHISPER_CPP_BIN", "").strip() or DEFAULT_WHISPER_CPP_BIN
    model_dir = os.getenv("WHISPER_CPP_MODEL_DIR", "").strip() or DEFAULT_WHISPER_CPP_MODEL_DIR
    if not bin_path or not os.path.isfile(bin_path):
        return False, "WHISPER_CPP_BIN 未設定或檔案不存在"
    if not model_dir or not os.path.isdir(model_dir):
        return False, "WHISPER_CPP_MODEL_DIR 未設定或目錄不存在"
    for size in ("small", "medium", "large-v3"):
        model_path = os.path.join(model_dir, f"ggml-{size}.bin")
        if not os.path.isfile(model_path):
            return False, f"缺少模型檔 ggml-{size}.bin"
    return True, ""


def _model_payload() -> dict:
    whisper_cpp_ok, whisper_cpp_reason = _whisper_cpp_availability()
    current = AVAILABLE_STT_MODELS.get(
        current_stt_model_key, {"label": "Unknown", "size": "small", "engine": "faster-whisper"}
    )
    return {
        "stt_model": {
            "key": current_stt_model_key,
            "label": current["label"],
            "size": current["size"],
            "engine": current.get("engine", "faster-whisper"),
        },
        "stt_models": [
            {
                "key": key,
                "label": item["label"],
                "size": item["size"],
                "engine": item.get("engine", "faster-whisper"),
                "model_id": item.get("model_id", item["size"]),
                "available": (
                    item.get("engine", "faster-whisper") != "whisper.cpp"
                    or whisper_cpp_ok
                ),
                "reason": (
                    "" if item.get("engine", "faster-whisper") != "whisper.cpp" else whisper_cpp_reason
                ),
            }
            for key, item in AVAILABLE_STT_MODELS.items()
        ],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return {"ok": True}


# ── SocketIO 事件處理 ───────────────────────────────────

@socketio.on("connect")
def handle_connect():
    ollama_ok = check_health()
    payload = {"state": stt.state, "ollama": ollama_ok}
    payload.update(_model_payload())
    emit("state_changed", payload)


@socketio.on("start_recording")
def handle_start(data=None):
    global transcript_lines, proofread_index, audio_save_enabled, audio_file_handle, current_meeting_name
    transcript_lines = []
    proofread_index = 0
    audio_save_enabled = False
    if audio_file_handle:
        try:
            audio_file_handle.close()
        except Exception:
            pass
        audio_file_handle = None

    meeting_name = ""
    save_audio = False
    if isinstance(data, dict):
        meeting_name = (data.get("meeting_name") or "").strip()
        save_audio = bool(data.get("save_audio"))
    if not meeting_name:
        meeting_name = time.strftime("meeting_%Y%m%d_%H%M%S")
    current_meeting_name = meeting_name

    if save_audio:
        export_dir = os.path.join(os.path.dirname(__file__), "download", meeting_name)
        os.makedirs(export_dir, exist_ok=True)
        audio_path = os.path.join(export_dir, "audio.webm")
        audio_file_handle = open(audio_path, "wb")
        audio_save_enabled = True

    state = stt.start()
    emit("state_changed", {"state": state})


@socketio.on("audio_chunk")
def handle_audio_chunk(data):
    """接收二進位音頻 chunk"""
    global audio_chunk_count
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

    result = stt.feed_audio(chunk)
    partial_text = ""
    if isinstance(result, dict):
        partial_text = result.get("partial", "") or ""
        segments = result.get("final", []) or []
    else:
        segments = result
    if partial_text:
        emit("transcript_partial", {
            "text": partial_text,
            "timestamp": time.strftime("%H:%M:%S"),
        })
    if segments:
        emit("transcript_partial_clear")
    for seg in segments:
        line = {
            "index": len(transcript_lines),
            "text": seg["text"],
            "timestamp": time.strftime("%H:%M:%S"),
            "language": seg.get("language", ""),
        }
        transcript_lines.append(line)
        emit("transcript_update", line)

        # 非同步校對
        idx = line["index"]
        original_text = line["text"]
        socketio.start_background_task(
            _proofread_line, idx, original_text
        )
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
    handle_audio_recording_done()
    state, final_segments = stt.stop()
    if final_segments:
        socketio.emit("transcript_partial_clear")
    for seg in final_segments:
        line = {
            "index": len(transcript_lines),
            "text": seg["text"],
            "timestamp": time.strftime("%H:%M:%S"),
            "language": seg.get("language", ""),
        }
        transcript_lines.append(line)
        socketio.emit("transcript_update", line)

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
    whisper_cpp_ok, whisper_cpp_reason = _whisper_cpp_availability()
    if AVAILABLE_STT_MODELS[model_key].get("engine") == "whisper.cpp" and not whisper_cpp_ok:
        emit("error", {"message": f"Whisper.cpp 尚未就緒：{whisper_cpp_reason}"})
        return

    model_size = AVAILABLE_STT_MODELS[model_key]["size"]
    model_engine = AVAILABLE_STT_MODELS[model_key].get("engine", "faster-whisper")
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


@socketio.on("request_summary")
def handle_summary(data):
    mode = data.get("mode", "full")
    full_text = "\n".join(
        line.get("proofread", line["text"]) for line in transcript_lines
    )

    if not full_text.strip():
        emit("error", {"message": "尚無逐字稿內容可供摘要"})
        return

    socketio.start_background_task(_generate_summary, mode, full_text)


@socketio.on("export_meeting")
def handle_export(data):
    meeting_name = data.get("meeting_name", "").strip()
    if not meeting_name:
        meeting_name = time.strftime("meeting_%Y%m%d_%H%M%S")
    socketio.start_background_task(_export_meeting, meeting_name)


@socketio.on("export_summary")
def handle_export_summary(data):
    meeting_name = data.get("meeting_name", "").strip()
    mode = data.get("mode", "full")
    if not meeting_name:
        meeting_name = time.strftime("meeting_%Y%m%d_%H%M%S")
    socketio.start_background_task(_export_summary, meeting_name, mode)


# ── 背景任務 ───────────────────────────────────────────

def _proofread_line(index: int, original_text: str):
    """背景校對單行逐字稿"""
    proofread = proofread_text(original_text)
    if proofread and not proofread.startswith("[錯誤]"):
        if index < len(transcript_lines):
            transcript_lines[index]["proofread"] = proofread
        socketio.emit("proofread_update", {
            "index": index,
            "original": original_text,
            "proofread": proofread,
        })


def _generate_summary(mode: str, full_text: str):
    """背景生成摘要"""
    if mode == "full":
        content = summarize_full(full_text)
    elif mode == "key_points":
        content = summarize_key_points(full_text)
    elif mode == "action_items":
        content = extract_action_items(full_text)
    elif mode == "all":
        summary_full = summarize_full(full_text)
        summary_key = summarize_key_points(full_text)
        summary_actions = extract_action_items(full_text)
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
        content = summarize_full(full_text)

    socketio.emit("summary_result", {"mode": mode, "content": content})


def _export_meeting(meeting_name: str):
    """背景匯出逐字稿與摘要"""
    full_text = "\n".join(
        line.get("proofread", line["text"]) for line in transcript_lines
    )

    # 組合逐字稿內容
    transcript_lines_out = []
    transcript_lines_out.append(f"會議名稱: {meeting_name}")
    transcript_lines_out.append(f"匯出時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    transcript_lines_out.append("=" * 50)
    transcript_lines_out.append("")
    transcript_lines_out.append("【逐字稿】")
    transcript_lines_out.append("")
    for item in transcript_lines:
        ts = item.get("timestamp", "")
        text = item.get("proofread", item["text"])
        transcript_lines_out.append(f"[{ts}] {text}")

    transcript_content = "\n".join(transcript_lines_out)

    # 生成摘要
    if full_text.strip():
        summary_full = summarize_full(full_text)
        summary_key = summarize_key_points(full_text)
        summary_actions = extract_action_items(full_text)
    else:
        summary_full = "[錯誤] 尚無逐字稿內容可供摘要"
        summary_key = summary_full
        summary_actions = summary_full

    summary_lines = []
    summary_lines.append(f"會議名稱: {meeting_name}")
    summary_lines.append(f"匯出時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("=" * 50)
    summary_lines.append("")
    summary_lines.append("【全文摘要】")
    summary_lines.append(summary_full)
    summary_lines.append("")
    summary_lines.append("【重點條列】")
    summary_lines.append(summary_key)
    summary_lines.append("")
    summary_lines.append("【待辦清單】")
    summary_lines.append(summary_actions)
    summary_content = "\n".join(summary_lines)

    # 寫入檔案
    export_dir = os.path.join(os.path.dirname(__file__), "download", meeting_name)
    os.makedirs(export_dir, exist_ok=True)
    transcript_path = os.path.join(export_dir, "transcript.txt")
    summary_path = os.path.join(export_dir, "summary.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript_content)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_content)

    socketio.emit("export_ready", {
        "files": [
            {
                "filename": f"{meeting_name}_transcript.txt",
                "content": transcript_content,
            },
            {
                "filename": f"{meeting_name}_summary.txt",
                "content": summary_content,
            },
        ]
    })


def _export_summary(meeting_name: str, mode: str):
    """背景匯出摘要"""
    full_text = "\n".join(
        line.get("proofread", line["text"]) for line in transcript_lines
    )
    summary_full = ""
    summary_key = ""
    summary_actions = ""
    if full_text.strip():
        if mode in ("full", "all"):
            summary_full = summarize_full(full_text)
        if mode in ("key_points", "all"):
            summary_key = summarize_key_points(full_text)
        if mode in ("action_items", "all"):
            summary_actions = extract_action_items(full_text)
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

    export_dir = os.path.join(os.path.dirname(__file__), "download", meeting_name)
    os.makedirs(export_dir, exist_ok=True)
    summary_path = os.path.join(export_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_content)

    socketio.emit("export_ready", {
        "files": [
            {
                "filename": f"{meeting_name}_summary.txt",
                "content": summary_content,
            },
        ]
    })


if __name__ == "__main__":
    print("=" * 50)
    print("  AI 會議助理 v1.1.0")
    print("  http://localhost:8000")
    print("=" * 50)
    socketio.run(
        app,
        host="0.0.0.0",
        port=8000,
        debug=True,
        allow_unsafe_werkzeug=True,
        use_reloader=False,
    )
