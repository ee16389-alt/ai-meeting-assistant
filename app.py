"""Flask + SocketIO 主伺服器 - 路由與事件處理"""

import os
import sys
import threading
import time


def _configure_console_encoding() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass


_configure_console_encoding()

from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from stt_engine import STTEngine
from cognition import (
    proofread_text,
    summarize_full,
    summarize_key_points,
    extract_action_items,
    check_health,
    summary_engine_status,
)

_base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, template_folder=os.path.join(_base_path, "templates"),
            static_folder=os.path.join(_base_path, "static"))
app.config["SECRET_KEY"] = "meeting-assistant-secret"
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    max_http_buffer_size=10 * 1024 * 1024,
    async_mode="threading",
)

class _UnavailableSTT:
    """STT 初始化失敗時的保底物件，避免後端整體啟動失敗。"""

    def __init__(self, error: Exception):
        self._error = str(error)

    @property
    def state(self) -> str:
        return "error"

    def start(self) -> str:
        return "error"

    def pause(self) -> str:
        return "error"

    def resume(self) -> str:
        return "error"

    def stop(self) -> tuple[str, list[dict]]:
        return "error", []

    def feed_audio(self, chunk: bytes) -> list[dict]:
        return []

    @property
    def error(self) -> str:
        return self._error


# 全域 STT 引擎實例（初始化失敗時不讓整個 Flask 進程退出）
_stt_init_error = ""
try:
    stt = STTEngine(model_size="small")
except Exception as e:
    _stt_init_error = str(e)
    print(f"[STT] 初始化失敗，後端將以降級模式啟動: {_stt_init_error}", flush=True)
    stt = _UnavailableSTT(e)

# 會議逐字稿暫存（用於摘要與匯出）
transcript_lines: list[dict] = []
_transcript_lock = threading.Lock()
proofread_index = 0  # 追蹤下一個待校對的行號
audio_chunk_count = 0
audio_save_enabled = False
audio_file_handle = None
current_meeting_name = ""


def _export_root_dir() -> str:
    configured = (os.environ.get("AMA_EXPORT_ROOT") or "").strip()
    if configured:
        return configured

    home = os.path.expanduser("~")
    if sys.platform.startswith("win"):
        documents = os.path.join(home, "Documents")
        return os.path.join(documents, "AI Meeting Assistant")
    if sys.platform == "darwin":
        return os.path.join(home, "Documents", "AI Meeting Assistant")
    return os.path.join(home, "ai-meeting-assistant")


def _meeting_output_dir(meeting_name: str) -> str:
    safe_name = (meeting_name or "").strip() or current_meeting_name or time.strftime("meeting_%Y%m%d_%H%M%S")
    return os.path.join(_export_root_dir(), "download", safe_name)


def _open_folder(path: str) -> None:
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        os.system(f'open "{path}"')
        return
    os.system(f'xdg-open "{path}"')


@app.route("/")
def index():
    return send_from_directory(os.path.join(_base_path, "templates"), "index.html")


@app.route("/health")
def health():
    summary_status = summary_engine_status()
    stt_ok = stt.state != "error"
    return {
        "ok": True,
        "models_ready": stt_ok,
        "stt_ready": stt_ok,
        "stt_error": _stt_init_error or None,
        "summary_ready": summary_status["ready"],
        "summary_mode": summary_status["mode"],
        "summary_error": summary_status["error"],
        "summary_gguf_path": summary_status["gguf_path"],
    }


# ── SocketIO 事件處理 ───────────────────────────────────

@socketio.on("connect")
def handle_connect():
    ollama_ok = check_health()
    payload = {"state": stt.state, "ollama": ollama_ok}
    if _stt_init_error:
        payload["stt_error"] = _stt_init_error
    emit("state_changed", payload)


@socketio.on("start_recording")
def handle_start(data=None):
    global transcript_lines, proofread_index, audio_save_enabled, audio_file_handle, current_meeting_name, audio_chunk_count
    with _transcript_lock:
        transcript_lines = []
        proofread_index = 0
    audio_chunk_count = 0
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
        export_dir = _meeting_output_dir(meeting_name)
        os.makedirs(export_dir, exist_ok=True)
        audio_path = os.path.join(export_dir, "audio.webm")
        audio_file_handle = open(audio_path, "wb")
        audio_save_enabled = True

    state = stt.start()
    if state == "error":
        emit("error", {"message": f"語音模型初始化失敗：{_stt_init_error or '未知錯誤'}"})
        emit("state_changed", {"state": state})
        return {"ok": False, "state": state, "error": _stt_init_error or "未知錯誤"}
    emit("state_changed", {"state": state})
    return {"ok": True, "state": state}


@socketio.on("audio_chunk")
def handle_audio_chunk(data):
    """接收二進位音頻 chunk"""
    global audio_chunk_count
    if stt.state == "error":
        return {"ok": False, "reason": "stt_unavailable", "error": _stt_init_error}
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

    segments = stt.feed_audio(chunk)
    partial_text = getattr(stt, "partial_text", "").strip()
    if partial_text:
        emit("transcript_partial", {
            "text": partial_text,
            "timestamp": time.strftime("%H:%M:%S"),
        })
    else:
        emit("transcript_partial_clear")

    for seg in segments:
        with _transcript_lock:
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
    return {
        "ok": True,
        "size": size,
        "count": audio_chunk_count,
        "partial": partial_text,
        "audio_rms": round(float(getattr(stt, "last_audio_rms", 0.0)), 5),
    }


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
    if stt.state == "error":
        emit("error", {"message": f"語音模型不可用：{_stt_init_error or '未知錯誤'}"})
        emit("state_changed", {"state": "error"})
        return {"ok": False, "state": "error", "error": _stt_init_error or "未知錯誤"}
    state = stt.pause()
    emit("state_changed", {"state": state})
    return {"ok": True, "state": state}


@socketio.on("resume_recording")
def handle_resume():
    if stt.state == "error":
        emit("error", {"message": f"語音模型不可用：{_stt_init_error or '未知錯誤'}"})
        emit("state_changed", {"state": "error"})
        return {"ok": False, "state": "error", "error": _stt_init_error or "未知錯誤"}
    state = stt.resume()
    emit("state_changed", {"state": state})
    return {"ok": True, "state": state}


@socketio.on("stop_recording")
def handle_stop():
    if stt.state == "error":
        emit("state_changed", {"state": "error"})
        return {"ok": False, "state": "error", "error": _stt_init_error or "未知錯誤"}
    handle_audio_recording_done()
    state, final_segments = stt.stop()
    socketio.emit("transcript_partial_clear")
    for seg in final_segments:
        with _transcript_lock:
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
    return {"ok": True, "state": "idle"}


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


@socketio.on("get_storage_path")
def handle_get_storage_path(data=None):
    meeting_name = ""
    if isinstance(data, dict):
        meeting_name = (data.get("meeting_name") or "").strip()
    emit("storage_path", {"path": _meeting_output_dir(meeting_name)})


@socketio.on("open_storage_folder")
def handle_open_storage_folder(data=None):
    meeting_name = ""
    if isinstance(data, dict):
        meeting_name = (data.get("meeting_name") or "").strip()
    target = _meeting_output_dir(meeting_name)
    os.makedirs(target, exist_ok=True)
    try:
        _open_folder(target)
        emit("folder_opened", {"path": target})
    except Exception as e:
        emit("error", {"message": f"無法開啟資料夾：{e}"})


@socketio.on("export_meeting")
def handle_export(data):
    meeting_name = data.get("meeting_name", "").strip()
    transcript_override = data.get("transcript_override", "").strip()
    if not meeting_name:
        meeting_name = time.strftime("meeting_%Y%m%d_%H%M%S")
    socketio.start_background_task(_export_meeting, meeting_name, transcript_override)


@socketio.on("export_summary")
def handle_export_summary(data):
    meeting_name = data.get("meeting_name", "").strip()
    mode = data.get("mode", "full")
    transcript_override = data.get("transcript_override", "").strip()
    if not meeting_name:
        meeting_name = time.strftime("meeting_%Y%m%d_%H%M%S")
    socketio.start_background_task(_export_summary, meeting_name, mode, transcript_override)


# ── 背景任務 ───────────────────────────────────────────

def _proofread_line(index: int, original_text: str):
    """背景校對單行逐字稿"""
    proofread = proofread_text(original_text)
    if proofread and not proofread.startswith("[錯誤]"):
        with _transcript_lock:
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
        socketio.emit("summary_result", {"mode": mode, "content": content})
    elif mode == "key_points":
        content = summarize_key_points(full_text)
        socketio.emit("summary_result", {"mode": mode, "content": content})
    elif mode == "action_items":
        content = extract_action_items(full_text)
        socketio.emit("summary_result", {"mode": mode, "content": content})
    elif mode == "all":
        # 優化：合併一次推論，速度提升 3 倍
        combo = summarize_all_in_one(full_text)
        
        # 發送各個部分的結果回前端
        socketio.emit("summary_result", {"mode": "full", "content": combo["full"]})
        socketio.emit("summary_result", {"mode": "key_points", "content": combo["key_points"]})
        socketio.emit("summary_result", {"mode": "action_items", "content": combo["action_items"]})
        
        # 也發送一個整體的 "all" 供相容性使用
        full_all_text = (
            f"【全文摘要】\n{combo['full']}\n\n"
            f"【重點條列】\n{combo['key_points']}\n\n"
            f"【待辦清單】\n{combo['action_items']}"
        )
        socketio.emit("summary_result", {"mode": "all", "content": full_all_text})


def _export_meeting(meeting_name: str, transcript_override: str = ""):
    """背景匯出逐字稿與摘要"""
    if transcript_override:
        full_text = transcript_override
    else:
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
    
    if transcript_override:
        transcript_lines_out.append(transcript_override)
    else:
        for item in transcript_lines:
            ts = item.get("timestamp", "")
            text = item.get("proofread", item["text"])
            transcript_lines_out.append(f"[{ts}] {text}")

    transcript_content = "\n".join(transcript_lines_out)

    # 使用快速合併摘要
    if full_text.strip():
        combo = summarize_all_in_one(full_text)
        summary_full = combo["full"]
        summary_key = combo["key_points"]
        summary_actions = combo["action_items"]
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
    export_dir = _meeting_output_dir(meeting_name)
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
                "saved_path": transcript_path,
            },
            {
                "filename": f"{meeting_name}_summary.txt",
                "saved_path": summary_path,
            },
        ]
    })


def _export_summary(meeting_name: str, mode: str, transcript_override: str = ""):
    """背景匯出摘要"""
    if transcript_override:
        full_text = transcript_override
    else:
        full_text = "\n".join(
            line.get("proofread", line["text"]) for line in transcript_lines
        )

    summary_content = ""
    if full_text.strip():
        combo = summarize_all_in_one(full_text)
        
        summary_lines = []
        summary_lines.append(f"會議名稱: {meeting_name}")
        summary_lines.append(f"匯出時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("=" * 50)
        summary_lines.append("")
        
        if mode in ("full", "all"):
            summary_lines.append("【全文摘要】")
            summary_lines.append(combo["full"])
            summary_lines.append("")
        if mode in ("key_points", "all"):
            summary_lines.append("【重點條列】")
            summary_lines.append(combo["key_points"])
            summary_lines.append("")
        if mode in ("action_items", "all"):
            summary_lines.append("【待辦清單】")
            summary_lines.append(combo["action_items"])
            
        summary_content = "\n".join(summary_lines)
    else:
        summary_content = "尚無內容可供匯出"

    export_dir = _meeting_output_dir(meeting_name)
    os.makedirs(export_dir, exist_ok=True)
    summary_path = os.path.join(export_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_content)

    socketio.emit("export_ready", {
        "files": [
            {
                "filename": f"{meeting_name}_summary.txt",
                "saved_path": summary_path,
            },
        ]
    })


if __name__ == "__main__":
    print("=" * 50, flush=True)
    print("  AI 會議助理後端啟動中...", flush=True)
    print(f"  根目錄: {_base_path}", flush=True)
    print("  http://localhost:8000", flush=True)
    print("=" * 50, flush=True)
    socketio.run(
        app,
        host="0.0.0.0",
        port=8000,
        debug=False,
        allow_unsafe_werkzeug=True,
        use_reloader=False,
    )
