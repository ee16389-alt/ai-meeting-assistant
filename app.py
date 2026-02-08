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

# 全域 STT 引擎實例
stt = STTEngine(model_size="small")

# 會議逐字稿暫存（用於摘要與匯出）
transcript_lines: list[dict] = []
proofread_index = 0  # 追蹤下一個待校對的行號
audio_chunk_count = 0
audio_save_enabled = False
audio_file_handle = None
current_meeting_name = ""


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
    emit("state_changed", {"state": stt.state, "ollama": ollama_ok})


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

    segments = stt.feed_audio(chunk)
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
