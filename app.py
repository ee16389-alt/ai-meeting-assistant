"""Flask + SocketIO 主伺服器 - 路由與事件處理"""

import os
import sys
import threading
import time

import numpy as np


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

from flask import Flask, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit
from stt_engine import STTEngine
from cognition import (
    proofread_text,
    summarize_full,
    summarize_key_points,
    summarize_all_in_one,
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

    def request_stop(self) -> np.ndarray:
        return np.array([], dtype=np.float32)

    def stop(self) -> tuple[str, list[dict]]:
        return "error", []

    def feed_audio(self, chunk: bytes) -> list[dict]:
        return []

    @property
    def error(self) -> str:
        return self._error


# 全域 STT 引擎實例（背景執行緒初始化，避免模型下載阻塞 Flask 啟動）
_stt_init_error = ""
_stt_init_done = threading.Event()
stt: STTEngine | _UnavailableSTT = _UnavailableSTT(Exception("STT 初始化中..."))

# 後端診斷 log 緩衝（最多 100 行）
_debug_logs: list[str] = []
_debug_logs_lock = threading.Lock()

_orig_print = print
def _capturing_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    with _debug_logs_lock:
        _debug_logs.append(msg)
        if len(_debug_logs) > 100:
            del _debug_logs[:-100]
    _orig_print(*args, **kwargs)

import builtins
builtins.print = _capturing_print


def _on_stt_segments(segments: list[dict]):
    """STT 背景 worker 完成推論後的回呼，透過 socketio 推送結果"""
    sid = _active_sid
    print(f"[CB] _on_stt_segments called: {len(segments)} seg(s), sid={repr(sid)}", flush=True)
    if not sid:
        print("[CB] no active sid, dropping", flush=True)
        return
    for seg in segments:
        with _transcript_lock:
            line = {
                "index": len(transcript_lines),
                "text": seg["text"],
                "timestamp": time.strftime("%H:%M:%S"),
                "language": seg.get("language", ""),
            }
            transcript_lines.append(line)
        print(f"[CB] emit transcript_update: {repr(line['text'][:40])}", flush=True)
        socketio.emit("transcript_update", line, room=sid)


def _init_stt_background():
    global stt, _stt_init_error
    try:
        instance = STTEngine(model_size="base")
        instance.set_result_callback(_on_stt_segments)
        stt = instance
        print("[STT] 模型初始化完成，後端就緒", flush=True)
    except Exception as e:
        _stt_init_error = str(e)
        print(f"[STT] 初始化失敗，後端將以降級模式啟動: {_stt_init_error}", flush=True)
        stt = _UnavailableSTT(e)
    finally:
        _stt_init_done.set()


threading.Thread(target=_init_stt_background, daemon=True).start()

# 會議逐字稿暫存（用於摘要與匯出）
transcript_lines: list[dict] = []
_transcript_lock = threading.Lock()
proofread_index = 0  # 追蹤下一個待校對的行號
audio_chunk_count = 0
_active_sid: str = ""  # 目前錄音的 client session id
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


def _unique_meeting_name(name: str) -> str:
    """若資料夾已存在，自動加 _2、_3… 避免覆蓋舊會議。"""
    base = os.path.join(_export_root_dir(), "download", name)
    if not os.path.exists(base):
        return name
    counter = 2
    while os.path.exists(os.path.join(_export_root_dir(), "download", f"{name}_{counter}")):
        counter += 1
    return f"{name}_{counter}"


def _meeting_output_dir(meeting_name: str = "") -> str:
    # 優先使用錄音時設定的名稱，確保匯出與錄音音檔在同一資料夾
    safe_name = current_meeting_name or (meeting_name or "").strip() or time.strftime("meeting_%Y%m%d_%H%M%S")
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


@app.route("/api/debug")
def api_debug():
    with _debug_logs_lock:
        logs = list(_debug_logs)
    partial = stt.partial_text if hasattr(stt, "partial_text") else ""
    queue_size = stt._audio_queue.qsize() if hasattr(stt, "_audio_queue") else -1
    return {
        "logs": logs[-50:],
        "partial_text": partial,
        "queue_size": queue_size,
    }


@app.route("/health")
def health():
    summary_status = summary_engine_status()
    stt_initializing = not _stt_init_done.is_set()
    stt_ok = _stt_init_done.is_set() and stt.state != "error"
    return {
        "ok": True,
        "models_ready": stt_ok,
        "stt_ready": stt_ok,
        "stt_initializing": stt_initializing,
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
    global transcript_lines, proofread_index, audio_save_enabled, audio_file_handle, current_meeting_name, audio_chunk_count, _active_sid
    _active_sid = request.sid
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
    current_meeting_name = _unique_meeting_name(meeting_name)

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

    stt.feed_audio(chunk)  # 非阻塞，結果由背景 worker 透過 callback 推送
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
    # 原子操作：立即設為 IDLE 並取走剩餘 buffer，UI 立即響應
    remaining = stt.request_stop()
    socketio.emit("transcript_partial_clear")
    socketio.emit("state_changed", {"state": "idle"})
    # 非同步處理剩餘音頻，不阻塞回應
    socketio.start_background_task(_finish_transcription, remaining)
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
    summary_overrides = {
        "full": data.get("summary_full", "").strip(),
        "key_points": data.get("summary_key", "").strip(),
    }
    if not meeting_name:
        meeting_name = time.strftime("meeting_%Y%m%d_%H%M%S")
    socketio.start_background_task(_export_meeting, meeting_name, transcript_override, summary_overrides)


@socketio.on("export_summary")
def handle_export_summary(data):
    meeting_name = data.get("meeting_name", "").strip()
    mode = data.get("mode", "full")
    transcript_override = data.get("transcript_override", "").strip()
    summary_overrides = {
        "full": data.get("summary_full", "").strip(),
        "key_points": data.get("summary_key", "").strip(),
    }
    if not meeting_name:
        meeting_name = time.strftime("meeting_%Y%m%d_%H%M%S")
    socketio.start_background_task(_export_summary, meeting_name, mode, transcript_override, summary_overrides)


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


def _finish_transcription(remaining: np.ndarray):
    """背景完成停止後剩餘音頻的轉寫"""
    if remaining is None or remaining.size == 0:
        return
    final_segments = stt.transcribe_audio(remaining)
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


def _generate_summary(mode: str, full_text: str):
    """背景生成摘要（支援串流）"""
    system_prompt = ""
    guardrail = (
        "\n嚴格規則：只能使用逐字稿中明確出現的內容，禁止推測、補充或捏造任何未提及的資訊。"
        "若逐字稿資訊不足，直接說明「資訊不足」。"
    )
    if mode == "full":
        system_prompt = (
            "你是一位專業的會議記錄員。請用繁體中文輸出。\n"
            "請輸出精簡扼要的摘要，化繁為簡，必須包含逐字稿中具體提到的人名、數字、決議或事件。"
            + guardrail
        )
    elif mode == "key_points":
        system_prompt = (
            "你是一位專業的會議記錄員。請用繁體中文輸出。\n"
            "以條列式呈現 3-7 個重點，每個重點用「•」開頭。\n"
            "每個重點必須具體，包含逐字稿中實際提到的細節，不可寫泛泛的概念。"
            + guardrail
        )
    elif mode == "all":
        system_prompt = (
            "你是一位專業的會議記錄員。請用繁體中文輸出。\n"
            "請一次性提供以下內容：\n"
            "1. 【全文摘要】：精簡扼要，包含具體提到的決議或結論。\n"
            "2. 【重點條列】：3-7 個具體重點，每點用「•」開頭。"
            + guardrail
        )

    # 告訴前端準備開始串流
    socketio.emit("summary_start", {"mode": mode})
    
    from cognition import _call_model_stream
    accumulated = ""
    for chunk in _call_model_stream(system_prompt, full_text):
        accumulated += chunk
        socketio.emit("summary_chunk", {"mode": mode, "chunk": chunk})
    
    # 最終傳送完整結果以供快取
    socketio.emit("summary_result", {"mode": mode, "content": accumulated})


def _export_meeting(meeting_name: str, transcript_override: str = "", summary_overrides: dict = None):
    """背景匯出逐字稿與摘要"""
    if transcript_override:
        full_text = transcript_override
    else:
        full_text = "\n".join(
            line.get("proofread", line["text"]) for line in transcript_lines
        )

    # ... (逐字稿處理保持不變) ...
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

    # 處理摘要內容：僅使用前端傳來的快取內容，不在匯出時阻塞 LLM 重新推論
    if summary_overrides and summary_overrides.get("full"):
        summary_full = summary_overrides["full"]
        summary_key = summary_overrides.get("key_points", "")
    else:
        summary_full = ""
        summary_key = ""

    summary_lines = []
    summary_lines.append(f"會議名稱: {meeting_name}")
    summary_lines.append(f"匯出時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("=" * 50)
    summary_lines.append("")
    if summary_full or summary_key:
        summary_lines.append("【全文摘要】")
        summary_lines.append(summary_full or "（未生成）")
        summary_lines.append("")
        summary_lines.append("【重點條列】")
        summary_lines.append(summary_key or "（未生成）")
    else:
        summary_lines.append("（摘要尚未生成，請在主畫面點選摘要按鈕後再次匯出）")
    summary_content = "\n".join(summary_lines)

    # 寫入檔案
    export_dir = _meeting_output_dir(meeting_name)
    os.makedirs(export_dir, exist_ok=True)
    transcript_path = os.path.join(export_dir, "transcript.txt")
    summary_path = os.path.join(export_dir, "summary.txt")
    try:
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript_content)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_content)
    except Exception as e:
        socketio.emit("error", {"message": f"匯出失敗：{e}"})
        return

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


def _export_summary(meeting_name: str, mode: str, transcript_override: str = "", summary_overrides: dict = None):
    """背景匯出摘要（優先使用前端快取，避免重複推論）"""
    if transcript_override:
        full_text = transcript_override
    else:
        full_text = "\n".join(
            line.get("proofread", line["text"]) for line in transcript_lines
        )

    cached = summary_overrides or {}
    # 判斷是否有足夠的快取內容直接使用
    has_cache = any(cached.get(k) for k in ("full", "key_points"))

    summary_content = ""
    if has_cache or full_text.strip():
        if not has_cache:
            # 快取不足時才執行 LLM 推論
            try:
                combo = summarize_all_in_one(full_text)
                cached = {"full": combo["full"], "key_points": combo["key_points"]}
            except Exception as e:
                socketio.emit("error", {"message": f"摘要生成失敗，無法匯出：{e}"})
                return

        summary_lines = []
        summary_lines.append(f"會議名稱: {meeting_name}")
        summary_lines.append(f"匯出時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("=" * 50)
        summary_lines.append("")

        if mode in ("full", "all") and cached.get("full"):
            summary_lines.append("【全文摘要】")
            summary_lines.append(cached["full"])
            summary_lines.append("")
        if mode in ("key_points", "all") and cached.get("key_points"):
            summary_lines.append("【重點條列】")
            summary_lines.append(cached["key_points"])
            summary_lines.append("")

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
