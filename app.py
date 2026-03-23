"""Flask + SocketIO 主伺服器 - 路由與事件處理"""

import json
import os
import signal
import sys
import threading
import time
import uuid

import numpy as np

try:
    import opencc as _opencc
    _s2twp = _opencc.OpenCC("s2twp")
except Exception:
    _s2twp = None

def _to_traditional(text: str) -> str:
    if _s2twp is None:
        return text
    try:
        return _s2twp.convert(text)
    except Exception:
        return text


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

# ── 將 stdout/stderr 同時寫入桌面的 ama_debug.log，方便 exe 版本除錯 ──
import pathlib, io
_desktop = pathlib.Path.home() / "Desktop"
_log_path = _desktop / "ama_debug.log"
try:
    _log_fh = open(_log_path, "w", encoding="utf-8", errors="replace", buffering=1)

    class _Tee(io.TextIOBase):
        def __init__(self, *streams):
            self._streams = streams
        def write(self, s):
            for st in self._streams:
                try:
                    st.write(s)
                    st.flush()
                except Exception:
                    pass
            return len(s)
        def flush(self):
            for st in self._streams:
                try:
                    st.flush()
                except Exception:
                    pass

    sys.stdout = _Tee(sys.__stdout__, _log_fh)
    sys.stderr = _Tee(sys.__stderr__, _log_fh)
    print(f"[LOG] 診斷 log 已開啟，路徑: {_log_path}", flush=True)
except Exception as _e:
    print(f"[LOG] 無法建立 log 檔: {_e}", flush=True)

from flask import Flask, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit
from stt_engine import STTEngine
from cognition import (
    summarize_full,
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
    # 延長心跳超時，防止本地 LLM 推論期間（可能長達數分鐘）連線被切斷。
    # ping_interval=30：每 30 秒發一次心跳；
    # ping_timeout=120：120 秒內無回應才判定斷線。
    # threading 模式無 HTTP request timeout，SocketIO ping 是唯一的連線保活機制。
    ping_interval=30,
    ping_timeout=120,
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

    def request_stop(self) -> None:
        return None

    def take_pending_flush_stream(self):
        return None

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


def _make_stt_callback(session_sid: str, session_token: str):
    """每場錄音產生獨立的 STT callback，閉包鎖定 sid 與 token，
    避免舊 session 的 STT 結果污染新 session 的逐字稿。"""
    def _callback(segments: list[dict]):
        # 若 _recording_token 已換（新 session 開始），此 callback 已過期，直接丟棄
        if _recording_token != session_token:
            print(f"[CB] stale callback (token mismatch), dropping {len(segments)} seg(s)", flush=True)
            return
        print(f"[CB] _on_stt_segments called: {len(segments)} seg(s), sid={repr(session_sid)}", flush=True)
        for seg in segments:
            with _transcript_lock:
                line = {
                    "index": len(transcript_lines),
                    "text": seg["text"],
                    "timestamp": _elapsed_ts(_recording_start_time),
                    "language": seg.get("language", ""),
                    "token": session_token,
                }
                transcript_lines.append(line)
            print(f"[CB] emit transcript_update: {repr(line['text'][:40])}", flush=True)
            socketio.emit("transcript_update", line, room=session_sid)
    return _callback


def _on_stt_segments(segments: list[dict]):
    """相容舊介面，實際由 _make_stt_callback 產生的 closure 取代"""
    pass


def _on_stt_partial(text: str):
    """即時 partial text 回呼"""
    sid = _active_sid
    if not sid:
        return
    socketio.emit("partial_transcript", {"text": text, "is_partial": True}, room=sid)


def _init_stt_background():
    global stt, _stt_init_error
    try:
        instance = STTEngine(model_size="base")
        instance.set_result_callback(_on_stt_segments)
        instance.set_partial_callback(_on_stt_partial)
        stt = instance
        print("[STT] 模型初始化完成，後端就緒", flush=True)
    except Exception as e:
        _stt_init_error = str(e)
        print(f"[STT] 初始化失敗，後端將以降級模式啟動: {_stt_init_error}", flush=True)
        stt = _UnavailableSTT(e)
    finally:
        _stt_init_done.set()


threading.Thread(target=_init_stt_background, daemon=True).start()


def _warmup_llm():
    """背景預熱 LLM：延遲 60 秒後才啟動，避免與 STT 模型載入同時競搶 CPU。"""
    time.sleep(60)
    try:
        from cognition import _call_model_stream
        print("[LLM] 開始模型預熱...", flush=True)
        for _ in _call_model_stream("你是助理。", "你好", max_tokens=1):
            pass
        print("[LLM] 模型預熱完成，推理就緒", flush=True)
    except Exception as e:
        print(f"[LLM] 模型預熱失敗（不影響正常功能）: {e}", flush=True)


threading.Thread(target=_warmup_llm, daemon=True).start()

# 會議逐字稿暫存（用於摘要與匯出）
transcript_lines: list[dict] = []
_transcript_lock = threading.Lock()
audio_chunk_count = 0
_active_sid: str = ""  # 目前錄音的 client session id
_recording_token: str = ""  # 每次錄音生成的唯一 token，前端用來過濾舊事件
_recording_start_time: float = 0.0  # 本場錄音開始的 time.time()，用於計算經過時間戳記


def _elapsed_ts(start: float) -> str:
    """將 time.time() - start 換算成 HH:MM:SS 經過時間格式。"""
    elapsed = max(0.0, time.time() - start) if start else 0.0
    total_s = int(elapsed)
    h = total_s // 3600
    m = (total_s % 3600) // 60
    s = total_s % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
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
    global transcript_lines, audio_save_enabled, audio_file_handle, current_meeting_name, audio_chunk_count, _active_sid, _recording_token, _recording_start_time
    _active_sid = request.sid
    _recording_token = uuid.uuid4().hex  # 每場錄音唯一 token
    _recording_start_time = time.time()  # 記錄本場錄音開始時間
    # 重新綁定 STT callback，讓 closure 鎖定本場 sid 與 token
    if stt:
        stt.reset()  # 確保 STT 狀態從乾淨狀態開始，清除上場殘留 buffer
        stt.set_result_callback(_make_stt_callback(_active_sid, _recording_token))
    with _transcript_lock:
        transcript_lines = []
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
    return {"ok": True, "state": state, "recording_token": _recording_token}


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
    # request_stop：快速（drain queue + 設 IDLE），flush 交給 background task
    stt.request_stop()
    stop_sid = _active_sid
    stop_token = _recording_token
    socketio.emit("partial_transcript_clear", room=stop_sid)
    socketio.emit("state_changed", {"state": "idle"}, room=stop_sid)
    # 非同步 flush 剩餘音頻，不阻塞 ack 回傳
    socketio.start_background_task(_finish_transcription, stop_sid, stop_token)
    return {"ok": True, "state": "idle"}


@socketio.on("request_summary")
def handle_summary(data):
    sid = request.sid
    mode = data.get("mode", "full")
    # 優先使用前端傳來的逐字稿（含使用者編輯後內容），fallback 到後端快取
    transcript_override = (data.get("transcript_override") or "").strip()
    if transcript_override:
        full_text = transcript_override
    else:
        full_text = "\n".join(
            line["text"] for line in transcript_lines
        )

    if not full_text.strip():
        emit("error", {"message": "尚無逐字稿內容可供摘要"})
        return

    socketio.start_background_task(_generate_summary, mode, full_text, sid)


@socketio.on("cancel_summary")
def handle_cancel_summary():
    global _summary_cancelled
    with _summary_cancelled_lock:
        _summary_cancelled = True
    print("[Summary] 收到取消請求，設定取消旗標", flush=True)


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
    }
    if not meeting_name:
        meeting_name = time.strftime("meeting_%Y%m%d_%H%M%S")
    socketio.start_background_task(_export_summary, meeting_name, mode, transcript_override, summary_overrides)


# ── 背景任務 ───────────────────────────────────────────


_FINISH_TRANSCRIPTION_TIMEOUT = 30.0  # 最長等待 flush 完成的秒數

# 摘要取消旗標（每次新任務開始前重設）
_summary_cancelled = False
_summary_cancelled_lock = threading.Lock()


def _is_summary_cancelled() -> bool:
    with _summary_cancelled_lock:
        return _summary_cancelled


def _handle_shutdown_signal(signum, frame):
    """收到 SIGTERM / SIGINT 時設定取消旗標，讓進行中的推理盡快停止，再正常退出。"""
    global _summary_cancelled
    print(f"[App] 收到關閉信號 {signum}，設定取消旗標並退出", flush=True)
    with _summary_cancelled_lock:
        _summary_cancelled = True
    # 給 0.5 秒讓推理迴圈偵測到旗標後中止，再強制退出
    threading.Timer(0.5, lambda: os._exit(0)).start()


signal.signal(signal.SIGTERM, _handle_shutdown_signal)
signal.signal(signal.SIGINT, _handle_shutdown_signal)


def _finish_transcription(sid: str, token: str):
    """背景完成停止後的 STT flush（帶 30 秒超時保護）"""
    t0 = time.time()
    print(f"[Stop] _finish_transcription 開始", flush=True)

    # 取走等待 flush 的 stream（由 request_stop 存放）
    t_take = time.time()
    flush_stream = stt.take_pending_flush_stream()
    print(f"[Stop] take_pending_flush_stream 耗時 {(time.time()-t_take)*1000:.0f}ms", flush=True)
    if flush_stream is None:
        print(f"[Stop] 無待 flush stream，直接結束 ({time.time()-t0:.2f}s)", flush=True)
        socketio.emit("transcription_ready", {}, room=sid)
        return

    print(f"[Stop] 啟動 _do_flush thread（timeout={_FINISH_TRANSCRIPTION_TIMEOUT}s）", flush=True)
    t1 = time.time()
    flush_done = threading.Event()

    def _do_flush():
        t_flush = time.time()
        print(f"[Stop] _do_flush thread 開始（距 stop 呼叫 {time.time()-t0:.2f}s）", flush=True)
        try:
            stt._flush_stream(flush_stream)
        except Exception as e:
            print(f"[Stop] flush_stream 例外: {e}", flush=True)
        finally:
            print(f"[Stop] _do_flush thread 結束，耗時 {time.time()-t_flush:.2f}s", flush=True)
            flush_done.set()

    flush_thread = threading.Thread(target=_do_flush, daemon=True, name="stt-flush")
    flush_thread.start()
    print(f"[Stop] flush thread 已啟動，開始 Event.wait(timeout={_FINISH_TRANSCRIPTION_TIMEOUT})", flush=True)

    completed = flush_done.wait(timeout=_FINISH_TRANSCRIPTION_TIMEOUT)
    if not completed:
        print(f"[Stop] flush_stream 超時（>{_FINISH_TRANSCRIPTION_TIMEOUT}s），強制放棄", flush=True)
    else:
        print(f"[Stop] Event.wait 結束，flush 總耗時 {time.time()-t1:.2f}s", flush=True)

    print(f"[Stop] _finish_transcription 總耗時 {time.time()-t0:.2f}s，emit transcription_ready", flush=True)
    socketio.emit("transcription_ready", {}, room=sid)


# n_ctx=4096，扣除簡化 prompt(~50) + 輸出(256)，可用 input ≈ 3700 tokens ≈ 3000 中文字
_MAX_CHARS_SINGLE_PASS = 3000


def _compress_long_transcript(full_text: str, sid: str = "") -> str:
    """超長逐字稿：分段送 LLM 各自摘要，再把所有段落小摘要合併，作為最終摘要的輸入。
    chunk_size 設為 2000 字，對應 n_ctx=4096 的 80% 以內（含 system prompt 開銷後仍安全）。
    """
    from cognition import _call_model_stream
    chunk_size = 2000
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
    total = len(chunks)
    _CHUNK_TIMEOUT = 60.0
    mini_parts = []
    for idx, chunk in enumerate(chunks, 1):
        if _is_summary_cancelled():
            print(f"[Summary] map-reduce 已取消（第 {idx}/{total} 段前）", flush=True)
            return "\n\n".join(mini_parts) or full_text
        if sid:
            socketio.emit("summary_progress", {
                "current": idx,
                "total": total,
                "stage": "compress",
            }, room=sid)
        t_chunk = time.time()
        print(f"[Summary] map-reduce 第 {idx}/{total} 段開始，chunk 字數={len(chunk)}", flush=True)
        sys_p = "用繁體中文摘要以下內容成 1 句重點"
        user_p = f"【第 {idx}/{total} 段】\n{chunk}"
        mini_result = [""]
        chunk_done = threading.Event()

        def _do_chunk(sys_p=sys_p, user_p=user_p, result=mini_result, done=chunk_done):
            try:
                for tok in _call_model_stream(sys_p, user_p, max_tokens=64):
                    result[0] += tok
            except Exception as e:
                print(f"[Summary] map-reduce 第 {idx}/{total} 段例外: {e}", flush=True)
            finally:
                done.set()

        threading.Thread(target=_do_chunk, daemon=True).start()
        completed = chunk_done.wait(timeout=_CHUNK_TIMEOUT)
        if not completed:
            print(f"[Summary] 第 {idx}/{total} 段壓縮超時（>{_CHUNK_TIMEOUT}s），已跳過", flush=True)
            mini_parts.append(f"第{idx}段重點：（超時略過）")
            continue
        mini = mini_result[0]
        print(f"[Summary] map-reduce 第 {idx}/{total} 段完成，耗時 {time.time()-t_chunk:.1f}s，輸出 {len(mini)} 字", flush=True)
        mini_parts.append(f"第{idx}段重點：{mini.strip()}")
    return "\n\n".join(mini_parts)


def _generate_summary(mode: str, full_text: str, sid: str):
    """背景生成摘要（支援串流）"""
    global _summary_cancelled
    with _summary_cancelled_lock:
        _summary_cancelled = False
    system_prompt = "請用繁體中文條列本次會議重點，簡潔為主，只輸出重點，不要說明或前言"

    # ── 診斷 log ──────────────────────────────────────────
    char_count = len(full_text)
    use_map_reduce = char_count > _MAX_CHARS_SINGLE_PASS
    chunk_size = 2000
    chunk_count = (char_count + chunk_size - 1) // chunk_size if use_map_reduce else 1
    print(
        f"[Summary] mode={mode} | 逐字稿={char_count} 字 | "
        f"map-reduce={'是，共 ' + str(chunk_count) + ' 段' if use_map_reduce else '否'}",
        flush=True,
    )

    final_max_tokens = 256
    print(f"[Summary] {mode} mode max_tokens={final_max_tokens}（逐字稿 {char_count} 字）", flush=True)

    # 告訴前端準備開始串流
    socketio.emit("summary_start", {"mode": mode}, room=sid)

    from cognition import _call_model_stream

    # 逐字稿過長時先分段壓縮，再送入最終摘要
    if use_map_reduce:
        socketio.emit("summary_token", {
            "mode": mode,
            "token": f"逐字稿共約 {char_count} 字，將分段處理後再彙整摘要...\n\n"
        }, room=sid)
        full_text = _compress_long_transcript(full_text, sid=sid)
        if _is_summary_cancelled():
            socketio.emit("summary_error", {"mode": mode, "message": "摘要已取消"}, room=sid)
            return
        source_label = "以下是各段逐字稿的重點摘要，請根據這些重點產出最終摘要"
    else:
        source_label = "以下是本次會議的完整逐字稿，這是你唯一可以使用的資料來源"

    # 最終推理開始前 emit progress
    if sid:
        final_chunk_count = chunk_count if use_map_reduce else 1
        socketio.emit("summary_progress", {
            "current": final_chunk_count,
            "total": final_chunk_count,
            "stage": "final",
        }, room=sid)

    grounded_prompt = (
        f"【重要】{source_label}。"
        "你的摘要中出現的所有人名、日期、金額、地點、事件，都必須直接來自以下文字，絕對不可自行編造或引用外部知識。\n\n"
        "--- 內容開始 ---\n"
        + full_text
        + "\n--- 內容結束 ---"
    )
    _FINAL_TIMEOUT = 300.0
    accumulated = ""
    last_token_time = time.time()
    t_final = time.time()
    print(f"[Summary] 最終推理開始，mode={mode}", flush=True)

    final_result = {"tokens": [], "error": None}
    final_done = threading.Event()

    def _do_final():
        try:
            for chunk in _call_model_stream(system_prompt, grounded_prompt, max_tokens=final_max_tokens):
                final_result["tokens"].append(chunk)
        except Exception as e:
            final_result["error"] = str(e)
        finally:
            final_done.set()

    threading.Thread(target=_do_final, daemon=True, name="summary-final").start()

    # 在等待期間持續取 token 並串流給前端（輪詢間隔 0.05s）
    while not final_done.is_set():
        elapsed = time.time() - t_final
        if elapsed >= _FINAL_TIMEOUT:
            print(f"[Summary] 最終推理超時（>{_FINAL_TIMEOUT}s），中止", flush=True)
            socketio.emit("summary_error", {"mode": mode, "message": "摘要生成超時，請縮短逐字稿後重試"}, room=sid)
            return
        if _is_summary_cancelled():
            print(f"[Summary] 最終推理已取消", flush=True)
            socketio.emit("summary_error", {"mode": mode, "message": "摘要已取消"}, room=sid)
            return
        # 無 token 超過 10s 時發送 keep_alive，防止前端連線逾時（含等待第一個 token 的期間）
        now = time.time()
        if now - last_token_time >= 10:
            socketio.emit("keep_alive", {"mode": mode}, room=sid)
            print(f"[Summary] keep_alive sent (gap: {now - last_token_time:.1f}s)", flush=True)
            last_token_time = now
        # 取出已累積的 tokens 並串流
        while final_result["tokens"]:
            chunk = final_result["tokens"].pop(0)
            accumulated += chunk
            socketio.emit("summary_token", {"mode": mode, "token": _to_traditional(chunk)}, room=sid)
            last_token_time = time.time()
        final_done.wait(timeout=0.05)

    # 執行緒結束後清空剩餘 token buffer
    for chunk in final_result["tokens"]:
        now = time.time()
        if now - last_token_time >= 10:
            socketio.emit("keep_alive", {"mode": mode}, room=sid)
        accumulated += chunk
        socketio.emit("summary_token", {"mode": mode, "token": _to_traditional(chunk)}, room=sid)
        last_token_time = now

    if final_result["error"]:
        print(f"[Summary] 摘要生成例外: {final_result['error']}", flush=True)
        socketio.emit("summary_error", {"mode": mode, "message": "摘要產生中斷，請重試"}, room=sid)
        return  # 例外路徑不送 summary_done，由前端 summary_error handler 解鎖

    # 僅成功路徑送出 summary_done
    accumulated = _to_traditional(accumulated)
    print(f"[Summary] 最終推理完成，耗時 {time.time()-t_final:.1f}s，輸出 {len(accumulated)} 字", flush=True)
    socketio.emit("summary_done", {"mode": mode, "content": accumulated}, room=sid)


def _export_meeting(meeting_name: str, transcript_override: str = "", summary_overrides: dict = None):
    """背景匯出逐字稿（有摘要時一併匯出）"""
    if transcript_override:
        full_text = transcript_override
    else:
        full_text = "\n".join(
            line["text"] for line in transcript_lines
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
            text = item["text"]
            transcript_lines_out.append(f"[{ts}] {text}")

    transcript_content = "\n".join(transcript_lines_out)

    # 判斷是否有摘要：有則各模式分別寫入獨立檔案，無則只匯出逐字稿
    cached = summary_overrides or {}
    summary_full = cached.get("full", "").strip()

    def _build_summary_file(title: str, content: str) -> str:
        lines = [
            f"會議名稱: {meeting_name}",
            f"匯出時間: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 50,
            "",
            title,
            content,
        ]
        return "\n".join(lines)

    summary_files: list[tuple[str, str]] = []  # (path, content)
    if summary_full:
        summary_files.append((
            os.path.join(_meeting_output_dir(meeting_name), "summary_full.txt"),
            _build_summary_file("【全文摘要】", summary_full),
        ))

    # 寫入檔案（目標資料夾不存在時自動建立）
    export_dir = _meeting_output_dir(meeting_name)
    transcript_path = os.path.join(export_dir, "transcript.txt")

    try:
        os.makedirs(export_dir, exist_ok=True)
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript_content)
        for path, content in summary_files:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
    except PermissionError:
        socketio.emit("export_error", {"message": f"匯出失敗：權限不足，無法寫入至 {export_dir}"})
        return
    except OSError as e:
        if getattr(e, "errno", None) == 28:  # ENOSPC
            socketio.emit("export_error", {"message": "匯出失敗：磁碟空間不足"})
        elif getattr(e, "errno", None) == 2:  # ENOENT (父目錄仍不存在)
            socketio.emit("export_error", {"message": f"匯出失敗：路徑不存在 {export_dir}"})
        else:
            socketio.emit("export_error", {"message": f"匯出失敗：{e}"})
        return
    except Exception as e:
        socketio.emit("export_error", {"message": f"匯出失敗：{e}"})
        return

    saved_files = [{"filename": f"{meeting_name}_transcript.txt", "saved_path": transcript_path}]
    for path, _ in summary_files:
        filename = os.path.basename(path)
        saved_files.append({"filename": f"{meeting_name}_{filename}", "saved_path": path})

    socketio.emit("export_ready", {"files": saved_files})


def _export_summary(meeting_name: str, mode: str, transcript_override: str = "", summary_overrides: dict = None):
    """背景匯出摘要（優先使用前端快取，避免重複推論）"""
    if transcript_override:
        full_text = transcript_override
    else:
        full_text = "\n".join(
            line["text"] for line in transcript_lines
        )

    cached = summary_overrides or {}
    has_cache = bool(cached.get("full"))

    summary_content = ""
    if has_cache or full_text.strip():
        if not has_cache:
            try:
                cached = {"full": summarize_full(full_text)}
            except Exception as e:
                socketio.emit("error", {"message": f"摘要生成失敗，無法匯出：{e}"})
                return

        summary_lines = []
        summary_lines.append(f"會議名稱: {meeting_name}")
        summary_lines.append(f"匯出時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("=" * 50)
        summary_lines.append("")
        summary_lines.append("【全文摘要】")
        summary_lines.append(cached["full"])
        summary_lines.append("")

        summary_content = "\n".join(summary_lines)
    else:
        summary_content = "尚無內容可供匯出"

    mode_suffix = "full"
    export_dir = _meeting_output_dir(meeting_name)
    os.makedirs(export_dir, exist_ok=True)
    summary_filename = f"summary_{mode_suffix}.txt"
    summary_path = os.path.join(export_dir, summary_filename)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_content)

    socketio.emit("export_ready", {
        "files": [
            {
                "filename": f"{meeting_name}_{summary_filename}",
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
