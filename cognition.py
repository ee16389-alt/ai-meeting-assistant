"""認知層 - 本地 GGUF 優先，Ollama 作為 fallback"""

from __future__ import annotations

import atexit
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import requests

_LLAMA_IMPORT_ERROR: str | None = None
try:
    from llama_cpp import Llama  # type: ignore
except Exception as e:
    Llama = None  # type: ignore[assignment]
    _LLAMA_IMPORT_ERROR = str(e)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"
TEMPERATURE = 0.3

_LOCAL_LLM = None
_LOCAL_LLM_LOCK = threading.Lock()
_LOCAL_LLM_LOAD_ERROR: str | None = None

COMMON_OUTPUT_GUARDRAILS = (
    "禁止輸出「Human:」「Assistant:」或任何角色標籤。"
    "禁止輸出選擇題、問答題、測驗格式。"
    "不得臆測、不得補完逐字稿未提及內容。"
    "只輸出最終答案，不要展示思考過程。"
)

FORBIDDEN_SUMMARY_PATTERNS = (
    "Human:",
    "Assistant:",
    "Q:",
    "A:",
    "請問",
    "A. ",
    "B. ",
    "---",
    "資訊不足，僅供參考",
    "（僅供參考）",
    "逐字稿資訊不足",
)

FILLER_PHRASES = (
    "嗯",
    "嗯嗯",
    "啊",
    "哦",
    "然後",
    "好",
    "好哦",
    "真的假的",
    "好強哦",
    "那我的摘要呢",
    "已經開始錄了",
)

ACTION_KEYWORDS = (
    "要",
    "需要",
    "請",
    "確認",
    "安排",
    "完成",
    "處理",
    "整理",
    "更新",
    "追蹤",
    "提交",
    "修正",
    "記得",
)

CHINESE_ACTION_PREFIXES = (
    "請",
    "需要",
    "要",
    "記得",
    "先",
    "再",
    "安排",
    "確認",
    "處理",
    "整理",
    "更新",
    "追蹤",
)


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _resources_root() -> Path | None:
    if not _is_frozen():
        return None
    # Electron packages backend executable under Resources/backend/
    exe = Path(sys.executable).resolve()
    backend_dir = exe.parent
    if backend_dir.name.lower() == "backend":
        return backend_dir.parent
    return exe.parent


def _load_model_pack_config() -> dict | None:
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
    return None


def _find_local_gguf_path() -> Path | None:
    env_path = os.environ.get("AMA_GGUF_PATH", "").strip()
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p

    cfg = _load_model_pack_config() or {}
    gguf_name = str(cfg.get("ggufFilename", "")).strip()

    candidates = []
    resources = _resources_root()
    if resources:
        candidates.extend([
            resources / "models" / "llm" / gguf_name,
            resources / "models" / "llm",
        ])

    root = _project_root()
    candidates.extend([
        root / "desktop" / "models" / "llm" / gguf_name,
        root / "desktop" / "models" / "llm",
        root / "models" / "llm" / gguf_name,
        root / "models" / "llm",
    ])

    for p in candidates:
        if not p:
            continue
        if p.is_file() and p.suffix.lower() == ".gguf":
            return p
        if p.is_dir():
            files = sorted(p.glob("*.gguf"))
            if files:
                if gguf_name:
                    for f in files:
                        if f.name == gguf_name:
                            return f
                return files[0]
    return None


def _local_model_available() -> bool:
    return Llama is not None and _find_local_gguf_path() is not None


def _load_local_llm():
    global _LOCAL_LLM, _LOCAL_LLM_LOAD_ERROR
    if _LOCAL_LLM is not None:
        return _LOCAL_LLM
    if Llama is None:
        _LOCAL_LLM_LOAD_ERROR = f"llama-cpp-python 無法載入: {_LLAMA_IMPORT_ERROR or 'unknown import error'}"
        return None

    gguf_path = _find_local_gguf_path()
    if gguf_path is None:
        _LOCAL_LLM_LOAD_ERROR = "找不到 GGUF 模型檔"
        return None

    # 第一次嘗試：完整參數（關閉進階指令集相關選項）
    try:
        print(f"[LLM] 嘗試載入模型（完整參數）: {gguf_path}", flush=True)
        _LOCAL_LLM = Llama(
            model_path=str(gguf_path),
            n_ctx=int(os.environ.get("AMA_LLM_CTX", "4096")),
            n_threads=int(os.environ.get("AMA_LLM_THREADS", "4")),
            n_threads_batch=int(os.environ.get("AMA_LLM_THREADS_BATCH", "8")),
            n_batch=int(os.environ.get("AMA_LLM_BATCH", "256")),
            use_mmap=True,
            use_mlock=False,
            tensor_split=None,
            verbose=False,
        )
        _LOCAL_LLM_LOAD_ERROR = None
        print("[LLM] 模型載入成功（完整參數）", flush=True)
        return _LOCAL_LLM
    except Exception as e:
        print(f"[LLM] 第一次載入失敗：{e}", flush=True)
        print("[LLM] 嘗試最小化參數重試（移除 n_threads_batch）...", flush=True)

    # 第二次嘗試：移除可能觸發進階指令集的參數
    try:
        _LOCAL_LLM = Llama(
            model_path=str(gguf_path),
            n_ctx=int(os.environ.get("AMA_LLM_CTX", "4096")),
            n_threads=int(os.environ.get("AMA_LLM_THREADS", "4")),
            n_batch=int(os.environ.get("AMA_LLM_BATCH", "256")),
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
        _LOCAL_LLM_LOAD_ERROR = None
        print("[LLM] 模型載入成功（最小化參數）", flush=True)
        return _LOCAL_LLM
    except Exception as e2:
        print(f"[LLM] 第二次載入失敗：{e2}", flush=True)
        _LOCAL_LLM_LOAD_ERROR = f"本地 GGUF 載入失敗: {e2}"
        return None


def _release_local_llm():
    """程序退出時釋放 Llama 物件，避免殘留記憶體或 GPU context。"""
    global _LOCAL_LLM
    llm = _LOCAL_LLM
    _LOCAL_LLM = None
    if llm is not None:
        try:
            llm.close()
        except Exception:
            pass


atexit.register(_release_local_llm)


def _is_looping(text: str) -> bool:
    """偵測模型是否陷入重複循環。"""
    if len(text) < 80:
        return False
    tail = text[-300:]
    # 方法1：行級重複——最後一行若出現兩次以上（跨行比對）
    lines = [l.strip() for l in tail.splitlines() if l.strip()]
    if len(lines) >= 4:
        last = lines[-1]
        if last and lines.count(last) >= 2:
            return True
    # 方法2：字串片段重複——取最後 60 字，看前 200 字裡是否重複出現
    snippet = tail[-60:]
    if snippet and tail[:-60].count(snippet) >= 2:
        return True
    # 方法3：短周期循環——最後 150 字分成前後兩半，相似度極高
    if len(tail) >= 150:
        a, b = tail[-150:-75], tail[-75:]
        common = sum(ca == cb for ca, cb in zip(a, b))
        if common / 75 >= 0.85:
            return True
    return False


_LLM_STREAM_TIMEOUT = 120  # 單次串流最長允許時間（秒）


def _call_model_stream(system_prompt: str, user_prompt: str, max_tokens: int | None = None):
    """串流輸出模式，讓前端能即時看到字。

    Lock 策略：只在建立 stream 物件時持鎖，建立完成後立即釋放，
    讓 yield chunk 的過程在 lock 外執行，避免長時間佔用 LLM 資源。
    超過 _LLM_STREAM_TIMEOUT 秒後拋出 TimeoutError。

    max_tokens：若為 None 則沿用環境變數 AMA_LLM_MAX_TOKENS（預設 512）。
    """
    llm = _load_local_llm()
    if llm is None:
        yield f"[錯誤] {_LOCAL_LLM_LOAD_ERROR or '本地模型未就緒'}"
        return

    _max_tokens = max_tokens if max_tokens is not None else int(os.environ.get("AMA_LLM_MAX_TOKENS", "512"))

    # ── 建立 stream：持鎖期間只呼叫 create_*，不 yield ──
    stream = None
    use_chat_mode = True
    with _LOCAL_LLM_LOCK:
        try:
            stream = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                repeat_penalty=1.1,
                max_tokens=_max_tokens,
                stream=True,
            )
        except Exception:
            # Chat 模式不可用，改用 completion 模式
            use_chat_mode = False
            prompt = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"
            stream = llm.create_completion(
                prompt=prompt,
                temperature=TEMPERATURE,
                repeat_penalty=1.1,
                max_tokens=_max_tokens,
                stop=["User:", "\nSystem:"],
                stream=True,
            )
    # lock 已釋放，以下 yield 在 lock 外執行

    # ── 消費 stream：lock 外，帶超時保護 ──
    accumulated = ""
    token_usage: dict = {}
    start_time = time.time()

    for chunk in stream:
        if time.time() - start_time > _LLM_STREAM_TIMEOUT:
            raise TimeoutError(f"LLM 串流超過 {_LLM_STREAM_TIMEOUT} 秒，中止輸出")
        if chunk.get("usage"):
            token_usage = chunk["usage"]
        if use_chat_mode:
            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
        else:
            delta = chunk.get("choices", [{}])[0].get("text", "")
        if delta:
            accumulated += delta
            yield delta
            if _is_looping(accumulated):
                break

    if token_usage:
        print(
            f"[LLM] tokens — prompt: {token_usage.get('prompt_tokens', '?')}, "
            f"completion: {token_usage.get('completion_tokens', '?')}, "
            f"total: {token_usage.get('total_tokens', '?')}",
            flush=True,
        )
    else:
        print(f"[LLM] output ~{len(accumulated)} chars (usage not reported by this build)", flush=True)


def _call_local_gguf(system_prompt: str, user_prompt: str) -> str:
    llm = _load_local_llm()
    if llm is None:
        raise RuntimeError(_LOCAL_LLM_LOAD_ERROR or "本地 GGUF 未就緒")

    with _LOCAL_LLM_LOCK:
        try:
            resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=int(os.environ.get("AMA_LLM_MAX_TOKENS", "512")),
            )
            usage = resp.get("usage", {})
            print(
                f"[LLM] tokens — prompt: {usage.get('prompt_tokens', '?')}, "
                f"completion: {usage.get('completion_tokens', '?')}, "
                f"total: {usage.get('total_tokens', '?')}",
                flush=True,
            )
            return (
                resp.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
        except Exception:
            # Fallback for llama.cpp bindings/models without chat template support.
            prompt = (
                f"System:\n{system_prompt}\n\n"
                f"User:\n{user_prompt}\n\n"
                "Assistant:\n"
            )
            resp = llm.create_completion(
                prompt=prompt,
                temperature=TEMPERATURE,
                max_tokens=int(os.environ.get("AMA_LLM_MAX_TOKENS", "512")),
                stop=["User:", "\nSystem:"],
            )
            usage = resp.get("usage", {})
            print(
                f"[LLM] tokens — prompt: {usage.get('prompt_tokens', '?')}, "
                f"completion: {usage.get('completion_tokens', '?')}, "
                f"total: {usage.get('total_tokens', '?')}",
                flush=True,
            )
            return (
                resp.get("choices", [{}])[0]
                .get("text", "")
                .strip()
            )


def _call_model(system_prompt: str, user_prompt: str) -> str:
    gguf_only = os.environ.get("AMA_DISABLE_OLLAMA_FALLBACK") == "1"
    if _local_model_available():
        try:
            result = _call_local_gguf(system_prompt, user_prompt)
            if result:
                return result
        except Exception as e:
            if gguf_only:
                return f"[錯誤] 本地 GGUF 推理失敗: {e}"
    elif gguf_only:
        return f"[錯誤] 本地 GGUF 模式啟用，但模型不可用（{_LOCAL_LLM_LOAD_ERROR or '找不到可用 GGUF/llama-cpp-python'}）"
    return _call_ollama(system_prompt, user_prompt)


def _clean_transcript_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("[錯誤]"):
            continue
        lines.append(s)
    return lines


def _is_filler_line(text: str) -> bool:
    s = re.sub(r"\s+", "", text.strip())
    if not s:
        return True
    if len(s) <= 4 and s in FILLER_PHRASES:
        return True
    if s in FILLER_PHRASES:
        return True
    if len(s) <= 6 and all(ch in "嗯啊哦欸嘿哈好啦呢吧嗎呀喔然後" for ch in s):
        return True
    return False


def _meaningful_transcript_lines(text: str) -> list[str]:
    return [line for line in _clean_transcript_lines(text) if not _is_filler_line(line)]


def _looks_like_correction_text(text: str) -> bool:
    s = text.lower()
    correction_markers = (
        "should be corrected",
        "should be",
        "修正為",
        "應該修正",
        "待確認",
        "meeting record",
        "會議記錄時發現有誤",
    )
    return any(marker in s for marker in correction_markers)


def _extract_action_lines(lines: list[str]) -> list[str]:
    # "要" is too common in Chinese; only count it when paired with other indicators
    STRONG_ACTION_KEYWORDS = (
        "需要", "請", "確認", "安排", "完成", "處理", "整理", "更新", "追蹤", "提交", "修正", "記得",
    )
    actions = []
    for line in lines:
        if _looks_like_correction_text(line):
            continue
        if "待確認" in line or "下一步" in line or "補充背景" in line:
            continue
        stripped = line.strip()
        is_action = any(k in stripped for k in STRONG_ACTION_KEYWORDS)
        # "要" only qualifies when the line starts with an action prefix
        if not is_action and "要" in stripped:
            is_action = any(stripped.startswith(p) for p in CHINESE_ACTION_PREFIXES)
        if is_action:
            actions.append(line)
    return actions


def _action_result_invalid(text: str) -> bool:
    if not text.strip():
        return True
    if _looks_like_correction_text(text):
        return True
    return any(marker in text for marker in ("待確認", "下一步", "補充背景"))


def _is_opening_chatter_line(text: str) -> bool:
    s = text.strip()
    chatter_prefixes = (
        "哎呀",
        "呃",
        "啊",
        "你要",
        "你好好",
        "我可以不說話",
        "但是我常常會忍不住說",
        "一般人就會說",
    )
    return any(s.startswith(prefix) for prefix in chatter_prefixes)


def _line_summary_score(text: str) -> int:
    s = text.strip()
    score = min(len(s), 80)
    if _is_opening_chatter_line(s):
        score -= 40
    if len(s) < 8:
        score -= 30
    discourse_markers = (
        "其實",
        "本來",
        "不是",
        "而是",
        "因為",
        "所以",
        "如果",
        "但是",
        "不過",
        "重點",
        "問題",
        "應該",
        "所謂",
    )
    viewpoint_markers = (
        "我認為",
        "我覺得",
        "他認為",
        "她認為",
        "坦言",
    )
    for marker in discourse_markers + viewpoint_markers:
        if marker in s:
            score += 10
    if "，" in s or "。" in s or "；" in s:
        score += 8
    if s.endswith("嗎") or s.endswith("呢"):
        score -= 14
    if "？" in s or "!" in s or "！" in s:
        score -= 10
    return score


_MIN_SUMMARY_SCORE = 5


def _select_summary_lines(lines: list[str], limit: int) -> list[str]:
    indexed = []
    for idx, line in enumerate(lines):
        score = _line_summary_score(line)
        if idx < 2:
            score -= 12
        indexed.append((score, len(line), idx, line))
    ranked = sorted(indexed, key=lambda item: (item[0], item[1]), reverse=True)
    qualified = [item for item in ranked if item[0] >= _MIN_SUMMARY_SCORE]
    # Fall back to all lines if none pass the threshold
    if not qualified:
        qualified = ranked
    selected = {line for _, _, _, line in qualified[:limit]}
    return [line for line in lines if line in selected][:limit]


def _looks_mostly_chinese(text: str) -> bool:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    letters = sum(1 for ch in text if ch.isalpha())
    return cjk >= max(8, letters)


def _compress_clause(text: str) -> str:
    s = text.strip("，。；、 ")
    s = re.sub(r"^(其實|那麼|所以|然後|就是|而且呢?)", "", s).strip()
    s = re.sub(r"(吧|啊|呀|哦)$", "", s).strip()
    return s


def _chinese_summary_fallback(lines: list[str], mode: str) -> str:
    if not lines:
        return "逐字稿資訊不足"

    chosen = [_compress_clause(line) for line in _select_summary_lines(lines, 5)]
    chosen = [line for line in chosen if line]
    if not chosen:
        return "逐字稿資訊不足"

    if mode == "full":
        parts = chosen[:3]
        if len(parts) == 1:
            return f"這段內容主要提到{parts[0]}。"
        return "這段內容主要提到" + "；".join(parts[:-1]) + f"；並指出{parts[-1]}。"

    if mode == "key_points":
        return "\n".join(f"• {line}" for line in chosen[:5])

    return "逐字稿資訊不足"


def _normalize_action_items_result(text: str, transcript: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "逐字稿資訊不足"

    normalized = []
    for line in lines:
        body = _strip_summary_prefix(line)
        if not body:
            continue
        if _action_result_invalid(body):
            continue
        normalized.append(f"- [ ] {body}")

    return "\n".join(normalized[:5]) if normalized else "逐字稿資訊不足"


def _is_info_insufficient(text: str) -> bool:
    lines = _meaningful_transcript_lines(text)
    if len(lines) < 2:
        return True
    joined = "".join(lines)
    # Exclude spaces/newlines; keep threshold simple for Chinese transcript snippets.
    return len(joined) < 20


def _strip_summary_prefix(line: str) -> str:
    s = line.strip()
    s = re.sub(r"^(•|\- \[ \]|\-)\s*", "", s)
    return s.strip()


def _is_error_result(text: str) -> bool:
    return text.strip().startswith("[錯誤]")


def _summary_is_near_duplicate(summary: str, transcript: str) -> bool:
    """Returns True if the summary is mostly verbatim copies of transcript lines."""
    s_lines = [l.strip() for l in summary.splitlines() if len(l.strip()) > 8]
    t_lines = _meaningful_transcript_lines(transcript)
    if not s_lines or not t_lines:
        return False
    matches = sum(1 for sl in s_lines if any(sl in tl or tl in sl for tl in t_lines))
    return (matches / len(s_lines)) >= 0.7


def _summary_has_out_of_transcript_text(summary: str, transcript: str) -> bool:
    if not summary.strip():
        return True
    if _is_error_result(summary):
        return True
    for p in FORBIDDEN_SUMMARY_PATTERNS:
        if p in summary:
            return True
    if _looks_like_correction_text(summary):
        return True
    if _summary_is_near_duplicate(summary, transcript):
        return True
    return False


def _extractive_fallback(transcript: str, mode: str) -> str:
    lines = _meaningful_transcript_lines(transcript)
    if len(lines) < 2 or len("".join(lines)) < 20:
        return "逐字稿資訊不足"

    if mode == "action_items":
        actions = _extract_action_lines(lines)
        if not actions:
            return "無明確待辦事項"
        return "\n".join(f"- [ ] {line}" for line in actions[:5])

    if _looks_mostly_chinese("".join(lines)):
        return _chinese_summary_fallback(lines, mode)

    ordered = _select_summary_lines(lines, 5)

    if mode == "full":
        return "\n".join(ordered[:3]) if ordered else "逐字稿資訊不足"
    if mode == "key_points":
        return "\n".join(f"• {line}" for line in ordered[:5]) if ordered else "逐字稿資訊不足"
    return "逐字稿資訊不足"


def _insufficient_info_fallback(transcript: str, mode: str) -> str:
    lines = _meaningful_transcript_lines(transcript)
    if not lines:
        return "逐字稿資訊不足"

    chosen = _select_summary_lines(lines, 5)
    compressed = [_compress_clause(l) for l in chosen if _compress_clause(l)]

    if mode == "full":
        if not compressed:
            return "逐字稿資訊不足"
        if len(compressed) == 1:
            return f"這段內容主要提到{compressed[0]}。"
        return "這段內容主要提到" + "；".join(compressed[:-1]) + f"；並指出{compressed[-1]}。"

    if mode == "key_points":
        base = compressed if compressed else [l for l in chosen if l]
        if not base:
            return "逐字稿資訊不足"
        return "\n".join(f"• {line}" for line in base[:5])

    if mode == "action_items":
        actions = _extract_action_lines(chosen)
        if not actions:
            return "無明確待辦事項"
        return "\n".join(f"- [ ] {line}" for line in actions[:5])

    if not compressed:
        return "逐字稿資訊不足"
    return "這段內容主要提到" + "；".join(compressed) + "。"


def _summarize_with_guard(mode: str, text: str, system_prompt: str) -> str:
    if _is_info_insufficient(text):
        return _insufficient_info_fallback(text, mode)

    result = _call_model(system_prompt, text).strip()
    if _is_error_result(result):
        return _extractive_fallback(text, mode)
    if mode == "action_items":
        # Model correctly determined no action items
        if result in ("無明確待辦事項", "逐字稿資訊不足"):
            return result
        normalized = _normalize_action_items_result(result, text)
        if normalized not in ("逐字稿資訊不足", ""):
            return normalized
        return _extractive_fallback(text, mode)
    if _summary_has_out_of_transcript_text(result, text):
        return _extractive_fallback(text, mode)
    return result or "逐字稿資訊不足"


def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    """呼叫 Ollama REST API，回傳生成結果"""
    payload = {
        "model": MODEL,
        "prompt": user_prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if resp.status_code == 404:
            return _call_ollama_chat(system_prompt, user_prompt)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "[錯誤] 無法連線至 Ollama，請確認 ollama serve 已啟動"
    except requests.exceptions.Timeout:
        return "[錯誤] Ollama 回應逾時"
    except Exception as e:
        return f"[錯誤] Ollama 呼叫失敗: {e}"


def _call_ollama_chat(system_prompt: str, user_prompt: str) -> str:
    """使用 /api/chat 作為 fallback"""
    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": TEMPERATURE,
        },
    }
    resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "").strip()



def summarize_full(text: str):
    """全文摘要（streaming generator，逐 token yield）"""
    system_prompt = (
        "你是一位專業的會議記錄員。"
        "請用繁體中文輸出。"
        "必須用自己的語言重新整合與表達，絕對禁止逐句照抄逐字稿原文。"
        "只能根據逐字稿內容，不可補充或推測未提及的資訊。"
        "請輸出 2-4 句話的精簡摘要，整理主要脈絡、重點與結論。"
        "若資訊不足，僅輸出「逐字稿資訊不足」。"
        + COMMON_OUTPUT_GUARDRAILS
    )
    if _is_info_insufficient(text):
        yield _insufficient_info_fallback(text, "full")
        return
    yield from _call_model_stream(system_prompt, text)


def summarize_key_points(text: str):
    """重點條列摘要（streaming generator，逐 token yield）"""
    system_prompt = (
        "你是一位專業的會議記錄員。"
        "請用繁體中文輸出。"
        "必須用自己的語言重新整合與表達，絕對禁止逐句照抄逐字稿原文。"
        "只能根據逐字稿內容，不可補充或推測未提及的資訊。"
        "以條列式呈現，每個重點用「•」開頭，根據內容多寡自行決定數量，不強制固定點數，最多不超過 100 點。"
        "每個重點應為完整的觀念或結論，而非單一句子片段。"
        "若資訊不足，僅輸出「逐字稿資訊不足」。"
        + COMMON_OUTPUT_GUARDRAILS
    )
    if _is_info_insufficient(text):
        yield _insufficient_info_fallback(text, "key_points")
        return
    yield from _call_model_stream(system_prompt, text)


def summarize_all_in_one(text: str) -> dict:
    """一次性產出全文摘要與重點條列，節省推論時間。"""
    system_prompt = (
        "你是一位專業的會議記錄員。請用繁體中文輸出。\n"
        "請針對提供的逐字稿，一次性提供以下兩個部分的內容：\n"
        "1. 【全文摘要】：用 2-4 句話精簡整理主要脈絡與結論。\n"
        "2. 【重點條列】：列出 3-5 個核心重點，以「•」開頭。\n"
        "規則：禁止照抄原文、禁止臆測、禁止輸出角色標籤。若資訊不足，請在該項標註「逐字稿資訊不足」。"
        + COMMON_OUTPUT_GUARDRAILS
    )

    if _is_info_insufficient(text):
        return {
            "full": _insufficient_info_fallback(text, "full"),
            "key_points": _insufficient_info_fallback(text, "key_points"),
        }

    raw_result = _call_model(system_prompt, text).strip()

    parts = {"full": "", "key_points": ""}

    if "【全文摘要】" in raw_result:
        parts["full"] = raw_result.split("【全文摘要】")[-1].split("【重點條列】")[0].strip()
    if "【重點條列】" in raw_result:
        parts["key_points"] = raw_result.split("【重點條列】")[-1].strip()

    if not parts["full"] and not parts["key_points"]:
        return {"full": raw_result, "key_points": "請見上方摘要"}

    return parts


def check_health() -> bool:
    """檢查摘要引擎是否可用（本地 GGUF 或 Ollama）"""
    if _local_model_available():
        return True
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=1)
        return resp.status_code == 200
    except Exception:
        return False


def summary_engine_status() -> dict:
    """提供摘要引擎目前的可用性與診斷資訊。"""
    gguf_path = _find_local_gguf_path()
    ollama_ok = False
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=1)
        ollama_ok = resp.status_code == 200
    except Exception:
        ollama_ok = False

    if Llama is not None and gguf_path is not None:
        return {
            "ready": True,
            "mode": "local_gguf",
            "gguf_path": str(gguf_path),
            "error": None,
        }

    if ollama_ok:
        return {
            "ready": True,
            "mode": "ollama",
            "gguf_path": str(gguf_path) if gguf_path else None,
            "error": _LOCAL_LLM_LOAD_ERROR or _LLAMA_IMPORT_ERROR,
        }

    error = _LOCAL_LLM_LOAD_ERROR
    if error is None and Llama is None:
        error = f"llama-cpp-python 無法載入: {_LLAMA_IMPORT_ERROR or 'unknown import error'}"
    if error is None and gguf_path is None:
        error = "找不到 GGUF 模型檔"

    return {
        "ready": False,
        "mode": "unavailable",
        "gguf_path": str(gguf_path) if gguf_path else None,
        "error": error,
    }
