"""認知層 - 本地 GGUF 優先，Ollama 作為 fallback"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from pathlib import Path

import requests

try:
    from llama_cpp import Llama  # type: ignore
except Exception:
    Llama = None  # type: ignore[assignment]

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:1.5b"
TEMPERATURE = 0.0

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
        _LOCAL_LLM_LOAD_ERROR = "llama-cpp-python 未安裝"
        return None

    gguf_path = _find_local_gguf_path()
    if gguf_path is None:
        _LOCAL_LLM_LOAD_ERROR = "找不到 GGUF 模型檔"
        return None

    try:
        _LOCAL_LLM = Llama(
            model_path=str(gguf_path),
            n_ctx=int(os.environ.get("AMA_LLM_CTX", "4096")),
            n_threads=max(1, (os.cpu_count() or 4) - 1),
            verbose=False,
        )
        _LOCAL_LLM_LOAD_ERROR = None
        return _LOCAL_LLM
    except Exception as e:
        _LOCAL_LLM_LOAD_ERROR = f"本地 GGUF 載入失敗: {e}"
        return None


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


def _is_info_insufficient(text: str) -> bool:
    lines = _clean_transcript_lines(text)
    if len(lines) < 2:
        return True
    joined = "".join(lines)
    # Exclude spaces/newlines; keep threshold simple for Chinese transcript snippets.
    return len(joined) < 20


def _strip_summary_prefix(line: str) -> str:
    s = line.strip()
    s = re.sub(r"^(•|\- \[ \]|\-)\s*", "", s)
    return s.strip()


def _summary_has_out_of_transcript_text(summary: str, transcript: str) -> bool:
    if not summary.strip():
        return True
    for p in FORBIDDEN_SUMMARY_PATTERNS:
        if p in summary:
            return True

    source_lines = _clean_transcript_lines(transcript)
    if not source_lines:
        return True

    for raw in summary.splitlines():
        line = _strip_summary_prefix(raw)
        if not line:
            continue
        if line == "逐字稿資訊不足":
            continue
        # Require each content line to be an exact substring of original transcript lines.
        if not any(line in src for src in source_lines):
            return True
    return False


def _extractive_fallback(transcript: str, mode: str) -> str:
    lines = _clean_transcript_lines(transcript)
    if len(lines) < 2 or len("".join(lines)) < 20:
        return "逐字稿資訊不足"

    # Prefer longer lines and keep original wording only.
    ranked = sorted(lines, key=len, reverse=True)
    selected = ranked[:5]
    # Restore source order after selecting.
    selected_set = set(selected)
    ordered = [line for line in lines if line in selected_set][:5]

    if mode == "full":
        return "\n".join(ordered[:3]) if ordered else "逐字稿資訊不足"
    if mode == "key_points":
        return "\n".join(f"• {line}" for line in ordered[:5]) if ordered else "逐字稿資訊不足"
    if mode == "action_items":
        return "\n".join(f"- [ ] {line}" for line in ordered[:5]) if ordered else "逐字稿資訊不足"
    return "逐字稿資訊不足"


def _insufficient_info_fallback(transcript: str, mode: str) -> str:
    lines = _clean_transcript_lines(transcript)
    if not lines:
        return "逐字稿資訊不足"

    note = "（逐字稿資訊不足，故此呈現）"
    chosen = lines[:5]

    if mode == "full":
        return note + "\n" + "\n".join(chosen[:3])

    if mode == "key_points":
        keywords = ("要", "需要", "請", "先", "後", "確認", "聯絡", "安排", "完成", "處理")
        inferred = [line for line in chosen if any(k in line for k in keywords)]
        base = inferred if inferred else chosen
        return note + "\n" + "\n".join(f"• {line}" for line in base[:5])

    if mode == "action_items":
        return note + "\n" + "\n".join(f"- [ ] {line}" for line in chosen[:5])

    return note + "\n" + "\n".join(chosen[:3])


def _summarize_with_guard(mode: str, text: str, system_prompt: str) -> str:
    if _is_info_insufficient(text):
        return _insufficient_info_fallback(text, mode)

    result = _call_model(system_prompt, text).strip()
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


def proofread_text(text: str) -> str:
    """修正 STT 逐字稿的錯字、同音字、標點"""
    system_prompt = (
        "你是一位專業的繁體中文文字校對員。"
        "請修正以下語音辨識逐字稿中的錯字、同音字誤判和標點符號錯誤。"
        "只輸出修正後的結果，不要加任何說明或前綴。"
        "如果文字已經正確，直接輸出原文。"
        + COMMON_OUTPUT_GUARDRAILS
    )
    return _call_model(system_prompt, text)


def summarize_full(text: str) -> str:
    """全文摘要"""
    system_prompt = (
        "你是一位專業的會議記錄員。"
        "請以抽取為主、必要時可做精簡改寫。"
        "只能根據逐字稿內容，不可補充或推測未提及的資訊。"
        "請輸出一段精簡摘要，保留原句的關鍵內容與術語。"
        "若資訊不足，僅輸出「逐字稿資訊不足」。"
        + COMMON_OUTPUT_GUARDRAILS
    )
    return _summarize_with_guard("full", text, system_prompt)


def summarize_key_points(text: str) -> str:
    """重點條列摘要"""
    system_prompt = (
        "你是一位專業的會議記錄員。"
        "請以抽取為主、必要時可做精簡改寫。"
        "只能根據逐字稿內容，不可補充或推測未提及的資訊。"
        "以條列式呈現，每個重點用「•」開頭，列出 3-8 點。"
        "若資訊不足，僅輸出「逐字稿資訊不足」。"
        + COMMON_OUTPUT_GUARDRAILS
    )
    return _summarize_with_guard("key_points", text, system_prompt)


def extract_action_items(text: str) -> str:
    """提取待辦清單"""
    system_prompt = (
        "你是一位專業的會議記錄員。"
        "請以抽取為主、必要時可做精簡改寫。"
        "只能根據逐字稿內容，不可補充或推測未提及的資訊。"
        "使用繁體中文，每個項目用「- [ ]」格式呈現。"
        "若資訊不足或無待辦事項，僅輸出「逐字稿資訊不足」。"
        + COMMON_OUTPUT_GUARDRAILS
    )
    return _summarize_with_guard("action_items", text, system_prompt)


def check_health() -> bool:
    """檢查摘要引擎是否可用（本地 GGUF 或 Ollama）"""
    if _local_model_available():
        return True
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False
