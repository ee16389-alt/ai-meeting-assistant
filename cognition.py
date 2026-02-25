"""認知層 - 本地 GGUF 優先，Ollama 作為 fallback"""

from __future__ import annotations

import json
import os
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
    if _local_model_available():
        try:
            result = _call_local_gguf(system_prompt, user_prompt)
            if result:
                return result
        except Exception as e:
            if os.environ.get("AMA_DISABLE_OLLAMA_FALLBACK") == "1":
                return f"[錯誤] 本地 GGUF 推理失敗: {e}"
    return _call_ollama(system_prompt, user_prompt)


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
    )
    return _call_model(system_prompt, text)


def summarize_full(text: str) -> str:
    """全文摘要"""
    system_prompt = (
        "你是一位專業的會議記錄員。"
        "請以抽取為主、必要時可做精簡改寫。"
        "只能根據逐字稿內容，不可補充或推測未提及的資訊。"
        "請輸出一段精簡摘要，保留原句的關鍵內容與術語。"
        "若逐字稿資訊不足，請輸出「逐字稿未提供足夠資訊」."
    )
    return _call_model(system_prompt, text)


def summarize_key_points(text: str) -> str:
    """重點條列摘要"""
    system_prompt = (
        "你是一位專業的會議記錄員。"
        "請以抽取為主、必要時可做精簡改寫。"
        "只能根據逐字稿內容，不可補充或推測未提及的資訊。"
        "以條列式呈現，每個重點用「•」開頭，列出 3-8 點。"
        "若內容不足，請輸出「逐字稿未提供足夠資訊」."
    )
    return _call_model(system_prompt, text)


def extract_action_items(text: str) -> str:
    """提取待辦清單"""
    system_prompt = (
        "你是一位專業的會議記錄員。"
        "請以抽取為主、必要時可做精簡改寫。"
        "只能根據逐字稿內容，不可補充或推測未提及的資訊。"
        "使用繁體中文，每個項目用「- [ ]」格式呈現。"
        "若沒有待辦事項，請輸出「逐字稿未提供待辦事項」."
    )
    return _call_model(system_prompt, text)


def check_health() -> bool:
    """檢查摘要引擎是否可用（本地 GGUF 或 Ollama）"""
    if _local_model_available():
        return True
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False
