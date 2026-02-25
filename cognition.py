"""認知層 - 本地 LLM（llama.cpp）摘要與校對。"""

import os
import re
import threading
from typing import Optional

import requests

try:
    from llama_cpp import Llama  # type: ignore
except Exception:  # pragma: no cover
    Llama = None

_OPENCC_CONVERTER = None

BACKEND = (os.getenv("COGNITION_BACKEND", "llama_cpp").strip() or "llama_cpp").lower()

DEFAULT_MODEL = os.getenv(
    "LLM_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "models", "llm", "qwen2.5-1.5b-instruct-q5_k_m.gguf"),
).strip()

# 保留既有命名，避免 app.py 與前端協定大改。
AVAILABLE_OLLAMA_MODELS = {
    "qwen2_5_1_5b_gguf": {
        "label": "Qwen2.5 1.5B GGUF Q5_K_M (本地)",
        "model": DEFAULT_MODEL,
    },
}
DEFAULT_OLLAMA_MODEL_KEY = "qwen2_5_1_5b_gguf"

_MODEL = DEFAULT_MODEL
_MODEL_LOCK = threading.Lock()

_LLAMA_INSTANCE: Optional["Llama"] = None
_LLAMA_INSTANCE_PATH = ""
_LLAMA_LOCK = threading.Lock()
_LLAMA_INFER_LOCK = threading.Lock()

# ollama fallback（當 COGNITION_BACKEND=ollama 時）
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b").strip() or "qwen2.5:1.5b"

TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "320"))
LLAMA_CTX = int(os.getenv("LLAMA_CTX", "3072"))
LLAMA_THREADS = int(os.getenv("LLAMA_THREADS", str(max(2, (os.cpu_count() or 4) // 2))))
LLAMA_GPU_LAYERS = int(os.getenv("LLAMA_GPU_LAYERS", "-1"))


def get_current_model() -> str:
    with _MODEL_LOCK:
        return _MODEL


def set_current_model(model: str) -> None:
    global _MODEL, _LLAMA_INSTANCE, _LLAMA_INSTANCE_PATH
    with _MODEL_LOCK:
        _MODEL = (model or "").strip() or DEFAULT_MODEL
    with _LLAMA_LOCK:
        _LLAMA_INSTANCE = None
        _LLAMA_INSTANCE_PATH = ""


def _ensure_llama() -> tuple[Optional["Llama"], str]:
    global _LLAMA_INSTANCE, _LLAMA_INSTANCE_PATH
    model_path = get_current_model()

    if not model_path:
        return None, "[錯誤] 未設定 LLM_MODEL_PATH"
    if not os.path.isfile(model_path):
        return None, f"[錯誤] 找不到 GGUF 模型檔：{model_path}"
    if Llama is None:
        return None, "[錯誤] 未安裝 llama-cpp-python，請先 pip install llama-cpp-python"

    with _LLAMA_LOCK:
        if _LLAMA_INSTANCE is not None and _LLAMA_INSTANCE_PATH == model_path:
            return _LLAMA_INSTANCE, ""

        try:
            _LLAMA_INSTANCE = Llama(
                model_path=model_path,
                n_ctx=LLAMA_CTX,
                n_threads=LLAMA_THREADS,
                n_gpu_layers=LLAMA_GPU_LAYERS,
                verbose=False,
            )
            _LLAMA_INSTANCE_PATH = model_path
            return _LLAMA_INSTANCE, ""
        except Exception as e:  # pragma: no cover
            _LLAMA_INSTANCE = None
            _LLAMA_INSTANCE_PATH = ""
            return None, f"[錯誤] 載入本地模型失敗: {e}"


def _call_llama(system_prompt: str, user_prompt: str) -> str:
    llm, err = _ensure_llama()
    if llm is None:
        return err

    prompt = (
        "你必須嚴格遵守系統規則。\n"
        f"[系統規則]\n{system_prompt}\n\n"
        f"[使用者輸入]\n{user_prompt}\n\n"
        "[請直接輸出最終結果，不要解釋]"
    )
    try:
        # llama-cpp-python inference is not reliably thread-safe when multiple
        # summary requests hit the same in-process model concurrently.
        with _LLAMA_INFER_LOCK:
            out = llm(
                prompt,
                max_tokens=LLM_MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=0.8,
                top_k=40,
                repeat_penalty=1.15,
                stop=["[使用者輸入]", "[系統規則]"],
            )
        text = (out.get("choices", [{}])[0].get("text", "") or "").strip()
        if not text:
            return "[錯誤] 本地模型回傳空結果"
        return text
    except Exception as e:  # pragma: no cover
        return f"[錯誤] 本地模型推論失敗: {e}"


def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": user_prompt,
        "system": system_prompt,
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_predict": LLM_MAX_TOKENS},
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
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": TEMPERATURE, "num_predict": LLM_MAX_TOKENS},
    }
    resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "").strip()


def _call_model(system_prompt: str, user_prompt: str) -> str:
    if BACKEND == "ollama":
        return _call_ollama(system_prompt, user_prompt)
    return _call_llama(system_prompt, user_prompt)


def _normalize_text_for_summary(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    lines = []
    seen = set()
    for line in raw.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if not cleaned:
            continue
        # 過短碎片通常是即時轉寫噪音，先濾掉避免汙染摘要。
        if len(cleaned) <= 1:
            continue
        dedupe_key = re.sub(r"[^\w\u4e00-\u9fff]", "", cleaned).lower()
        if dedupe_key and dedupe_key in seen:
            continue
        if dedupe_key:
            seen.add(dedupe_key)
        lines.append(cleaned)
    return "\n".join(lines).strip()


def _looks_repetitive(text: str) -> bool:
    cleaned = re.sub(r"\s+", "", (text or ""))
    if not cleaned:
        return True

    # 檢查單詞/詞組重複（例如「心肝腎心肝腎...」）。
    tokens = re.findall(r"[\u4e00-\u9fff]{1,4}|[A-Za-z0-9_]+", text)
    if len(tokens) >= 12:
        freq = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        top = max(freq.values())
        if top >= max(6, len(tokens) // 3):
            return True

    # 連續重複字串樣式。
    for n in range(3, 8):
        for i in range(0, len(cleaned) - n * 4):
            seg = cleaned[i : i + n]
            if not seg:
                continue
            if cleaned[i : i + n * 4] == seg * 4:
                return True
    return False


def proofread_text(text: str) -> str:
    system_prompt = (
        "你是一位逐字稿校對員。"
        "請只做最小必要修正：錯字、同音字誤判、斷句與標點。"
        "嚴禁翻譯，嚴禁把英文改成中文或把中文改成英文。"
        "必須保留原始語言與專有名詞；中英混合內容也要維持原樣。"
        "若內容含中文，中文必須使用繁體中文，禁止輸出簡體字。"
        "只輸出修正後的結果，不要加任何說明或前綴；若無需修正，原文輸出。"
    )
    result = _call_model(system_prompt, text)
    return _to_traditional(result)


def _to_traditional(text: str) -> str:
    """盡量將中文轉為繁體；若 opencc 不可用，回傳原文。"""
    global _OPENCC_CONVERTER
    if not text:
        return text
    try:
        if _OPENCC_CONVERTER is None:
            from opencc import OpenCC  # type: ignore
            _OPENCC_CONVERTER = OpenCC("s2t")
        return _OPENCC_CONVERTER.convert(text)
    except Exception:
        return text


def _is_weak_full_summary(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    weak_set = {
        "（資訊不足，僅供參考）",
        "資訊不足",
        "逐字稿未提供足夠資訊。",
        "逐字稿未提供足夠資訊",
    }
    return cleaned in weak_set or (len(cleaned) < 24 and "資訊不足" in cleaned)

def _is_only_insufficient(text: str) -> bool:
    cleaned = re.sub(r"[\s\-\[\]\(\)（）•·]+", "", (text or ""))
    if not cleaned:
        return True
    cleaned = cleaned.replace("僅供參考", "")
    return cleaned in (
        "資訊不足",
        "逐字稿未提供足夠資訊",
        "逐字稿內容不足",
        "暫無可確認待辦",
    ) or (len(cleaned) <= 6 and "資訊不足" in cleaned)


def _compose_full_from_key_points(key_points: str) -> str:
    lines = []
    for raw in (key_points or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[-•]\s*", "", line)
        if not line or "資訊不足" in line:
            continue
        if line:
            lines.append(line.rstrip("。") + "。")
    if not lines:
        return ""
    return " ".join(lines)

def _compose_full_from_transcript(prepared: str) -> str:
    lines = []
    for raw in (prepared or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line in lines:
            continue
        lines.append(line.rstrip("。") + "。")
        if len(lines) >= 3:
            break
    return " ".join(lines).strip()


def _is_sparse_summary_input(prepared: str) -> bool:
    lines = [x.strip() for x in (prepared or "").splitlines() if x.strip()]
    if not lines:
        return True
    if len(lines) == 1 and len(lines[0]) < 80:
        return True
    total_chars = sum(len(x) for x in lines)
    return (len(lines) <= 2 and total_chars < 140) or (len(lines) <= 3 and total_chars < 90)


def _first_fact_line(prepared: str) -> str:
    for raw in (prepared or "").splitlines():
        line = raw.strip()
        if line:
            return line
    return ""


def _extractive_key_points(prepared: str) -> str:
    lines = []
    seen = set()
    for raw in (prepared or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        key = re.sub(r"\s+", "", line)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"• {line}")
        if len(lines) >= 5:
            break
    if not lines:
        return "• 逐字稿內容不足，無法生成重點"
    if len(lines) <= 2:
        lines.append("• （資訊不足，僅供參考）")
    return "\n".join(lines)


def _extractive_action_items(prepared: str) -> str:
    items = []
    seen = set()
    for raw in (prepared or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if "資訊不足" in line:
            continue
        key = re.sub(r"\s+", "", line)
        if key in seen:
            continue
        seen.add(key)
        items.append(f"- [ ] 針對「{line}」補充背景與下一步（待確認）")
        if len(items) >= 5:
            break
    if not items:
        return "- [ ] 補充更多逐字稿內容後再整理待辦項目（待確認）"
    return "\n".join(items)


def _contains_hallucination_artifacts(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    markers = (
        "Human:",
        "Assistant:",
        "User:",
        "System:",
        "Human：",
        "Assistant：",
        "User：",
        "System：",
        "系統提示",
        "請注意]",
        "```",
        "以下是詳細分析",
        "根據提供的信息和系統規則",
        "根據提供的資訊和系統規則",
        "最終結果是",
    )
    if any(m in s for m in markers):
        return True
    return bool(re.search(r"\b(Human|Assistant|User|System)\s*[:：]", s))


def _sanitize_summary_output(text: str, source: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""

    # Remove markdown fences and obvious prompt leakage lines.
    s = s.replace("```plaintext", "").replace("```", "").strip()
    leaked_patterns = (
        r"^\s*(Human|Assistant|User|System)\s*[:：].*$",
        r"^\s*根據提供[的]?資訊和系統規則.*$",
        r"^\s*以下是詳細分析.*$",
        r"^\s*最終結果是[:：]?\s*$",
    )
    kept_lines: list[str] = []
    for line in s.splitlines():
        line = line.strip()
        if not line:
            if kept_lines and kept_lines[-1] != "":
                kept_lines.append("")
            continue
        if any(re.match(p, line) for p in leaked_patterns):
            continue
        kept_lines.append(line)

    # Collapse repeated paragraphs/lines.
    deduped: list[str] = []
    seen = set()
    for line in kept_lines:
        key = re.sub(r"\s+", "", line)
        if line and key and key in seen:
            continue
        if line and key:
            seen.add(key)
        deduped.append(line)

    s = "\n".join(deduped).strip()
    s = re.sub(r"\n{3,}", "\n\n", s)

    # If output is much longer than short source text, keep it conservative.
    src_len = len((source or "").strip())
    if src_len and src_len <= 220 and len(s) > max(220, src_len * 2):
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        s = "\n".join(lines[:4]).strip()

    return s


def _tokenize_overlap_basis(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]{1,6}|[A-Za-z0-9_]+", (text or ""))


def _has_low_source_overlap(output: str, source: str) -> bool:
    out_tokens = _tokenize_overlap_basis(output)
    src_tokens = set(_tokenize_overlap_basis(source))
    if not out_tokens or not src_tokens:
        return False
    # Ignore boilerplate tokens that often appear in prompts/fallbacks.
    boilerplate = {"資訊不足", "僅供參考", "待確認", "摘要", "重點", "全文", "清單"}
    scored = [t for t in out_tokens if t not in boilerplate]
    if not scored:
        return False
    overlap = sum(1 for t in scored if t in src_tokens)
    return overlap / max(1, len(scored)) < 0.35


def summarize_full(text: str) -> str:
    prepared = _normalize_text_for_summary(text)
    if not prepared:
        return "逐字稿內容不足，無法生成摘要。"
    if _is_sparse_summary_input(prepared):
        first = _first_fact_line(prepared)
        if first:
            return f"可確認內容：{first}（資訊不足，僅供參考）"
        return "逐字稿內容較少，暫無法形成完整摘要。"

    system_prompt = (
        "你是一位專業的會議記錄員。"
        "請以抽取為主、必要時可做精簡改寫。"
        "只能根據逐字稿內容，不可補充或推測未提及的資訊。"
        "請輸出 1 段摘要（80-180 字），保留關鍵術語與數字。"
        "若資訊不足，仍要先輸出你能確定的內容，再加上一句「（資訊不足，僅供參考）」作結。"
    )
    result = _call_model(system_prompt, prepared)
    cleaned = _sanitize_summary_output(result, prepared)
    if _contains_hallucination_artifacts(cleaned) or _has_low_source_overlap(cleaned, prepared):
        fallback = _compose_full_from_transcript(prepared)
        return (fallback + "（資訊不足，僅供參考）") if fallback else "逐字稿內容較少，暫無法形成完整摘要。"
    if _is_weak_full_summary(cleaned) or _looks_repetitive(cleaned):
        key_points = summarize_key_points(prepared)
        composed = _compose_full_from_key_points(key_points)
        if composed:
            return composed
        fallback_prompt = (
            "你是一位會議紀錄員。"
            "請只根據逐字稿，輸出一段 2-4 句的抽取式摘要。"
            "至少要包含：討論主題、關鍵觀點或事件、可辨識的結論或未決事項。"
            "不可輸出空白、不可只輸出「資訊不足」。"
            "若資訊不足，也要先輸出可確認內容，再加「（資訊不足，僅供參考）」。"
        )
        retry = _call_model(fallback_prompt, prepared).strip()
        retry = _sanitize_summary_output(retry, prepared)
        if retry and not _is_weak_full_summary(retry) and not _looks_repetitive(retry):
            return retry
        composed_retry = _compose_full_from_key_points(key_points)
        if composed_retry:
            return composed_retry
        composed_from_text = _compose_full_from_transcript(prepared)
        if composed_from_text:
            return composed_from_text
    if _is_only_insufficient(cleaned):
        fallback = _compose_full_from_transcript(prepared)
        return fallback or "已擷取逐字稿重點，但內容仍不足以產生完整摘要。"
    return cleaned or _compose_full_from_transcript(prepared) or "已擷取逐字稿重點，但內容仍不足以產生完整摘要。"


def summarize_key_points(text: str) -> str:
    prepared = _normalize_text_for_summary(text)
    if not prepared:
        return "• 逐字稿內容不足，無法生成重點"
    if _is_sparse_summary_input(prepared):
        first = _first_fact_line(prepared)
        if not first:
            return "• 逐字稿內容不足，無法生成重點"
        return f"• {first}\n• （資訊不足，僅供參考）"

    system_prompt = (
        "你是一位專業的會議記錄員。"
        "請以抽取為主、必要時可做精簡改寫。"
        "只能根據逐字稿內容，不可補充或推測未提及的資訊。"
        "以條列式呈現，每個重點用「•」開頭，列出 3-6 點。"
        "若內容不足，至少輸出 1-2 點可確認資訊，最後補一行「• （資訊不足，僅供參考）」。"
    )
    result = _sanitize_summary_output(_call_model(system_prompt, prepared), prepared)
    if (
        _looks_repetitive(result)
        or _is_only_insufficient(result)
        or _contains_hallucination_artifacts(result)
        or _has_low_source_overlap(result, prepared)
    ):
        return _extractive_key_points(prepared)
    return result


def _is_weak_action_items(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    weak_keywords = (
        "暫無可確認待辦",
        "資訊不足",
        "無法判斷",
        "沒有待辦",
    )
    line_count = len([x for x in cleaned.splitlines() if x.strip()])
    return any(k in cleaned for k in weak_keywords) and line_count <= 2


def _compose_actions_from_key_points(key_points: str) -> str:
    lines = []
    for raw in (key_points or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[-•]\s*", "", line)
        if not line or "資訊不足" in line:
            continue
        if not line.endswith("（待確認）"):
            line = f"{line}（待確認）"
        lines.append(f"- [ ] {line}")
    if not lines:
        return ""
    return "\n".join(lines[:5])


def extract_action_items(text: str) -> str:
    prepared = _normalize_text_for_summary(text)
    if not prepared:
        return "- [ ] 逐字稿內容不足，無法整理待辦"
    if _is_sparse_summary_input(prepared):
        first = _first_fact_line(prepared)
        if first:
            return f"- [ ] 針對「{first}」補充會議背景與具體需求（待確認）"
        return "- [ ] 補充更多逐字稿內容後再整理待辦項目（待確認）"

    system_prompt = (
        "你是一位專業的會議記錄員。"
        "請以抽取為主、必要時可做精簡改寫。"
        "只能根據逐字稿內容，不可補充或推測未提及的資訊。"
        "使用繁體中文，每個項目用「- [ ]」格式呈現。"
        "即使未明確指定負責人與日期，也要整理出 2-5 項可執行跟進事項，並在項目後標記「（待確認）」。"
        "除非逐字稿幾乎沒有內容，否則禁止只輸出「資訊不足」或「暫無待辦」。"
        "每行只能是一個待辦項目，不要輸出其他說明文字。"
    )
    result = _call_model(system_prompt, prepared)
    cleaned = _sanitize_summary_output(result, prepared)
    if (
        not _is_weak_action_items(cleaned)
        and not _looks_repetitive(cleaned)
        and not _contains_hallucination_artifacts(cleaned)
        and not _has_low_source_overlap(cleaned, prepared)
    ):
        return cleaned

    composed = _extractive_action_items(prepared)
    if composed:
        return composed

    return "- [ ] 依逐字稿整理下一步行動項目（待確認）"


def check_health() -> bool:
    if BACKEND == "ollama":
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
    model_path = get_current_model()
    return bool(model_path and os.path.isfile(model_path) and Llama is not None)
