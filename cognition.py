"""認知層 - Ollama/Qwen2.5 校對與摘要"""

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:1.5b"
TEMPERATURE = 0.0


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
    return _call_ollama(system_prompt, text)


def summarize_full(text: str) -> str:
    """全文摘要"""
    system_prompt = (
        "你是一位專業的會議記錄員。"
        "請以抽取為主、必要時可做精簡改寫。"
        "只能根據逐字稿內容，不可補充或推測未提及的資訊。"
        "請輸出一段精簡摘要，保留原句的關鍵內容與術語。"
        "若逐字稿資訊不足，請輸出「逐字稿未提供足夠資訊」."
    )
    return _call_ollama(system_prompt, text)


def summarize_key_points(text: str) -> str:
    """重點條列摘要"""
    system_prompt = (
        "你是一位專業的會議記錄員。"
        "請以抽取為主、必要時可做精簡改寫。"
        "只能根據逐字稿內容，不可補充或推測未提及的資訊。"
        "以條列式呈現，每個重點用「•」開頭，列出 3-8 點。"
        "若內容不足，請輸出「逐字稿未提供足夠資訊」."
    )
    return _call_ollama(system_prompt, text)


def extract_action_items(text: str) -> str:
    """提取待辦清單"""
    system_prompt = (
        "你是一位專業的會議記錄員。"
        "請以抽取為主、必要時可做精簡改寫。"
        "只能根據逐字稿內容，不可補充或推測未提及的資訊。"
        "使用繁體中文，每個項目用「- [ ]」格式呈現。"
        "若沒有待辦事項，請輸出「逐字稿未提供待辦事項」."
    )
    return _call_ollama(system_prompt, text)


def check_health() -> bool:
    """檢查 Ollama 是否可用"""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False
