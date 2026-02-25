#!/usr/bin/env python3
"""Run summary benchmarks with mixed Chinese/English transcripts."""

from __future__ import annotations

import argparse
import os
import time

import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cognition import extract_action_items, summarize_full, summarize_key_points, _looks_repetitive

SAMPLES: dict[str, str] = {
    "mix_project_sync": """
    今天先 sync 專案進度。Alice: We finished the landing page v2, but the CTA copy still feels off.
    Bob: 我昨天把 pricing 表格改成三欄，但行動版還會擠。
    Alice: Let's A/B test version A vs B, and we need numbers by Friday.
    另外還有 API rate limit 的問題，後端說可能要加 cache。
    Bob: OK, cache layer can use Redis, ETA two days.
    結論：本週要完成 landing page 文案與 pricing 修正，下週再看廣告投放。
    """,
    "mix_product_planning": """
    產品規劃會議：目前 onboarding drop-off 在 step2，conversion 只有 32%。
    PM: We should simplify the flow and reduce required fields.
    工程：如果要改驗證規則，至少要 3 天開發。
    Design: suggest inline hints instead of modal tips.
    決議：優先做欄位縮減與提示文案，AB test 在 3/15 上線。
    """,
    "mix_support_issue": """
    客服回報：iOS 端登入常失敗，錯誤碼 401/403。
    Engineer:可能是 token refresh race condition, need to review auth flow.
    CS: 有客戶抱怨付款頁 redirect 太慢。
    結論：先修 token refresh，付款頁效能列入下一 sprint。
    """,
}


def _run_one(mode: str, text: str) -> str:
    if mode == "full":
        return summarize_full(text)
    if mode == "key_points":
        return summarize_key_points(text)
    if mode == "action_items":
        return extract_action_items(text)
    if mode == "all":
        full = summarize_full(text)
        key = summarize_key_points(text)
        actions = extract_action_items(text)
        return f"[FULL]\n{full}\n\n[KEY]\n{key}\n\n[ACTION]\n{actions}"
    raise ValueError(f"Unknown mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["full", "key_points", "action_items", "all"])
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    print("== Summary Benchmark ==")
    print(f"backend={os.getenv('COGNITION_BACKEND', 'llama_cpp')} model={os.getenv('LLM_MODEL_PATH', '')}")
    print(f"max_tokens={os.getenv('LLM_MAX_TOKENS', '512')} runs={args.runs} mode={args.mode}")

    for name, text in SAMPLES.items():
        for i in range(args.runs):
            start = time.perf_counter()
            result = _run_one(args.mode, text)
            elapsed = time.perf_counter() - start
            repetitive = _looks_repetitive(result)
            length = len(result)
            print(f"[{name}] run={i+1} time={elapsed:.2f}s len={length} repetitive={repetitive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
