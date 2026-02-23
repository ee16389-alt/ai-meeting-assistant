#!/usr/bin/env python3
"""Prepare desktop packaging assets (GGUF models)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models" / "llm"
DST = ROOT / "desktop" / "models" / "llm"


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"missing source models dir: {SRC}")

    DST.mkdir(parents=True, exist_ok=True)
    copied = 0
    for p in SRC.glob("*.gguf"):
        shutil.copy2(p, DST / p.name)
        copied += 1

    if copied == 0:
        raise SystemExit(f"no .gguf files found in {SRC}")

    print(f"copied {copied} gguf file(s) -> {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
