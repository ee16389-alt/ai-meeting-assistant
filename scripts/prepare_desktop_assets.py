#!/usr/bin/env python3
"""Prepare desktop packaging assets (GGUF / Sherpa-ONNX models)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGUF_SRC = ROOT / "models" / "llm"
GGUF_DST = ROOT / "desktop" / "models" / "llm"
SHERPA_SRC = ROOT / "models" / "sherpa-onnx"
SHERPA_DST = ROOT / "desktop" / "models" / "sherpa-onnx"


def _copy_gguf() -> int:
    if not GGUF_SRC.exists():
        raise SystemExit(f"missing source models dir: {GGUF_SRC}")

    GGUF_DST.mkdir(parents=True, exist_ok=True)
    copied = 0
    for p in GGUF_SRC.glob("*.gguf"):
        shutil.copy2(p, GGUF_DST / p.name)
        copied += 1

    if copied == 0:
        raise SystemExit(f"no .gguf files found in {GGUF_SRC}")
    return copied


def _copy_sherpa() -> int:
    if not SHERPA_SRC.exists():
        raise SystemExit(f"missing sherpa model dir: {SHERPA_SRC}")
    if SHERPA_DST.exists():
        shutil.rmtree(SHERPA_DST)
    shutil.copytree(SHERPA_SRC, SHERPA_DST, ignore=shutil.ignore_patterns("__MACOSX", ".DS_Store"))
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-sherpa",
        action="store_true",
        help="also copy models/sherpa-onnx into desktop/models/sherpa-onnx",
    )
    args = parser.parse_args()

    copied_gguf = _copy_gguf()
    print(f"copied {copied_gguf} gguf file(s) -> {GGUF_DST}")
    if args.include_sherpa:
        _copy_sherpa()
        print(f"copied sherpa-onnx model dir -> {SHERPA_DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
