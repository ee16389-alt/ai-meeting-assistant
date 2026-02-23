#!/usr/bin/env python3
"""Offline readiness checks for AI Meeting Assistant."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Tuple


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_WHISPER_CPP_BIN = os.path.join(PROJECT_ROOT, "third_party", "whisper-bin", "bin", "whisper-cli")
DEFAULT_WHISPER_CPP_MODEL_DIR = "/Users/minashih/models/whisper.cpp"
REQUIRED_WCPP_MODELS = ("ggml-small.bin", "ggml-medium.bin", "ggml-large-v3.bin")
OLLAMA_URL = "http://localhost:11434/api/tags"
OLLAMA_MODEL = "qwen2.5:1.5b"
DEFAULT_SHERPA_ONNX_MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "sherpa-onnx",
    "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
)


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def check_imports() -> Tuple[bool, list[str]]:
    missing = []
    for mod in ("flask", "flask_socketio", "faster_whisper", "numpy", "requests", "sherpa_onnx"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        _fail(f"Missing Python packages: {', '.join(missing)}")
        return False, missing
    _ok("Python dependencies look good")
    return True, []


def check_whisper_cpp() -> bool:
    ok = True
    bin_path = os.getenv("WHISPER_CPP_BIN", "").strip() or DEFAULT_WHISPER_CPP_BIN
    model_dir = os.getenv("WHISPER_CPP_MODEL_DIR", "").strip() or DEFAULT_WHISPER_CPP_MODEL_DIR

    if not os.path.isfile(bin_path):
        _warn(f"whisper.cpp bin not found: {bin_path}")
        ok = False
    else:
        _ok(f"whisper.cpp bin found: {bin_path}")

    if not os.path.isdir(model_dir):
        _warn(f"whisper.cpp model dir not found: {model_dir}")
        return False
    missing = [m for m in REQUIRED_WCPP_MODELS if not os.path.isfile(os.path.join(model_dir, m))]
    if missing:
        _warn(f"Missing whisper.cpp models: {', '.join(missing)}")
        ok = False
    else:
        _ok("whisper.cpp models present (small/medium/large-v3)")
    return ok


def check_ollama() -> bool:
    try:
        import requests  # type: ignore
    except Exception:
        _warn("requests not available; cannot check Ollama API")
        return False

    try:
        resp = requests.get(OLLAMA_URL, timeout=3)
        if resp.status_code != 200:
            _warn(f"Ollama API not ready (status {resp.status_code})")
            return False
        data = resp.json()
        models = [m.get("name", "") for m in data.get("models", [])]
        if OLLAMA_MODEL in models:
            _ok(f"Ollama model available: {OLLAMA_MODEL}")
            return True
        _warn(f"Ollama running but model missing: {OLLAMA_MODEL}")
        return False
    except Exception:
        _warn("Ollama API not reachable (ollama serve not running?)")

    if shutil.which("ollama"):
        try:
            out = subprocess.check_output(["ollama", "list"], text=True)
            if OLLAMA_MODEL in out:
                _ok(f"Ollama model listed by CLI: {OLLAMA_MODEL}")
                return True
            _warn(f"Ollama CLI available but model missing: {OLLAMA_MODEL}")
            return False
        except Exception:
            _warn("Ollama CLI available but failed to run")
    else:
        _warn("Ollama CLI not found in PATH")
    return False


def check_sherpa_onnx_model() -> bool:
    model_dir = os.getenv("SHERPA_ONNX_MODEL_DIR", "").strip() or DEFAULT_SHERPA_ONNX_MODEL_DIR
    if not os.path.isdir(model_dir):
        _warn(f"sherpa-onnx model dir not found: {model_dir}")
        return False
    files = set(os.listdir(model_dir))
    ok = True
    if "tokens.txt" not in files:
        _warn("sherpa-onnx missing tokens.txt")
        ok = False
    if not any(f in files for f in ("encoder-epoch-99-avg-1.int8.onnx", "encoder-epoch-99-avg-1.onnx")):
        _warn("sherpa-onnx missing encoder model")
        ok = False
    if not any(f in files for f in ("joiner-epoch-99-avg-1.int8.onnx", "joiner-epoch-99-avg-1.onnx")):
        _warn("sherpa-onnx missing joiner model")
        ok = False
    if "decoder-epoch-99-avg-1.onnx" not in files:
        _warn("sherpa-onnx missing decoder model")
        ok = False
    if ok:
        _ok("sherpa-onnx model files present")
    return ok


def main() -> int:
    print("== Offline Readiness Check ==")
    ok_imports, _ = check_imports()
    ok_wcpp = check_whisper_cpp()
    ok_ollama = check_ollama()
    ok_sherpa = check_sherpa_onnx_model()

    all_ok = ok_imports and ok_wcpp and ok_ollama and ok_sherpa
    if all_ok:
        _ok("Offline readiness looks good")
        return 0
    _warn("Offline readiness incomplete; see warnings above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
