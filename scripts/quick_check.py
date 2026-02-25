#!/usr/bin/env python3
"""Quick health checks for AI Meeting Assistant."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_WHISPER_CPP_BIN = os.path.join(PROJECT_ROOT, "third_party", "whisper-bin", "bin", "whisper-cli")
DEFAULT_WHISPER_CPP_MODEL_DIR = "/Users/minashih/models/whisper.cpp"
REQUIRED_WCPP_MODELS = ("ggml-small.bin", "ggml-medium.bin", "ggml-large-v3.bin")
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


def _ensure_root_in_path() -> None:
    root = os.path.dirname(os.path.dirname(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)


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


def check_opencc() -> bool:
    try:
        from opencc import OpenCC  # type: ignore
        _ = OpenCC("s2t")
        _ok("opencc available")
        return True
    except Exception:
        _warn("opencc not available (Traditional conversion disabled)")
        return False


def check_meeting_name() -> None:
    _ensure_root_in_path()
    from utils import sanitize_meeting_name
    tests = [
        "../secret",
        "My Meeting 001",
        "會議/測試..",
        "   ",
    ]
    for t in tests:
        print(f"[CHECK] sanitize '{t}' -> '{sanitize_meeting_name(t, 'fallback')}'")


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
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code != 200:
            _warn(f"Ollama API not ready (status {resp.status_code})")
            return False
        _ok("Ollama API reachable")
        return True
    except Exception:
        _warn("Ollama API not reachable (ollama serve not running?)")
    return False


def check_sherpa_onnx_model() -> bool:
    model_dir = os.getenv("SHERPA_ONNX_MODEL_DIR", "").strip() or DEFAULT_SHERPA_ONNX_MODEL_DIR
    required_any = {
        "encoder": ("encoder-epoch-99-avg-1.int8.onnx", "encoder-epoch-99-avg-1.onnx"),
        "joiner": ("joiner-epoch-99-avg-1.int8.onnx", "joiner-epoch-99-avg-1.onnx"),
        "decoder": ("decoder-epoch-99-avg-1.onnx",),
    }
    if not os.path.isdir(model_dir):
        _warn(f"sherpa-onnx model dir not found: {model_dir}")
        return False
    files = set(os.listdir(model_dir))
    ok = True
    if "tokens.txt" not in files:
        _warn("sherpa-onnx missing tokens.txt")
        ok = False
    for key, choices in required_any.items():
        if not any(c in files for c in choices):
            _warn(f"sherpa-onnx missing {key} model: {choices}")
            ok = False
    if ok:
        _ok("sherpa-onnx model files present")
    return ok


def check_load_medium() -> bool:
    try:
        _ensure_root_in_path()
        from stt_engine import STTEngine
        _ = STTEngine(model_size="medium", engine="faster-whisper")
        _ok("Faster-Whisper medium loaded")
        return True
    except Exception as e:
        _fail(f"Failed to load Faster-Whisper medium: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-medium", action="store_true", help="Load Faster-Whisper medium model")
    args = parser.parse_args()

    print("== Quick Health Check ==")
    ok_imports, _ = check_imports()
    ok_opencc = check_opencc()
    ok_whisper_cpp = check_whisper_cpp()
    ok_sherpa = check_sherpa_onnx_model()
    ok_ollama = check_ollama()
    check_meeting_name()

    ok_medium = True
    if args.load_medium:
        ok_medium = check_load_medium()

    all_ok = ok_imports and ok_opencc and ok_whisper_cpp and ok_sherpa and ok_ollama and ok_medium
    if all_ok:
        _ok("Quick check passed")
        return 0
    _warn("Quick check has warnings; see above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
