#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
OUT_DIR = ROOT / "desktop" / "backend"


def run(cmd):
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main():

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data_sep = ";" if os.name == "nt" else ":"

    run([
        "pyinstaller",
        "--clean",
        "--onedir",
        "--noconsole",
        "--name", "ai_meeting_backend",
        "--add-data", f"templates{data_sep}templates",
        "--add-data", f"static{data_sep}static",
        "--collect-all", "llama_cpp",
        "--collect-all", "opencc",
        "--collect-all", "sherpa_onnx",
        "--collect-binaries", "sherpa_onnx",
        "--hidden-import", "engineio.async_drivers.threading",
        "--hidden-import", "simple_websocket",
        "--hidden-import", "llama_cpp",
        "app.py",
    ])

    built_dir = DIST / "ai_meeting_backend"
    if not built_dir.exists():
        raise SystemExit("build failed: onedir output not found")

    # Copy the entire onedir bundle (exe + _internal/) into desktop/backend/
    shutil.copytree(built_dir, OUT_DIR, dirs_exist_ok=True)
    exe_name = "ai_meeting_backend.exe" if os.name == "nt" else "ai_meeting_backend"
    print(f"backend onedir bundle -> {OUT_DIR / exe_name}")


if __name__ == "__main__":
    main()
