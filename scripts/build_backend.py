#!/usr/bin/env python3
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
OUT_DIR = ROOT / "desktop" / "backend"
OUT_BIN = OUT_DIR / "ai_meeting_backend"
DATA_SEP = os.pathsep


def run(cmd):
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def add_data_arg(cmd, src: str, dest: str, *, required: bool = False):
    src_path = ROOT / src
    if not src_path.exists():
        if required:
            raise SystemExit(f"build failed: required path not found: {src}")
        print(f"Skipping missing PyInstaller data path: {src}")
        return
    cmd.extend(["--add-data", f"{src}{DATA_SEP}{dest}"])


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "pyinstaller",
        "--clean",
        "--onefile",
        "--name", "ai_meeting_backend",
        "--hidden-import", "engineio.async_drivers.threading",
        "--hidden-import", "socketio.async_drivers.threading",
        "app.py",
    ]
    add_data_arg(cmd, "templates", "templates", required=True)
    add_data_arg(cmd, "static", "static")
    add_data_arg(cmd, "models/sherpa-onnx", "models/sherpa-onnx")
    run(cmd)

    built = DIST / "ai_meeting_backend"
    if not built.exists():
        raise SystemExit("build failed: backend binary not found")
    shutil.copy2(built, OUT_BIN)
    print(f"backend binary -> {OUT_BIN}")


if __name__ == "__main__":
    main()
