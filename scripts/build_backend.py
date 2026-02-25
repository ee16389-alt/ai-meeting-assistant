#!/usr/bin/env python3
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
OUT_DIR = ROOT / "desktop" / "backend"
OUT_BIN = OUT_DIR / "ai_meeting_backend"


def run(cmd):
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run([
        "pyinstaller",
        "--clean",
        "--onefile",
        "--name", "ai_meeting_backend",
        "--add-data", "templates:templates",
        "app.py",
    ])

    candidates = [
        DIST / "ai_meeting_backend",
        DIST / "ai_meeting_backend.exe",
    ]
    built = next((p for p in candidates if p.exists()), None)
    if built is None:
        raise SystemExit("build failed: backend binary not found")
    shutil.copy2(built, OUT_BIN)
    print(f"backend binary -> {OUT_BIN}")


if __name__ == "__main__":
    main()
