#!/usr/bin/env python3
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
OUT_DIR = ROOT / "desktop" / "backend"
EXE_SUFFIX = ".exe" if os.name == "nt" else ""
OUT_BIN = OUT_DIR / f"ai_meeting_backend{EXE_SUFFIX}"
DATA_SEP = os.pathsep
# Desktop packaging already resolves Sherpa model from an external model pack path.
# Keep backend lean by default; opt in only when a fully self-contained binary is needed.
EMBED_SHERPA_ONNX = os.getenv("EMBED_SHERPA_ONNX", "0").strip().lower() not in {"0", "false", "no"}
PYINSTALLER_MODE = (os.getenv("PYINSTALLER_MODE", "onedir").strip() or "onedir").lower()


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
    if PYINSTALLER_MODE not in {"onefile", "onedir"}:
        raise SystemExit("build failed: PYINSTALLER_MODE must be onefile or onedir")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "pyinstaller",
        "--clean",
        "--name", "ai_meeting_backend",
        "--hidden-import", "engineio.async_drivers.threading",
        "--hidden-import", "socketio.async_drivers.threading",
    ]
    cmd.append("--onefile" if PYINSTALLER_MODE == "onefile" else "--onedir")
    add_data_arg(cmd, "templates", "templates", required=True)
    add_data_arg(cmd, "static", "static")
    if EMBED_SHERPA_ONNX:
        add_data_arg(cmd, "models/sherpa-onnx", "models/sherpa-onnx")
    else:
        print("Skipping embedded sherpa-onnx model bundle (EMBED_SHERPA_ONNX disabled)")
    cmd.append("app.py")
    run(cmd)

    if PYINSTALLER_MODE == "onefile":
        built = DIST / f"ai_meeting_backend{EXE_SUFFIX}"
        if not built.exists():
            raise SystemExit("build failed: backend binary not found")
        shutil.copy2(built, OUT_BIN)
        print(f"backend binary (onefile) -> {OUT_BIN}")
        return

    built_dir = DIST / "ai_meeting_backend"
    built_bin = built_dir / f"ai_meeting_backend{EXE_SUFFIX}"
    if not built_bin.exists():
        raise SystemExit("build failed: backend onedir output not found")
    shutil.rmtree(OUT_DIR)
    shutil.copytree(built_dir, OUT_DIR)
    print(f"backend folder (onedir) -> {OUT_DIR}")


if __name__ == "__main__":
    main()
