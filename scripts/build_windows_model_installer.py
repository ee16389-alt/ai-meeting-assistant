#!/usr/bin/env python3
"""Build a standalone Windows installer for the GGUF model pack."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "desktop" / "model_pack_config.json"
NSI_SCRIPT = ROOT / "packaging" / "windows" / "model_pack_installer.nsi"
SRC_DIR = ROOT / "models" / "llm"
SHERPA_SRC_DIR = ROOT / "models" / "sherpa-onnx"
OUT_DIR = ROOT / "desktop" / "dist-model"


def load_config() -> dict[str, str]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_makensis() -> str | None:
    found = shutil.which("makensis")
    if found:
        return found

    candidates = []
    for env_var in ("ProgramFiles(x86)", "ProgramFiles"):
        base = os.environ.get(env_var)
        if base:
            candidates.append(Path(base) / "NSIS" / "makensis.exe")

    candidates.extend(
        [
            Path(r"C:\Program Files (x86)\NSIS\makensis.exe"),
            Path(r"C:\Program Files\NSIS\makensis.exe"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def main() -> int:
    cfg = load_config()
    model_filename = cfg["ggufFilename"]
    model_version = cfg["versionLabel"]
    installer_filename = cfg["installerFilename"]

    model_file = SRC_DIR / model_filename
    if not model_file.exists():
        raise SystemExit(f"missing GGUF model for installer: {model_file}")
    if not SHERPA_SRC_DIR.exists():
        raise SystemExit(f"missing sherpa-onnx model dir for installer: {SHERPA_SRC_DIR}")
    if not NSI_SCRIPT.exists():
        raise SystemExit(f"missing NSIS script: {NSI_SCRIPT}")

    makensis = resolve_makensis()
    if not makensis:
        raise SystemExit("makensis not found. Please install NSIS and ensure it is on PATH.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / installer_filename

    cmd = [
        makensis,
        f"/DMODEL_FILE={model_file}",
        f"/DMODEL_FILENAME={model_filename}",
        f"/DMODEL_VERSION_LABEL={model_version}",
        f"/DSHERPA_DIR={SHERPA_SRC_DIR}",
        f"/DOUTPUT_EXE={out_file}",
        str(NSI_SCRIPT),
    ]
    print(" ".join(map(str, cmd)))
    subprocess.check_call(cmd, cwd=ROOT)

    if not out_file.exists():
        raise SystemExit(f"model installer build failed: missing output {out_file}")

    print(f"model installer -> {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
