#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found"
  exit 1
fi

if ! command -v pyinstaller >/dev/null 2>&1; then
  echo "pyinstaller not found (install in your build venv first)"
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm not found"
  exit 1
fi

echo "[1/3] Preparing desktop model assets (GGUF + Sherpa-ONNX)..."
python3 scripts/prepare_desktop_assets.py --include-sherpa

echo "[2/3] Building backend binary (PyInstaller onedir, external Sherpa pack)..."
PYINSTALLER_MODE=onedir EMBED_SHERPA_ONNX=0 python3 scripts/build_backend.py

echo "[3/3] Building macOS DMG (bundled GGUF + Sherpa-ONNX)..."
(
  cd desktop
  BUNDLE_GGUF=1 BUNDLE_SHERPA_MODELS=1 npm run build:mac
)

echo "Done. Check desktop/dist/ for the generated .dmg"
