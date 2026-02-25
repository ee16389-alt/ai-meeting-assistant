#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${ROOT_DIR}/static/vendor"
mkdir -p "${VENDOR_DIR}"

echo "Downloading vendor assets to ${VENDOR_DIR}"

curl -L -o "${VENDOR_DIR}/tailwind.js" "https://cdn.tailwindcss.com"
curl -L -o "${VENDOR_DIR}/lucide.min.js" "https://unpkg.com/lucide@0.469.0/dist/umd/lucide.min.js"
curl -L -o "${VENDOR_DIR}/socket.io.min.js" "https://cdn.socket.io/4.7.5/socket.io.min.js"

echo "Done."
