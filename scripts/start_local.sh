#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$ROOT_DIR/.venv310/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv310/bin/activate"
fi

export COGNITION_BACKEND="llama_cpp"
export LLM_MODEL_PATH="$ROOT_DIR/models/llm/qwen2.5-1.5b-instruct-q5_k_m.gguf"

python "$ROOT_DIR/app.py"
