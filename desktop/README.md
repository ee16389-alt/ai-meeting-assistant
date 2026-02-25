# Desktop (macOS .dmg) — Electron

This is a desktop wrapper that starts the Flask backend and opens a desktop window.
By default, Ollama model is not bundled and will be pulled on first run.
By default, GGUF and Ollama model files are not bundled into the installer to keep DMG size smaller.
You can optionally bundle model files into installer (see section below).

## Bundle Models Into Installer (optional)
- `ollama-models` is only used when running desktop app with `SUMMARY_ENGINE=ollama`.
- In default mode (`SUMMARY_ENGINE=llama_cpp`), the app does not need `ollama-models`.
- Sherpa-ONNX is not embedded into backend by default (to reduce backend size). The app expects an external model pack.
- Ollama model can be bundled before packaging:
  1. Ensure target model already exists in local Ollama:
     - `ollama pull qwen2.5:1.5b`
  2. From project root, export model files into `desktop/ollama-models`:
     - `python3 scripts/prepare_ollama_bundle.py --model qwen2.5:1.5b`
  3. Build with bundled Ollama model:
     - `cd desktop && npm run build:mac:bundle-ollama`

- To bundle GGUF into DMG (larger installer):
  1. Prepare assets:
     - `python scripts/prepare_desktop_assets.py`
  2. Build:
     - `cd desktop && npm run build:mac:bundle-gguf`

- To bundle GGUF + Sherpa-ONNX into DMG (recommended full offline package):
  1. Prepare assets:
     - `python scripts/prepare_desktop_assets.py --include-sherpa`
  2. Build:
     - `cd desktop && npm run build:mac:bundle-full`

- To bundle everything (largest installer):
  - `cd desktop && npm run build:mac:bundle-all`

At first launch, app will try to import bundled model files into local Ollama store first.
If bundled files are missing or invalid, it falls back to `ollama pull`.

## Dev
1. `cd desktop`
2. `npm install`
3. `npm run dev`

## Build .dmg (macOS)
Fast path (full offline DMG with bundled GGUF + Sherpa-ONNX):
- `./scripts/build_mac_dmg_full.sh`

1. Build backend binary first:
   - From project root:
     - `python3 -m venv .venv-build`
     - `source .venv-build/bin/activate`
     - `pip install -r requirements.txt`
     - `pip install pyinstaller`
      - `python scripts/build_backend.py`
   - Output:
     - `desktop/backend/ai_meeting_backend` (plus `_internal/` in default `onedir` mode)
2. `cd desktop`
3. `npm run build:mac`

## Build .exe (Windows)
1. Build backend binary first (same as above)
2. `cd desktop`
3. `npm run build:win`

## Notes
- In dev mode, it runs `python3 app.py` from the project root.
- In production, it expects the backend binary in `resources/backend/ai_meeting_backend`.
- First launch auto-installs/pulls Ollama models only when `SUMMARY_ENGINE=ollama`.
- Default backend build is `onedir` + `EMBED_SHERPA_ONNX=0` to reduce installer size. Override with env vars if needed.
- Production desktop launcher defaults to lower startup load (`STT_LAZY_INIT=1`, `SHERPA_ONNX_THREADS=2`, `SHERPA_ONNX_WARMUP_SECONDS=0`, `LLAMA_THREADS=2`) unless env vars are provided.
