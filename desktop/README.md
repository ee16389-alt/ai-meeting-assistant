# Desktop (macOS .dmg) — Electron

This is a desktop wrapper that starts the Flask backend and opens a desktop window.
By default, Ollama model is not bundled and will be pulled on first run.
You can optionally bundle Ollama model files into installer (see section below).

## Bundle Models Into Installer (optional)
- STT model (Sherpa) is bundled into backend binary by `scripts/build_backend.py`.
- Ollama model can be bundled before packaging:
  1. Ensure target model already exists in local Ollama:
     - `ollama pull qwen2.5:1.5b`
  2. From project root, export model files into `desktop/ollama-models`:
     - `python3 scripts/prepare_ollama_bundle.py --model qwen2.5:1.5b`
  3. Then build desktop installer as usual.

At first launch, app will try to import bundled model files into local Ollama store first.
If bundled files are missing or invalid, it falls back to `ollama pull`.

## Dev
1. `cd desktop`
2. `npm install`
3. `npm run dev`

## Build .dmg (macOS)
1. Build backend binary first:
   - From project root:
     - `python3 -m venv .venv-build`
     - `source .venv-build/bin/activate`
     - `pip install -r requirements.txt`
     - `pip install pyinstaller`
     - `python scripts/build_backend.py`
   - Output:
     - `desktop/backend/ai_meeting_backend`
2. `cd desktop`
3. `npm run build:mac`

## Build .exe (Windows)
1. Build backend binary first (same as above)
2. `cd desktop`
3. `npm run build:win`

## Notes
- In dev mode, it runs `python3 app.py` from the project root.
- In production, it expects the backend binary in `resources/backend/ai_meeting_backend`.
- First launch will auto-install Ollama (with admin prompt) and pull models.
