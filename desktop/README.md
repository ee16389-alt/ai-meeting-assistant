# Desktop (macOS .dmg) — Electron

This is a desktop wrapper that starts the Flask backend and opens a desktop window.
Models are not bundled. The app will prompt to install Ollama and then pull models on first run.

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

## Notes
- In dev mode, it runs `python3 app.py` from the project root.
- In production, it expects the backend binary in `resources/backend/ai_meeting_backend`.
- First launch will auto-install Ollama (with admin prompt) and pull models.
