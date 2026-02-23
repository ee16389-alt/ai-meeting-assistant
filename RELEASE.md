# Release Guide (Offline DMG/EXE)

This guide builds fully offline installers that bundle:
- Backend binary (ai_meeting_backend)
- STT model (sherpa-onnx)
- LLM model (GGUF)

## Release Metadata
<!-- RELEASE_METADATA_START -->
- Version: 0.1.0
- Build date: 2026-02-23
- Target: macOS arm64 DMG + Windows x64 EXE
<!-- RELEASE_METADATA_END -->

## Artifacts (current build)
<!-- RELEASE_ARTIFACTS_START -->
- DMG: `desktop/dist/AI Meeting Assistant-0.1.0-arm64.dmg` (2.7G)
- EXE: `desktop/dist/AI Meeting Assistant Setup 0.1.0.exe` (1.5G)
<!-- RELEASE_ARTIFACTS_END -->

## 0) Preconditions
- macOS build for DMG (arm64)
- Python venv for build: `.venv-build`
- Node/Electron build tools installed

## 1) Prepare LLM Assets
Copy GGUF models into the desktop packaging folder.

```bash
cd /Users/minashih/ai-gent
python scripts/prepare_desktop_assets.py
```

Expected:
- `desktop/models/llm/*.gguf`

## 2) Build Backend Binary
This bundles templates/static/sherpa-onnx into a single backend executable.

```bash
cd /Users/minashih/ai-gent
source .venv-build/bin/activate
python scripts/build_backend.py
```

Expected:
- `desktop/backend/ai_meeting_backend`

## 3) Build macOS DMG

```bash
cd /Users/minashih/ai-gent/desktop
npm run build:mac
```

Expected:
- `desktop/dist/AI Meeting Assistant-0.1.0-arm64.dmg`

## 4) Install & Verify (Clean Machine)
1. Open the DMG and drag the app into Applications.
2. Verify the model is inside the app bundle:

```bash
ls -lah "/Applications/AI Meeting Assistant.app/Contents/Resources/models/llm"
```

3. Launch the app normally (no terminal).
4. Verify:
- Recording produces transcript in ~10s
- Full summary works
- Key points summary works
- Export works and opens folder

If backend fails to start, run:

```bash
"/Applications/AI Meeting Assistant.app/Contents/MacOS/AI Meeting Assistant"
```

## 5) Windows Build (Notes)
Windows installer should be built on Windows or CI:

```powershell
cd C:\path\to\ai-gent
python scripts\prepare_desktop_assets.py
python scripts\build_backend.py
cd desktop
npm run build:win
```

Expected:
- `desktop\dist\AI Meeting Assistant Setup 0.1.0.exe`

## Troubleshooting
- Missing model inside app bundle: re-run `scripts/prepare_desktop_assets.py` then rebuild DMG.
- Backend startup error: ensure `scripts/build_backend.py` includes hidden imports for
  `engineio.async_drivers.threading` and `socketio.async_drivers.threading`.

## Risks / Limitations
- Large installer sizes due to bundled GGUF and STT models (multi-GB).
- Unsigned DMG may trigger macOS Gatekeeper warnings on other machines.
- Windows build should be produced on Windows or CI to avoid cross-platform packaging issues.
