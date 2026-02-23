# Release Checklist (Offline Package)

## Model & Backend
- [ ] `models/llm/qwen2.5-1.5b-instruct-q5_k_m.gguf` exists and is readable
- [ ] `models/sherpa-onnx/` contains required STT assets
- [ ] Backend binary built: `desktop/backend/ai_meeting_backend`
- [ ] `COGNITION_BACKEND=llama_cpp` is used by default
- [ ] App shows a clear error if GGUF file is missing

## Packaging (macOS + Windows)
- [ ] macOS `.dmg` built
- [ ] Windows `.exe` built
- [ ] Package contains backend binary + STT models + GGUF model
- [ ] No Ollama dependency required on user machines

## Clean Machine Verification
- [ ] Install app on a clean machine (no Python, no Ollama)
- [ ] Launch app without using terminal
- [ ] Record audio and get transcript within 10 seconds
- [ ] Generate summary (full/key points/action items)
- [ ] Export transcript and open export folder

## Report Outputs
- [ ] Record artifact paths and sizes for `.dmg` and `.exe`
- [ ] Note any known limitations or risks
