# AI Meeting Assistant

A fully offline desktop meeting assistant that records audio, transcribes speech in real time using Sherpa-ONNX, and generates AI summaries using a local GGUF language model. No cloud services, no internet required after initial model download.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Hardware Requirements & Limitations](#3-hardware-requirements--limitations)
4. [Installation & Setup](#4-installation--setup)
5. [Features & Usage](#5-features--usage)
6. [Configuration & Tuning Guide](#6-configuration--tuning-guide)
7. [Known Issues & Limitations](#7-known-issues--limitations)
8. [FAQ](#8-faq)
9. [Windows Build Guide](#9-windows-build-guide)

---

## 1. Project Overview

AI Meeting Assistant captures microphone audio, transcribes it in real time using a streaming Zipformer model (Sherpa-ONNX), and generates a concise full-text summary using Phi-4-mini-instruct running locally via llama-cpp-python. Everything runs on the local machine — no API keys, no network calls during a session.

**Key features:**

- Real-time streaming speech-to-text with timestamp tracking
- Local LLM summarization (Phi-4-mini Q4_K_M)
- Full-text editor for transcript correction
- Transcript keyword search with result navigation
- Recording duration timer
- Auto-save draft to disk every 30 seconds
- Export transcript and summary as separate files
- Electron desktop shell with offline model download on first launch
- Works entirely offline once models are downloaded

---

## 2. System Architecture

### File Responsibilities

| File | Responsibility |
|------|---------------|
| `app.py` | Flask + Flask-SocketIO server. Handles all SocketIO events, manages recording session state, orchestrates STT callbacks, LLM summarization, and file export. |
| `stt_engine.py` | Sherpa-ONNX streaming STT engine. Wraps `OnlineRecognizer`, manages audio queue, worker thread, endpoint detection, and `_flush_stream` on stop. |
| `cognition.py` | LLM layer. Loads GGUF via llama-cpp-python, provides `summarize_full()` streaming generator. Falls back to Ollama if local GGUF is unavailable. Implements map-reduce chunking for transcripts > 3,000 characters. |
| `desktop/main.js` | Electron main process. Launches the Python backend as a child process, manages model download on first launch, exposes draft IPC handlers (`draft:save`, `draft:load`, `draft:clear`). |
| `desktop/preload.js` | Electron preload script. Exposes `window.electronAPI.draft` to the renderer via `contextBridge`. |
| `templates/index.html` | Single-page frontend. All UI logic in vanilla JS with Socket.IO client, Tailwind CSS, and Lucide icons. |
| `scripts/build_backend.py` | PyInstaller packaging script for the Python backend. |
| `scripts/prepare_desktop_assets.py` | Copies bundled models into the Electron `desktop/` directory before building. |

### Data Flow

```
Microphone (browser getUserMedia)
    │  Float32 PCM → downsampled to 16 kHz → Int16 PCM
    │
    ▼
AudioWorklet (pcm-capture-processor, 4096-sample batches)
    │  socket.emit('audio_chunk')
    │
    ▼
app.py  handle_audio_chunk
    │  bytes → numpy float32
    │
    ▼
stt_engine.py  feed_audio()
    │  accumulate in _sample_buffer
    │  process 480-sample (30 ms) windows
    │  Sherpa-ONNX OnlineRecognizer.decode_stream()
    │
    ▼  endpoint detected / stale timeout / flush on stop
    │
    ▼
_on_stt_segments callback → app.py
    │  append to transcript_lines[]
    │  socket.emit('transcript_update', {index, text, timestamp})
    │
    ▼
Frontend transcript panel (real-time display)
    │
    └──► User clicks "Generate Summary"
             │  socket.emit('request_summary')
             │
             ▼
         app.py  _run_summary_task()
             │  if transcript > 3,000 chars → map-reduce chunking
             │  cognition.summarize_full(text)
             │  stream tokens via socket.emit('summary_token')
             │
             ▼
         Frontend summary panel (streaming display)
```

### SocketIO Event Reference

#### Client → Server

| Event | Payload | Description |
|-------|---------|-------------|
| `start_recording` | `{meeting_name, save_audio, language_mode}` | Begin a new recording session |
| `audio_chunk` | `{chunk: ArrayBuffer}` | Raw PCM audio data (16 kHz, Int16) |
| `audio_record_chunk` | `{chunk: ArrayBuffer}` | WebM audio for file saving |
| `audio_recording_done` | — | Signal end of audio file stream |
| `pause_recording` | — | Pause STT engine |
| `resume_recording` | — | Resume STT engine |
| `stop_recording` | — | Stop recording and trigger flush |
| `request_summary` | `{mode, transcript_override}` | Request LLM summary |
| `cancel_summary` | — | Cancel in-progress summary |
| `export_meeting` | `{meeting_name, transcript_override, summary_full}` | Export transcript + summary |
| `export_summary` | `{mode, meeting_name, ...}` | Export summary only |
| `get_storage_path` | `{meeting_name}` | Query export directory path |
| `open_storage_folder` | `{meeting_name}` | Open export folder in OS file manager |

#### Server → Client

| Event | Payload | Description |
|-------|---------|-------------|
| `state_changed` | `{state, ollama?, stt_error?}` | STT engine state update |
| `transcript_update` | `{index, text, timestamp, token}` | New confirmed transcript line |
| `transcript_partial` | `{text}` | In-progress partial recognition result |
| `transcript_partial_clear` | — | Clear partial line display |
| `transcription_ready` | — | Flush complete, all lines delivered |
| `summary_start` | `{mode}` | Summary generation started |
| `summary_token` | `{mode, token}` | Streaming summary token |
| `summary_progress` | `{current, total, stage}` | Map-reduce chunk progress |
| `summary_done` | `{mode, content}` | Summary complete |
| `summary_error` | `{mode, message}` | Summary failed or cancelled |
| `keep_alive` | `{mode}` | Heartbeat during long inference |
| `export_ready` | `{files}` | Export completed, file list |
| `export_error` | `{message}` | Export failed |

---

## 3. Hardware Requirements & Limitations

### Minimum Recommended Specs

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Intel Core i5 (8th gen+) or AMD Ryzen 5 3000+ | i7 / Ryzen 7 with AVX2 |
| RAM | 8 GB | 16 GB |
| Storage | 5 GB free (models + app) | 10 GB |
| OS | Windows 10 64-bit | Windows 11 |
| GPU | Not required | Not required |

### Intel i7-1255U + 16 GB RAM (No Discrete GPU)

This is the primary tested configuration. Key characteristics:

- **STT (Sherpa-ONNX Zipformer zh-int8)**: Runs comfortably in real time. `num_threads=4` is the tuned value — higher may compete with the LLM warmup thread.
- **LLM (Phi-4-mini Q4_K_M, n_ctx=4096)**: Pure CPU inference. Expect ~20–60 seconds per summary for a 30–60 minute meeting.
- **LLM warmup delay**: A 60-second delay is intentionally inserted after app launch before LLM warms up, to avoid competing with STT model loading.
- **No GPU acceleration**: `n_gpu_layers=0` is enforced. GPU offloading is not supported in the current build.
- **Concurrent STT + LLM**: Running both simultaneously is not recommended. The app serializes LLM calls with a lock.

### Summary Time Estimates (i7-1255U)

| Transcript Length | Mode | Estimated Time |
|-------------------|------|----------------|
| < 500 chars | Single-pass | 10–20 s |
| 500–3,000 chars | Single-pass | 20–60 s |
| 3,000–6,000 chars | Map-reduce (2 chunks) | 60–120 s |
| > 6,000 chars | Map-reduce (3+ chunks) | 2–5 min |

---

## 4. Installation & Setup

### Prerequisites

- Python 3.11 (3.10–3.12 should also work)
- Node.js 20 LTS
- Windows 10/11 x64 (for production use; macOS/Linux for development)

### Python Environment

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

# Install all dependencies except llama-cpp-python
pip install flask flask-socketio sherpa-onnx requests numpy opencc-python-reimplemented
```

### llama-cpp-python — Windows AVX2 Installation

**Important:** The default `pip install llama-cpp-python` wheel on Windows uses AVX-512 instructions. On CPUs that do not support AVX-512 (most consumer laptops including i7-1255U), this causes a `WinError 0xc000001d` (`STATUS_ILLEGAL_INSTRUCTION`) crash at runtime.

Always build from source with AVX2 enabled and AVX-512 disabled:

```powershell
$env:CMAKE_ARGS = "-DLLAMA_AVX=ON -DLLAMA_AVX2=ON -DLLAMA_F16C=ON -DLLAMA_FMA=ON -DLLAMA_AVX512=OFF"
pip install llama-cpp-python --force-reinstall --no-cache-dir --no-binary llama-cpp-python
```

This requires CMake and a C++ build toolchain (Visual Studio Build Tools or MSVC). The GitHub Actions workflow handles this automatically.

### STT Model — Sherpa-ONNX Zipformer

The default model is `sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30` (Chinese, int8 quantized).

**Download:**
```
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30.tar.bz2
```

Place the extracted directory at:
```
models/sherpa-onnx/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30/
```

Required files: `encoder-*.onnx`, `decoder-*.onnx`, `joiner-*.onnx`, `tokens.txt`

To use a different model, update `desktop/model_pack_config.json`:
```json
{
  "sherpaModelDirName": "your-model-dir-name"
}
```

The engine auto-detects model type: presence of `joiner*.onnx` → Transducer (Zipformer); otherwise → Paraformer.

### LLM Model — Phi-4-mini GGUF

The default model is `Phi-4-mini-instruct-Q4_K_M.gguf` (~2.5 GB).

**Download:**
```
https://huggingface.co/lmstudio-community/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q4_K_M.gguf
```

Place at:
```
models/llm/Phi-4-mini-instruct-Q4_K_M.gguf
```

To override the model path at runtime:
```
AMA_GGUF_PATH=C:\path\to\model.gguf
```

### Electron Desktop Shell

```bash
cd desktop
npm install
npm start        # Development mode (connects to running backend)
```

To start the full stack in development:
```bash
# Terminal 1 — Python backend
python app.py

# Terminal 2 — Electron shell
cd desktop && npm start
```

---

## 5. Features & Usage

### Recording & Live Transcript

1. Enter a meeting name and click **Start Meeting**
2. A 3-second countdown runs while the microphone initializes in the background
3. Speech is transcribed in real time and displayed in the left panel with elapsed recording time (HH:MM:SS) per line
4. Click **Pause** to suspend audio capture; click **Resume** to continue
5. Click **Stop** to end the session — the system flushes the last audio segment before marking the transcript complete

### Recording Timer

A `MM:SS` counter appears in the header during recording. It resets when a new session starts and stops when recording ends.

### Full-Text Summary

After recording stops and the transcript is ready:

1. Click **Generate Summary** (top-right of the summary panel)
2. The local LLM processes the transcript and streams the result token by token
3. For transcripts longer than 3,000 characters, a map-reduce pipeline compresses the text in 2,000-character chunks before final summarization — a progress bar shows chunk completion
4. Click **Regenerate** to re-run summarization on the current transcript

### Edit Full Text

Click **Edit Full Text** (top-right of the transcript panel) to open an inline editor. Format is `HH:MM:SS text` per line. Save to update the transcript used for summarization and export.

### Transcript Search

Click the **Search** button in the transcript panel header:

- Type a keyword to highlight all matches in yellow
- Use the **↑ / ↓** arrow buttons to navigate between matches (current match highlighted in orange)
- A counter shows `N / M total` results
- Clear the input or click **×** to restore normal view

### Word Count Hint

Before generating a summary, the summary panel shows the current transcript character count (e.g., `逐字稿共 1,234 字`). This helps estimate how long summarization will take.

### Auto-Save Draft

The app saves a draft to disk every 30 seconds via Electron IPC:

- **Location**: `%APPDATA%\AI Meeting Assistant\draft.json`
- **Contents**: transcript lines, timestamps, summary, meeting name
- On next launch, the app checks for a draft and prompts: *"Found unsaved meeting notes from last session. Restore?"*
- The draft is automatically deleted when you export or start a new recording

### Export

- **Export Transcript**: saves `transcript.txt` with timestamps to the meeting folder
- **Export Summary**: saves `summary_full.txt` to the meeting folder
- **Open Folder**: opens the export directory in Windows Explorer

Default export location:
```
C:\Users\<name>\Documents\AI Meeting Assistant\download\<meeting-name>\
```

Override with environment variable:
```
AMA_EXPORT_ROOT=D:\my-meetings
```

### Language Switching

Toggle between **中文 (Chinese)** and **English** in the header dropdown. All UI labels, toasts, and guide text update immediately.

---

## 6. Configuration & Tuning Guide

### LLM Parameters

Configured via environment variables or hardcoded defaults in `cognition.py`:

| Parameter | Env Var | Default | Notes |
|-----------|---------|---------|-------|
| Context window | `AMA_LLM_CTX` | `4096` | **Do not exceed 4096** without testing. Larger values increase RAM usage significantly and may cause OOM on 8 GB systems. |
| Inference threads | `AMA_LLM_THREADS` | `2` | Increase to 4 only if STT is not running concurrently. Higher values do not linearly improve speed on efficiency-core CPUs. |
| Batch threads | `AMA_LLM_THREADS_BATCH` | `8` | Controls prompt processing parallelism. Reduce to 4 if the system becomes unresponsive during summarization. |
| Max output tokens | `AMA_LLM_MAX_TOKENS` | `512` | Increase for longer summaries, but each additional token extends inference time. |
| GPU layers | — | `0` | GPU offloading is disabled. Do not enable without verifying CUDA/Metal toolchain compatibility. |

**Before adjusting threads**, verify your CPU's instruction set:
```powershell
# Check AVX2 support
(Get-WmiObject Win32_Processor).Description
```

### STT Parameters

Configured in `stt_engine.py`:

| Parameter | Default | Adjustment Notes |
|-----------|---------|-----------------|
| `SILENCE_THRESHOLD` | `0.003` (RMS) | Lower to transcribe quieter voices; raise if background noise causes spurious output. Practical range: `0.001`–`0.01`. |
| `CHUNK_SAMPLES` | `480` (30 ms) | **Do not set below 480**. This matches the Zipformer encoder's minimum chunk size. Smaller values cause decoder errors. |
| `num_threads` | `4` | Matched to the i7-1255U P-core count. Reduce to `2` on older dual-core machines. |
| `_TEXT_STALE_TIMEOUT` | `1.5 s` | Time without text change before forcing a segment flush. Lower for faster sentence commits; raise if words get cut mid-sentence. |
| `rule1_min_trailing_silence` | `2.4 s` | Long silence triggers endpoint. Increase for speakers who pause frequently. |
| `rule2_min_trailing_silence` | `0.8 s` | Short silence within active speech. |
| `rule3_min_utterance_length` | `10 s` | Force segment after 10 s even without endpoint. |

### Prompt Adjustments

Prompts are in `cognition.py`. When editing:

- Keep the instruction concise — every token in the system prompt consumes context budget
- Do not remove `COMMON_OUTPUT_GUARDRAILS`; they suppress hallucinated role labels and prompt echoing
- Test with short (< 500 char) transcripts first before deploying changes to production

### SocketIO Parameters

In `app.py`:

```python
ping_interval=30   # Heartbeat every 30 s
ping_timeout=120   # Disconnect after 120 s without pong
```

Do not reduce `ping_timeout` below 60 seconds. LLM inference on slow hardware can block the event loop for 30–90 seconds, causing false disconnections.

---

## 7. Known Issues & Limitations

### LLM Inference Cannot Be Interrupted Mid-Token

`llama-cpp-python`'s token streaming does not support cancellation at the native level. The cancel mechanism sets a flag that is checked between tokens, so the model finishes the current token before stopping. On slow hardware, this may mean waiting 1–2 seconds after clicking Cancel.

### Occasional Simplified Chinese Characters

The Zipformer model is trained primarily on Mandarin and may output simplified Chinese characters. `opencc` (s2twp conversion) is applied to each segment, but some rare characters or technical terms may slip through.

### Long Meeting Summarization Time

Meetings longer than 30 minutes (> 6,000 characters of transcript) trigger map-reduce chunking. Each 2,000-character chunk requires a separate LLM call (~20–40 s per chunk on i7-1255U). A 60-minute meeting may take 3–5 minutes to fully summarize.

### Windows AVX-512 Compatibility

Pre-built `llama-cpp-python` wheels from PyPI may include AVX-512 instructions. These cause `WinError 0xc000001d` on CPUs without AVX-512 (i7-1255U, most consumer laptops). Always build from source with `CMAKE_ARGS` as documented in Section 4.

### Chinese-Only STT Model

The default `sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30` model is Chinese-only. English words spoken in a Chinese meeting may be misrecognized or omitted. For bilingual meetings, switch to the bilingual Zipformer model (`sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`).

### No Multi-Speaker Diarization

The current pipeline does not perform speaker diarization. All transcript lines are attributed to a single speaker.

---

## 8. FAQ

### Q: I get `WinError 0xc000001d` when the app starts

This means `llama-cpp-python` was compiled with AVX-512 instructions that your CPU does not support. Reinstall from source:

```powershell
$env:CMAKE_ARGS = "-DLLAMA_AVX=ON -DLLAMA_AVX2=ON -DLLAMA_F16C=ON -DLLAMA_FMA=ON -DLLAMA_AVX512=OFF"
pip install llama-cpp-python --force-reinstall --no-cache-dir --no-binary llama-cpp-python
```

### Q: The summary appears stuck / the progress bar is not moving

The LLM is processing on CPU — this is expected. Check:

1. The `keep_alive` heartbeat should fire every ~5 seconds in the browser console
2. If the progress bar is on a specific chunk number and does not advance after 60 seconds, the map-reduce chunk has timed out — the app will skip it and continue
3. If the spinner has been running for more than 5 minutes, click **Cancel**, then try again with a shorter transcript or use **Edit Full Text** to trim it

### Q: STT accuracy is poor — many missed or wrong words

1. **Check microphone gain**: the RMS diagnostic strip at the bottom of the screen shows live audio levels. If consistently below `0.003`, lower `SILENCE_THRESHOLD`
2. **Reduce background noise**: the Zipformer model is sensitive to reverb and fan noise
3. **Switch to bilingual model** if the meeting mixes Chinese and English
4. **Verify model integrity**: re-download the Sherpa model archive if you suspect corruption

### Q: How do I switch the STT model?

1. Download the new model archive from the [Sherpa-ONNX releases page](https://github.com/k2-fsa/sherpa-onnx/releases)
2. Extract to `models/sherpa-onnx/<model-dir-name>/`
3. Update `desktop/model_pack_config.json`:
   ```json
   { "sherpaModelDirName": "your-new-model-dir-name" }
   ```
4. Restart the backend

The engine auto-detects Zipformer vs Paraformer by checking for `joiner*.onnx`.

---

## 9. Windows Build Guide

### GitHub Actions (Recommended)

The repository includes `.github/workflows/windows-build-light.yml` which builds a Windows NSIS installer without bundled models (models are downloaded on first launch).

**To trigger a build:**

1. Go to **Actions → Build Windows Installer (Light - First Launch Download)**
2. Click **Run workflow**
3. Select the branch to build from
4. Click **Run workflow**

The artifact `windows-exe-light` (containing the `.exe` installer) is available for download once the run completes (~20–30 minutes).

For a build with bundled models, use `windows-build-bundled-models.yml` — this requires providing direct download URLs for the GGUF and Sherpa archives as workflow inputs.

### Local Windows Build

Requires: Windows 10/11, Python 3.11, Node.js 20, NSIS, Visual Studio Build Tools 2022.

```powershell
# 1. Install Python deps (see Section 4 for llama-cpp-python)
python -m pip install -r requirements.txt
python -m pip install pyinstaller

# 2. Build backend executable
python scripts\build_backend.py

# 3. Install Electron deps
cd desktop
npm ci

# 4. Download VC++ redistributable (optional, for installer)
New-Item -ItemType Directory -Force -Path prereqs
Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile "prereqs\vc_redist.x64.exe"

# 5. Build installer
npm run build:win

# Output: desktop\dist\AI-Meeting-Assistant-*.exe
```

### Important Notes

- **Windows only**: Electron Builder's NSIS target only runs on Windows. Do not attempt to cross-compile the installer from macOS or Linux.
- **Architecture**: Build on the same architecture as the target (x64). ARM64 Windows is not currently supported.
- **NSIS**: Install via `choco install nsis` or from [nsis.sourceforge.io](https://nsis.sourceforge.io).
- **Model download**: The light installer does not bundle models. On first launch, the Electron shell downloads models automatically to `%APPDATA%\AI Meeting Assistant\models\`.
- **Signing**: Code signing is disabled (`CSC_IDENTITY_AUTO_DISCOVERY=false`) in CI. Windows Defender may flag unsigned executables — users can bypass by clicking "More info → Run anyway" in the SmartScreen dialog.
