#!/usr/bin/env python3
"""Simple A/B benchmark for STT models (faster-whisper / whisper.cpp).

Usage:
  python scripts/benchmark_stt.py /path/to/audio.wav
  python scripts/benchmark_stt.py /path/to/audio.wav --engine whisper.cpp --models small,medium,large
"""

import argparse
import os
import time
import tempfile
import subprocess
from typing import List

import numpy as np
from pydub import AudioSegment
from faster_whisper import WhisperModel

DEFAULT_WHISPER_CPP_BIN = "/Users/minashih/tools/whisper.cpp/bin/whisper-cli"
DEFAULT_WHISPER_CPP_MODEL_DIR = "/Users/minashih/models/whisper.cpp"


def load_audio_16k_mono(path: str) -> np.ndarray:
    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    samples = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


def run_faster_whisper(model_size: str, audio: np.ndarray) -> str:
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio, vad_filter=True, beam_size=5)
    texts = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            texts.append(text)
    return " ".join(texts)


def run_whisper_cpp(model_size: str, audio: np.ndarray) -> str:
    bin_path = os.getenv("WHISPER_CPP_BIN", "").strip() or DEFAULT_WHISPER_CPP_BIN
    model_dir = os.getenv("WHISPER_CPP_MODEL_DIR", "").strip() or DEFAULT_WHISPER_CPP_MODEL_DIR
    model_path = os.path.join(model_dir, f"ggml-{model_size}.bin")
    if not os.path.isfile(bin_path):
        raise RuntimeError(f"whisper.cpp binary not found: {bin_path}")
    if not os.path.isfile(model_path):
        raise RuntimeError(f"whisper.cpp model not found: {model_path}")

    with tempfile.TemporaryDirectory(prefix="whispercpp_bench_") as td:
        wav_path = os.path.join(td, "audio.wav")
        out_prefix = os.path.join(td, "out")
        # write wav
        audio_i16 = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)
        wav = AudioSegment(
            audio_i16.tobytes(),
            frame_rate=16000,
            sample_width=2,
            channels=1,
        )
        wav.export(wav_path, format="wav")
        cmd = [
            bin_path,
            "-m", model_path,
            "-f", wav_path,
            "-otxt",
            "-of", out_prefix,
            "-l", "auto",
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        txt_path = f"{out_prefix}.txt"
        if not os.path.isfile(txt_path):
            return ""
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()


def bench(engine: str, models: List[str], audio: np.ndarray) -> None:
    for model in models:
        start = time.time()
        if engine == "whisper.cpp":
            text = run_whisper_cpp(model, audio)
        else:
            text = run_faster_whisper(model, audio)
        elapsed = time.time() - start
        print("=" * 60)
        print(f"engine: {engine}")
        print(f"model : {model}")
        print(f"time  : {elapsed:.2f}s")
        print(f"chars : {len(text)}")
        print("preview:")
        print(text[:280].replace("\n", " "))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--engine", default="faster-whisper", choices=["faster-whisper", "whisper.cpp"])
    parser.add_argument("--models", default="small,medium,large")
    args = parser.parse_args()

    if not os.path.isfile(args.audio_path):
        raise SystemExit(f"Audio file not found: {args.audio_path}")

    audio = load_audio_16k_mono(args.audio_path)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    bench(args.engine, models, audio)


if __name__ == "__main__":
    main()
