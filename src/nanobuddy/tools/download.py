"""Unified data acquisition from HuggingFace datasets.

Subcommands:
  noise  -- freesound-laion-640k background noise clips
  rir    -- MIT Impulse Response Survey (room impulse responses)
  speech -- LibriSpeech English speech (negative samples with reservoir sampling)
"""

import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset

from .config import SAMPLE_RATE, DEFAULT_SKIP_WORDS


def download_noise(output_dir: str, max_clips: int = 0):
    """Download freesound-laion-640k background noise clips as 16kHz WAV."""
    import io
    from datasets import Audio
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "benjamin-paine/freesound-laion-640k-commercial-16khz-full",
        split="train",
        streaming=True,
    ).cast_column("audio", Audio(decode=False))

    count = 0
    skipped = 0
    for i, example in enumerate(ds):
        fname = f"fs_{i:06d}.wav"
        out_path = out / fname
        if out_path.exists():
            skipped += 1
            count += 1
            if max_clips and count >= max_clips:
                break
            continue
        audio_raw = example["audio"]
        samples, sr = sf.read(io.BytesIO(audio_raw["bytes"]), dtype="float32")
        if sr != SAMPLE_RATE:
            import soxr
            samples = soxr.resample(samples, sr, SAMPLE_RATE)
        samples_int16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
        sf.write(str(out_path), samples_int16, SAMPLE_RATE, subtype="PCM_16")
        count += 1
        if count % 1000 == 0:
            print(f"  {count} clips saved ({skipped} skipped)...", flush=True)
        if max_clips and count >= max_clips:
            break
    print(f"Done: {count} clips ({skipped} already existed) in {out}")


def download_rir(output_dir: str):
    """Download MIT Impulse Response Survey as WAV files (~270 IRs, ~3.7MB)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from datasets import Audio
    ds = load_dataset(
        "benjamin-paine/mit-impulse-response-survey-16khz",
        split="train",
    ).cast_column("audio", Audio(decode=False))

    count = 0
    import io
    for i, example in enumerate(ds):
        fname = f"rir_{i:03d}.wav"
        out_path = out / fname
        if out_path.exists():
            continue
        audio_raw = example["audio"]
        samples, sr = sf.read(io.BytesIO(audio_raw["bytes"]), dtype="float32")
        if sr != SAMPLE_RATE:
            import soxr
            samples = soxr.resample(samples, sr, SAMPLE_RATE)
        samples_int16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
        sf.write(str(out_path), samples_int16, SAMPLE_RATE, subtype="PCM_16")
        count += 1
    print(f"Done: {count} IRs saved to {out} ({len(ds)} total)")


def download_speech(output_dir: str, max_clips: int = 1500,
                    skip_words: set[str] | None = None, seed: int = 42):
    """Download LibriSpeech clips as negative samples via reservoir sampling."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    if skip_words is None:
        skip_words = DEFAULT_SKIP_WORDS

    existing = list(out.glob("cv_*.wav"))
    start_idx = len(existing)
    if start_idx >= max_clips:
        print(f"Already have {start_idx} cv_* files, nothing to do")
        return
    remaining = max_clips - start_idx
    if start_idx > 0:
        print(f"Continuing from {start_idx} existing cv_* files")

    print("Streaming LibriSpeech train-clean-100...")
    from datasets import Audio
    ds = load_dataset(
        "openslr/librispeech_asr", "clean",
        split="train.100", streaming=True,
    ).cast_column("audio", Audio(decode=False))

    reservoir = []
    for i, example in enumerate(ds):
        transcript = example.get("text", "").lower()
        if any(w in transcript for w in skip_words):
            continue
        if len(transcript.strip()) < 3:
            continue
        if len(reservoir) < remaining:
            reservoir.append(example)
        else:
            j = random.randint(0, i)
            if j < remaining:
                reservoir[j] = example
        if (i + 1) % 5000 == 0:
            print(f"  Scanned {i + 1} clips, reservoir full: {len(reservoir) >= remaining}", flush=True)
        if i >= remaining * 20:
            break

    print(f"Selected {len(reservoir)} clips, saving...")
    import io
    idx = start_idx
    for example in reservoir:
        audio_raw = example["audio"]
        samples, sr = sf.read(io.BytesIO(audio_raw["bytes"]), dtype="float32")
        if sr != SAMPLE_RATE:
            import soxr
            samples = soxr.resample(samples, sr, SAMPLE_RATE)
        samples_int16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
        idx += 1
        fname = f"cv_{idx:04d}.wav"
        sf.write(str(out / fname), samples_int16, SAMPLE_RATE, subtype="PCM_16")
        if idx % 100 == 0:
            print(f"  Saved {idx} clips...", flush=True)
    print(f"Done: {len(reservoir)} LibriSpeech clips saved as cv_*.wav in {out}")


def run_download(download_type: str, output_dir: str,
                 max_clips: int = 0, skip_words: str | None = None):
    """CLI entry point for 'nanobuddy download'."""
    if download_type == "noise":
        download_noise(output_dir, max_clips=max_clips)
    elif download_type == "rir":
        download_rir(output_dir)
    elif download_type == "speech":
        skip = set(skip_words.split(",")) if skip_words else None
        download_speech(output_dir, max_clips=max_clips, skip_words=skip)
    else:
        print(f"Unknown download type: {download_type}")
        sys.exit(1)
