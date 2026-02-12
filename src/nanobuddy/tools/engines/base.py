"""Shared audio utilities matching nanobuddy's output format.

Output spec: 16kHz, mono, 16-bit signed integer WAV.
Post-processing: median filter (kernel=3) + 7kHz low-pass Butterworth (4th order).
"""

import wave
import os
import numpy as np
import scipy.signal as sps

from ..config import SAMPLE_RATE

TARGET_SAMPLE_RATE = SAMPLE_RATE  # 16000


def resample(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample audio to TARGET_SAMPLE_RATE if needed."""
    if orig_sr == TARGET_SAMPLE_RATE:
        return audio
    num_samples = int(len(audio) * TARGET_SAMPLE_RATE / orig_sr)
    return sps.resample(audio, num_samples)


def post_process(audio: np.ndarray) -> np.ndarray:
    """Apply the same post-processing as nanobuddy's generate_samples.

    1. Normalize to int16 range
    2. Median filter (kernel=3)
    3. Low-pass Butterworth (4th order, 7kHz cutoff)
    4. Clip to int16 range
    """
    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 32767:
        audio = audio / max_val * 32767

    audio_float = audio.astype(np.float32)

    # Median filter
    try:
        audio_float = sps.medfilt(audio_float, kernel_size=3)
    except Exception:
        pass

    # Low-pass Butterworth
    try:
        sos = sps.butter(4, 7000, 'low', fs=TARGET_SAMPLE_RATE, output='sos')
        audio_float = sps.sosfilt(sos, audio_float)
    except Exception:
        pass

    return np.clip(audio_float, -32767, 32767).astype(np.int16)


def save_wav(audio_int16: np.ndarray, path: str):
    """Write mono 16kHz 16-bit WAV."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())


def make_filename(file_prefix: str, engine_tag: str, voice_tag: str,
                  index: int) -> str:
    """Generate unique filename matching nanobuddy's pattern."""
    import time
    import random
    timestamp_ms = int(time.time() * 1000)
    rand = random.randint(100, 999)
    return f"{file_prefix}_{timestamp_ms}_{rand}_{engine_tag}_{voice_tag}_{index}.wav"
