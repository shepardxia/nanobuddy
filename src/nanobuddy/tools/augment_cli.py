"""Augment wake word audio clips via audiomentations.

Applies pitch shift, 7-band parametric EQ, and gain transforms.
No heavy deps -- just audiomentations (numpy-based, fast).
"""

import wave
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import SAMPLE_RATE


def _read_wav(path: Path) -> np.ndarray:
    """Read 16kHz mono WAV as float32 [-1, 1]."""
    with wave.open(str(path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def _write_wav(audio: np.ndarray, path: Path):
    """Write float32 audio as 16kHz mono int16 WAV."""
    audio_int16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())


def augment_clips(
    input_dir: Path,
    output_dir: Path,
    repeats: int = 5,
    pitch_shift_prob: float = 0.3,
    eq_prob: float = 0.2,
    gain_prob: float = 0.4,
    noise_prob: float = 0.3,
    time_stretch_prob: float = 0.4,
):
    """Augment all WAV files in input_dir, writing to output_dir.

    Transforms: pitch shift (+/-3 semitones), 7-band parametric EQ (+/-6dB),
    gain (+/-8dB), gaussian noise (SNR -10 to 20dB).
    Output: same sample rate / format as input (16kHz mono int16).
    """
    from audiomentations import (
        AddGaussianSNR, Compose, Gain, PitchShift, SevenBandParametricEQ,
        TimeStretch,
    )

    wavs = sorted(input_dir.glob("*.wav"))
    if not wavs:
        print(f"No .wav files found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    transform = Compose([
        PitchShift(min_semitones=-3, max_semitones=3, p=pitch_shift_prob),
        SevenBandParametricEQ(
            min_gain_db=-6.0, max_gain_db=6.0, p=eq_prob,
        ),
        Gain(min_gain_db=-8.0, max_gain_db=8.0, p=gain_prob),
        AddGaussianSNR(min_snr_db=-10.0, max_snr_db=20.0, p=noise_prob),
        TimeStretch(min_rate=0.8, max_rate=1.2, leave_length_unchanged=False,
                    p=time_stretch_prob),
    ])

    total = len(wavs) * repeats
    print(f"Input: {len(wavs)} clips from {input_dir}")
    print(f"Repeats: {repeats} -> {total} augmented clips")
    print(f"Output: {output_dir}")
    print(f"Transforms: pitch shift (p={pitch_shift_prob}), "
          f"7-band EQ (p={eq_prob}), gain (p={gain_prob}), "
          f"noise (p={noise_prob}), time stretch (p={time_stretch_prob})")

    saved = 0
    for wav_path in tqdm(wavs, desc="Augmenting", unit="src"):
        audio = _read_wav(wav_path)
        stem = wav_path.stem

        for r in range(repeats):
            augmented = transform(samples=audio, sample_rate=SAMPLE_RATE)
            out_path = output_dir / f"{stem}_aug{r:02d}.wav"
            _write_wav(augmented, out_path)
            saved += 1

    print(f"\nDone. {saved} augmented clips saved to {output_dir}")


def run_augment(input_dir: str, output_dir: str,
                repeats: int = 5, batch_size: int = 32):
    """CLI entry point for augment subcommand."""
    augment_clips(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        repeats=repeats,
    )
