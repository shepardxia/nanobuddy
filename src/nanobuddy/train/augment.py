"""Audio augmentation pipeline for training data."""

from __future__ import annotations

import logging
import random
from typing import Generator

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


def augment_clips(
    clip_paths: list[str],
    total_length: int,
    *,
    sr: int = 16000,
    batch_size: int = 128,
    background_paths: list[str] | None = None,
    rir_paths: list[str] | None = None,
    settings: dict | None = None,
) -> Generator[np.ndarray, None, None]:
    """Batch-wise audio augmentation generator.

    Loads clips, resamples to ``sr``, enforces ``total_length`` via crop/pad,
    applies gain → background noise → RIR → pitch shift → colored noise.

    Yields int16 arrays of shape ``(batch_size, total_length)``.
    """
    from torch_audiomentations import (
        AddBackgroundNoise,
        AddColoredNoise,
        ApplyImpulseResponse,
        Compose,
        Gain,
        PitchShift,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = {
        "min_snr_db": 3.0, "max_snr_db": 30.0,
        "rir_prob": 0.2,
        "pitch_prob": 0.3, "min_pitch_semitones": -2.0, "max_pitch_semitones": 2.0,
        "gain_prob": 1.0, "min_gain_db": -6.0, "max_gain_db": 6.0,
        "colored_noise_prob": 0.3,
    }
    if settings:
        cfg.update(settings)

    transforms = [
        Gain(min_gain_in_db=cfg["min_gain_db"], max_gain_in_db=cfg["max_gain_db"],
             p=cfg["gain_prob"], sample_rate=sr),
    ]
    if background_paths:
        transforms.append(AddBackgroundNoise(
            background_paths=background_paths,
            min_snr_in_db=cfg["min_snr_db"], max_snr_in_db=cfg["max_snr_db"],
            p=0.8, sample_rate=sr,
        ))
    if rir_paths:
        transforms.append(ApplyImpulseResponse(
            ir_paths=rir_paths, p=cfg["rir_prob"], sample_rate=sr,
        ))
    transforms.append(PitchShift(
        min_transpose_semitones=cfg["min_pitch_semitones"],
        max_transpose_semitones=cfg["max_pitch_semitones"],
        p=cfg["pitch_prob"], sample_rate=sr,
    ))
    transforms.append(AddColoredNoise(
        min_snr_in_db=20.0, max_snr_in_db=40.0,
        p=cfg["colored_noise_prob"], sample_rate=sr,
    ))

    augmenter = Compose(transforms=transforms, output_type="dict")
    augmenter.to(device)

    random.shuffle(clip_paths)

    for i in range(0, len(clip_paths), batch_size):
        batch = []
        for path in clip_paths[i : i + batch_size]:
            try:
                waveform, clip_sr = torchaudio.load(path)
                if clip_sr != sr:
                    waveform = torchaudio.transforms.Resample(clip_sr, sr)(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                n = waveform.shape[1]
                if n > total_length:
                    start = random.randint(0, n - total_length)
                    waveform = waveform[:, start : start + total_length]
                elif n < total_length:
                    pad = total_length - n
                    lpad = random.randint(0, pad)
                    waveform = torch.nn.functional.pad(waveform, (lpad, pad - lpad))
                batch.append(waveform)
            except Exception as e:
                logger.warning("Skipping %s: %s", path, e)

        if not batch:
            continue

        tensor = torch.stack(batch).to(device)
        with torch.no_grad():
            out = augmenter(samples=tensor, sample_rate=sr)["samples"]

        arr = out.cpu().numpy()
        peak = np.abs(arr).max(axis=-1, keepdims=True)
        peak[peak < 1e-8] = 1.0
        yield ((arr / peak).squeeze(1) * 32767).astype(np.int16)
