"""Auto-generate training hyperparameters from dataset statistics."""

from __future__ import annotations

import math

import numpy as np


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def auto_config(stats: dict) -> dict:
    """Generate training config values from dataset statistics.

    ``stats`` keys: ``H_pos`` (hours positive), ``H_neg`` (hours negative),
    ``H_noise`` (hours noise), ``A_noise`` (noise amplitude 0..1), ``N_rir`` (RIR count).

    Returns a flat dict of generated hyperparameters.
    """
    h_pos = stats.get("H_pos", 0.0)
    h_neg = stats.get("H_neg", 0.0)
    h_noise = stats.get("H_noise", 0.0)
    a_noise = stats.get("A_noise", 0.0)
    n_rir = stats.get("N_rir", 0)

    base = max(h_pos + h_neg, 0.01)

    # Augmentation rounds
    progress = _clamp(np.log1p(base) / np.log1p(5), 0.0, 1.0)
    target_hours = 8.0 + 12.0 * progress
    multiplier = target_hours / base if base > 0.01 else 10
    aug_rounds = int(round(_clamp(multiplier, 2, 5)))

    effective = base * aug_rounds

    # Quality score
    quality = ((1 - _clamp(a_noise, 0, 1)) + _clamp(n_rir / 500, 0, 1)) / 2

    # Steps
    steps = int(_clamp(int(effective * 1000 * (1.1 - 0.2 * quality)), 10000, 40000))

    # Model complexity
    complexity = _clamp(np.log10(effective + 1) * 2.0, 1.0, 4.0)
    n_blocks = int(round(complexity))
    layer_size = int(_clamp(64 * (2 ** (n_blocks - 1)), 64, 512))

    # Learning rate
    size_factor = _clamp((effective / 20) ** 0.1, 0.8, 2.0)
    noise_factor = _clamp((1 - _clamp(a_noise, 0, 1)) ** 2, 0.5, 1.0)
    lr_max = 5e-5 * size_factor * noise_factor
    lr_base = lr_max / 10

    # Dropout
    capacity = n_blocks * (layer_size**2)
    dataset_proxy = effective * 3600
    risk = capacity / (dataset_proxy * 1000 + 1e-6)
    dropout = _clamp(0.6 + risk * 0.75, 0.4, 0.8)

    # LR schedule
    num_cycles = _clamp(effective / 25, 2, 4)
    cycle_steps = steps / num_cycles
    clr_up = int(cycle_steps * 0.4)
    clr_down = int(cycle_steps * 0.6)

    # Augmentation batch size
    try:
        import os
        import psutil

        ram_gb = max(0, psutil.virtual_memory().total / (1024**3) - 2.0)
        core_factor = math.sqrt((os.cpu_count() or 4) / 4.0)
        aug_bs = min([16, 32, 64, 128], key=lambda x: abs(x - _clamp(16 * (ram_gb / 6) * core_factor, 16, 128)))
    except ImportError:
        aug_bs = 32

    return {
        "augmentation_rounds": aug_rounds,
        "steps": steps,
        "n_blocks": n_blocks,
        "layer_size": layer_size,
        "learning_rate_max": lr_max,
        "learning_rate_base": lr_base,
        "dropout": dropout,
        "clr_step_size_up": clr_up,
        "clr_step_size_down": clr_down,
        "augmentation_batch_size": aug_bs,
    }
