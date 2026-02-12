"""Piper TTS engine — delegates to nanobuddy's built-in generate_samples.

This is a thin wrapper that exposes the same interface as the other engines
(kokoro, coqui) but uses nanobuddy's built-in Piper TTS (904+ speakers).
"""

from functools import partial
import random as _rng

from nanobuddy.data.generator.generate_samples import generate_samples as _piper_generate


def generate_samples(text, output_dir, max_samples, file_prefix="sample", **kwargs):
    """Generate TTS samples using Piper with random speed variation."""
    _rng.seed(42)
    random_speeds = [round(_rng.uniform(0.8, 1.2), 3) for _ in range(200)]
    fn = partial(_piper_generate, models="en_US-libritts-high",
                 length_scales=random_speeds)
    fn(text=text, output_dir=output_dir, max_samples=max_samples,
       file_prefix=file_prefix)
