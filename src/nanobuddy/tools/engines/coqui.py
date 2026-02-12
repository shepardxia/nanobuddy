"""Coqui TTS (VITS VCTK) engine wrapper for wake word sample generation.

Coqui VITS with VCTK dataset: 109 speakers, outputs float32 at 22050Hz.
We resample to 16kHz and apply standard post-processing.
Requires espeak-ng system dependency.
"""

import os
import itertools
import numpy as np
from tqdm import tqdm

from .base import resample, post_process, save_wav, make_filename, TARGET_SAMPLE_RATE

MODEL_NAME = "tts_models/en/vctk/vits"


def _load_tts():
    """Lazy-load Coqui TTS model."""
    from TTS.api import TTS
    return TTS(MODEL_NAME, progress_bar=False)


def generate_samples(text, output_dir, max_samples, file_prefix="sample",
                     **kwargs):
    """Generate TTS samples using Coqui VITS, matching nanobuddy's interface.

    Args:
        text: String or list of strings to synthesize.
        output_dir: Directory to write WAV files.
        max_samples: Total number of samples to generate.
        file_prefix: Filename prefix for output files.
    """
    os.makedirs(output_dir, exist_ok=True)
    tts = _load_tts()
    speakers = tts.speakers
    orig_sr = tts.synthesizer.output_sample_rate

    # Skip the first speaker 'ED\n' which seems malformed
    speakers = [s for s in speakers if s.strip() and '\n' not in s]

    print(f"Coqui VITS: {len(speakers)} speakers, native {orig_sr}Hz")

    if isinstance(text, str):
        text = [text]
    prompts = (text * ((max_samples // len(text)) + 1))[:max_samples]
    speaker_cycle = itertools.cycle(speakers)

    for i, prompt in enumerate(tqdm(prompts, desc="Coqui TTS", unit="sample")):
        speaker = next(speaker_cycle)
        try:
            # tts_to_file returns wav path, but we want the raw audio
            # Use tts() which returns a list of floats
            audio_list = tts.tts(text=prompt, speaker=speaker)
            audio = np.array(audio_list, dtype=np.float32)

            # Resample 22050Hz -> 16kHz and scale to int16 range
            audio_16k = resample(audio * 32767, orig_sr)
            audio_out = post_process(audio_16k)

            speaker_tag = speaker.replace(' ', '').lower()
            filename = make_filename(file_prefix, "coq", speaker_tag, i)
            save_wav(audio_out, os.path.join(output_dir, filename))
        except Exception as e:
            print(f"Coqui error on sample {i} ('{prompt}', speaker={speaker}): {e}")
