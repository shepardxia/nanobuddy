"""Kokoro TTS engine wrapper for wake word sample generation.

Kokoro (hexgrad/Kokoro-82M): 82M param StyleTTS2-based model, 54 voices,
outputs float32 audio at 24kHz. We resample to 16kHz and apply standard
post-processing.
"""

import os
import itertools
import numpy as np
from tqdm import tqdm

from .base import resample, post_process, save_wav, make_filename, TARGET_SAMPLE_RATE

KOKORO_SAMPLE_RATE = 24000

# English voices only (af=American female, am=American male, bf=British female, bm=British male)
ENGLISH_VOICE_PREFIXES = ("af_", "am_", "bf_", "bm_")


def _load_pipeline():
    """Lazy-load Kokoro pipeline."""
    import warnings
    warnings.filterwarnings("ignore")
    from kokoro import KPipeline
    return KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')


def _get_english_voices(pipeline):
    """Get all English voice names from the pipeline's cache."""
    from huggingface_hub import list_repo_tree
    voices = []
    for item in list_repo_tree('hexgrad/Kokoro-82M', path_in_repo='voices'):
        if item.path.endswith('.pt'):
            name = item.path.split('/')[-1].replace('.pt', '')
            if name.startswith(ENGLISH_VOICE_PREFIXES):
                voices.append(name)
    return sorted(voices)


def generate_samples(text, output_dir, max_samples, file_prefix="sample",
                     **kwargs):
    """Generate TTS samples using Kokoro with voice mixing and random speed.

    Args:
        text: String or list of strings to synthesize.
        output_dir: Directory to write WAV files.
        max_samples: Total number of samples to generate.
        file_prefix: Filename prefix for output files.
    """
    import random
    from itertools import combinations

    os.makedirs(output_dir, exist_ok=True)
    pipeline = _load_pipeline()
    voices = _get_english_voices(pipeline)

    if not voices:
        raise RuntimeError("No English Kokoro voices found. Run download first.")

    # Build voice pool: singles + pairwise mixes
    voice_pool = list(voices)
    pairs = [f"{a},{b}" for a, b in combinations(voices, 2)]
    random.seed(42)
    random.shuffle(pairs)
    voice_pool.extend(pairs)

    print(f"Kokoro: {len(voices)} base voices, {len(pairs)} mixed pairs = {len(voice_pool)} total")

    if isinstance(text, str):
        text = [text]
    prompts = (text * ((max_samples // len(text)) + 1))[:max_samples]
    voice_cycle = itertools.cycle(voice_pool)

    for i, prompt in enumerate(tqdm(prompts, desc="Kokoro TTS", unit="sample")):
        voice_name = next(voice_cycle)
        speed = random.uniform(0.8, 1.2)
        try:
            chunks = list(pipeline(prompt, voice=voice_name, speed=speed))
            if not chunks:
                continue
            audio = np.concatenate([c.audio.numpy() for c in chunks if c.audio is not None])

            # Resample 24kHz -> 16kHz
            audio_16k = resample(audio * 32767, KOKORO_SAMPLE_RATE)
            audio_out = post_process(audio_16k)

            voice_tag = voice_name.replace(",", "+")
            filename = make_filename(file_prefix, "kok", voice_tag, i)
            save_wav(audio_out, os.path.join(output_dir, filename))
        except Exception as e:
            print(f"Kokoro error on sample {i} ('{prompt}', voice={voice_name}, speed={speed:.2f}): {e}")
