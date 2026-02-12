"""TTS positive sample generation for wake word training.

Generates spoken variants of the target wake word using multiple TTS engines:
  - piper: nanobuddy's built-in Piper TTS (904+ speakers)
  - kokoro: Kokoro-82M StyleTTS2-based (54 voices)
  - coqui: Coqui VITS VCTK (109 speakers)
  - all: split count across all engines

Supports:
  - Bare wake word ("clarvis")
  - Prefixed variants ("hey clarvis", "ok clarvis", etc.)
"""

DEFAULT_PREFIXES = []

ALL_ENGINES = ["piper", "kokoro", "coqui"]


def _get_generate_fn(engine: str):
    """Return the generate_samples function for the given engine."""
    if engine == "piper":
        from nanobuddy.data.generator.generate_samples import generate_samples
        from functools import partial
        import random as _rng
        _rng.seed(42)
        random_speeds = [round(_rng.uniform(0.8, 1.2), 3) for _ in range(200)]
        return partial(generate_samples, models="en_US-libritts-high",
                       length_scales=random_speeds)
    elif engine == "kokoro":
        from .engines.kokoro import generate_samples
        return generate_samples
    elif engine == "coqui":
        from .engines.coqui import generate_samples
        return generate_samples
    else:
        raise ValueError(f"Unknown engine: {engine}")


def generate_positives(target_word: str, output_dir: str, count: int = 2500,
                       prefixes: list[str] | None = None,
                       prefix_count: int = 250,
                       engine: str = "piper"):
    """Generate TTS positive samples for the target wake word."""
    if prefixes is None:
        prefixes = DEFAULT_PREFIXES

    if engine == "all":
        per_engine = count // len(ALL_ENGINES)
        per_prefix = prefix_count // len(ALL_ENGINES)
        remainder = count % len(ALL_ENGINES)
        for i, eng in enumerate(ALL_ENGINES):
            extra = remainder if i == 0 else 0
            generate_positives(target_word, output_dir,
                               count=per_engine + extra,
                               prefixes=prefixes,
                               prefix_count=max(1, per_prefix),
                               engine=eng)
        return

    generate_samples = _get_generate_fn(engine)
    print(f"\n[{engine}] Generating {count} '{target_word}' clips")

    # Bare wake word
    print(f"\n{'='*60}")
    print(f"[{engine}] {count} '{target_word}' clips (prefix: pos)")
    print(f"{'='*60}\n")
    generate_samples(
        text=target_word,
        output_dir=output_dir,
        max_samples=count,
        file_prefix="pos",
    )

    # Prefixed variants
    for prefix in prefixes:
        phrase = f"{prefix} {target_word}"
        print(f"\n{'='*60}")
        print(f"[{engine}] {prefix_count} '{phrase}' clips (prefix: {prefix})")
        print(f"{'='*60}\n")
        generate_samples(
            text=phrase,
            output_dir=output_dir,
            max_samples=prefix_count,
            file_prefix=prefix,
        )

    print(f"\n[{engine}] All positives generated in {output_dir}")


def run_generate(target: str, output_dir: str = "./positive",
                 count: int = 2500, prefixes: str | None = None,
                 prefix_count: int = 250, engine: str = "piper"):
    """CLI entry point for 'nanobuddy generate'."""
    generate_positives(
        target_word=target,
        output_dir=output_dir,
        count=count,
        prefixes=prefixes.split(",") if prefixes else None,
        prefix_count=prefix_count,
        engine=engine,
    )
