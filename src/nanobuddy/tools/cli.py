"""NanoBuddy CLI — training, evaluation, data collection, and preparation tools.

Usage::

    nanobuddy collect clarvis --target 400 --output ./collected
    nanobuddy validate ./collected
    nanobuddy stats ./collected
    nanobuddy train -c config.yaml
    nanobuddy evaluate --model model.onnx --positive-dir pos/ --negative-dir neg/
    nanobuddy adversarial clarvis --engine piper
    nanobuddy generate clarvis --output-dir ./positive
    nanobuddy segment ./recordings --output-dir ./negative_real
    nanobuddy download noise --output-dir ./freesound
    nanobuddy split ./project --test-fraction 0.05
    nanobuddy augment ./clips --output-dir ./augmented
    nanobuddy ambient ./recordings --output-dir ./noise
"""

import click


@click.group()
def main():
    """NanoBuddy — wake word training toolkit."""


# ── train ──────────────────────────────────────────────────────────

@main.command()
@click.option("-c", "--config", "config_path", required=True, help="Path to YAML config file")
@click.option("-o", "--output-dir", default="output", help="Output directory")
@click.option("--override", multiple=True, help="Dot-notation override (e.g. training.steps=20000)")
def train(config_path, output_dir, override):
    """Train a wake word model."""
    from nanobuddy.train import Trainer, load_config

    config = load_config(config_path, overrides=list(override))
    trainer = Trainer(config)
    click.echo(f"Training {config.model.architecture} for {config.training.steps} steps...")
    # Note: full data loading + train_loader creation is project-specific
    # This is a placeholder for the CLI entry point
    click.echo("Train command registered. Full data pipeline integration TBD.")


# ── evaluate ───────────────────────────────────────────────────────

@main.command()
@click.option("--model", required=True, help="Path to .onnx model")
@click.option("--positive-dir", multiple=True, help="Positive test directory (repeatable)")
@click.option("--negative-dir", multiple=True, help="Negative test directory (repeatable)")
@click.option("--output", help="Output path for JSON metrics")
def evaluate(model, positive_dir, negative_dir, output):
    """Evaluate a trained model on test data."""
    from nanobuddy.tools.evaluate import evaluate as run_eval

    run_eval(
        model_path=model,
        positive_dirs=list(positive_dir),
        negative_dirs=list(negative_dir),
        output_path=output,
    )


# ── adversarial ────────────────────────────────────────────────────

@main.command()
@click.argument("target")
@click.option("--output-dir", default="./negative", help="Output directory")
@click.option("--phonemes", multiple=True, help="Explicit phonemes (e.g. K L AA R V IH S)")
@click.option("--min-score", type=int, default=1, help="Min bigram overlap score")
@click.option("--max-words", type=int, default=0, help="Max total words (0=no cap)")
@click.option("--dry-run", is_flag=True, help="Show words without generating")
@click.option("--engine", type=click.Choice(["piper", "kokoro", "coqui", "all"]), default="piper")
@click.option("--cartesian/--no-cartesian", default=True, help="Include cartesian prefixed negatives")
def adversarial(target, output_dir, phonemes, min_score, max_words, dry_run, engine, cartesian):
    """Generate phoneme-based adversarial negatives."""
    from nanobuddy.tools.adversarial import run_adversarial

    run_adversarial(
        target=target, output_dir=output_dir,
        phonemes=list(phonemes) if phonemes else None,
        min_score=min_score, max_words=max_words,
        dry_run=dry_run, engine=engine, cartesian=cartesian,
    )


# ── segment ────────────────────────────────────────────────────────

@main.command()
@click.argument("input_path")
@click.option("--output-dir", default="./negative_real", help="Output directory")
@click.option("--target", default="clarvis", help="Target wake word for filtering")
@click.option("--prefix", default="real", help="Output file prefix")
@click.option("--whisper-model", default="base", help="Whisper model size")
@click.option("--skip-words", help="Comma-separated words to skip")
def segment(input_path, output_dir, target, prefix, whisper_model, skip_words):
    """Segment recordings into word-level clips via Whisper."""
    from nanobuddy.tools.segment import run_segment

    run_segment(
        input_path=input_path, output_dir=output_dir,
        target=target, prefix=prefix, whisper_model=whisper_model,
        skip_words=skip_words,
    )


# ── download ───────────────────────────────────────────────────────

@main.command()
@click.argument("download_type", type=click.Choice(["noise", "rir", "speech"]))
@click.option("--output-dir", required=True, help="Output directory")
@click.option("--max-clips", type=int, default=0, help="Max clips (0=all)")
@click.option("--skip-words", help="Words to skip in transcripts (speech)")
def download(download_type, output_dir, max_clips, skip_words):
    """Download training data from HuggingFace."""
    from nanobuddy.tools.download import run_download

    run_download(
        download_type=download_type, output_dir=output_dir,
        max_clips=max_clips, skip_words=skip_words,
    )


# ── generate ───────────────────────────────────────────────────────

@main.command()
@click.argument("target")
@click.option("--output-dir", default="./positive", help="Output directory")
@click.option("--count", type=int, default=2500, help="Number of bare wake word clips")
@click.option("--prefixes", help="Comma-separated prefixes (default: hey,hi)")
@click.option("--prefix-count", type=int, default=250, help="Clips per prefix variant")
@click.option("--engine", type=click.Choice(["piper", "kokoro", "coqui", "all"]), default="piper")
def generate(target, output_dir, count, prefixes, prefix_count, engine):
    """Generate TTS positive samples."""
    from nanobuddy.tools.generate import run_generate

    run_generate(
        target=target, output_dir=output_dir,
        count=count, prefixes=prefixes,
        prefix_count=prefix_count, engine=engine,
    )


# ── split ──────────────────────────────────────────────────────────

@main.command()
@click.argument("project_dir")
@click.option("--test-fraction", type=float, default=0.05, help="Test set fraction")
@click.option("--seed", type=int, default=42, help="Random seed")
def split(project_dir, test_fraction, seed):
    """Split data into train/test sets."""
    from nanobuddy.tools.split import run_split

    run_split(project_dir=project_dir, test_fraction=test_fraction, seed=seed)


# ── augment ────────────────────────────────────────────────────────

@main.command()
@click.argument("input_dir")
@click.option("--output-dir", required=True, help="Output directory")
@click.option("--repeats", type=int, default=5, help="Augmented variants per clip")
@click.option("--batch-size", type=int, default=32, help="Processing batch size")
def augment(input_dir, output_dir, repeats, batch_size):
    """Augment WAV clips (pitch, EQ, gain, noise)."""
    from nanobuddy.tools.augment_cli import run_augment

    run_augment(
        input_dir=input_dir, output_dir=output_dir,
        repeats=repeats, batch_size=batch_size,
    )


# ── ambient ────────────────────────────────────────────────────────

@main.command()
@click.argument("input_dir")
@click.option("--output-dir", required=True, help="Output directory")
@click.option("--vad-threshold", type=float, default=0.3, help="Max VAD prob for ambient")
@click.option("--rms-threshold", type=float, default=-40.0, help="Min RMS in dBFS")
@click.option("--min-segment", type=int, default=500, help="Min segment length (ms)")
@click.option("--max-segment", type=int, default=5000, help="Max segment length (ms)")
def ambient(input_dir, output_dir, vad_threshold, rms_threshold, min_segment, max_segment):
    """Extract ambient noise from recordings via VAD."""
    from nanobuddy.tools.ambient import run_ambient

    run_ambient(
        input_dir=input_dir, output_dir=output_dir,
        vad_threshold=vad_threshold, rms_threshold=rms_threshold,
        min_segment=min_segment, max_segment=max_segment,
    )


# ── collect ───────────────────────────────────────────────────────

@main.command()
@click.argument("phrase")
@click.option("--target", type=int, default=400, help="Target positive samples")
@click.option("--output", default="./collected", help="Output directory")
@click.option("--device", type=int, default=None, help="Audio input device index")
@click.option("--auto", is_flag=True, help="Hands-free auto-capture mode")
@click.option("--auto-interval", type=float, default=3.0, help="Seconds between auto captures")
@click.option("--auto-duration", type=float, default=1.5, help="Capture duration in auto mode")
def collect(phrase, target, output, device, auto, auto_interval, auto_duration):
    """Record wake word samples interactively.

    SPACE to mark positive, N for negative, R to undo, Q to quit.
    """
    from nanobuddy.tools.collect import run_collect

    run_collect(
        phrase=phrase, target=target, output=output,
        device=device, auto=auto,
        auto_interval=auto_interval, auto_duration=auto_duration,
    )


# ── validate (collection) ────────────────────────────────────────

@main.command("validate")
@click.argument("path")
def validate_collection(path):
    """Validate quality of collected samples."""
    from nanobuddy.tools.collect import run_validate

    run_validate(path=path)


# ── stats ─────────────────────────────────────────────────────────

@main.command()
@click.argument("path")
def stats(path):
    """Show statistics for a collected dataset."""
    from nanobuddy.tools.collect import run_stats

    run_stats(path=path)


if __name__ == "__main__":
    main()
