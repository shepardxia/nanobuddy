"""Split data directories into train/test sets.

Moves a fraction of WAV files from each source directory into a corresponding
test directory. Idempotent: reverses any existing split before re-splitting.
"""

import random
import shutil
from pathlib import Path


def split_data(project_dir: str, categories: list[tuple[str, str]] | None = None,
               test_fraction: float = 0.05, seed: int = 42):
    """Split data directories into train/test.

    Args:
        project_dir: Root directory containing the data subdirectories.
        categories: List of (source_name, test_name) tuples.
                    Defaults to standard wake word layout.
        test_fraction: Fraction of files to move to test set.
        seed: Random seed for reproducibility.
    """
    random.seed(seed)
    base = Path(project_dir)

    if categories is None:
        categories = [
            ("positive_real", "test_positive_real"),
            ("negative_real", "test_negative_real"),
            ("positive", "test_positive"),
            ("negative", "test_negative"),
        ]

    for src_name, dst_name in categories:
        src = base / src_name
        dst = base / dst_name
        if not src.exists():
            print(f"Skipping {src_name}: directory not found")
            continue
        dst.mkdir(exist_ok=True)

        # Move any existing test files back first (idempotent)
        for f in dst.glob("*.wav"):
            shutil.move(str(f), str(src / f.name))

        wavs = sorted(src.glob("*.wav"))
        n_test = max(1, int(len(wavs) * test_fraction))
        test_set = random.sample(wavs, min(n_test, len(wavs)))

        for f in test_set:
            shutil.move(str(f), str(dst / f.name))

        print(f"{src_name}: {len(wavs)} total -> {len(test_set)} test, "
              f"{len(wavs) - len(test_set)} train")


def run_split(project_dir: str, test_fraction: float = 0.05, seed: int = 42):
    """CLI entry point for 'nanobuddy split'."""
    split_data(
        project_dir=project_dir,
        test_fraction=test_fraction,
        seed=seed,
    )
