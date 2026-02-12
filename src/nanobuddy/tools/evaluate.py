"""Model evaluation on held-out test data.

Runs nanobuddy inference on every WAV in the specified test directories,
computes detection scores, and reports recall/FPR/precision at multiple
thresholds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from nanobuddy import WakeEngine

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def score_clip(engine: WakeEngine, wav_path: str) -> float:
    """Return the max detection score across all chunks of a clip."""
    engine.reset()
    predictions = engine.predict_clip(str(wav_path))
    if not predictions:
        return 0.0
    return max(predictions)


def score_directory(engine: WakeEngine, directory: Path) -> list[tuple[str, float]]:
    """Score every WAV in a directory."""
    wavs = sorted(directory.glob("*.wav"))
    return [(wav.name, score_clip(engine, str(wav))) for wav in wavs]


def evaluate(
    model_path: str,
    positive_dirs: list[str],
    negative_dirs: list[str],
    output_path: str | None = None,
) -> dict:
    """Run full evaluation and return metrics dict."""
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    engine = WakeEngine(str(model_path))

    all_scores: dict[str, list[tuple[str, float]]] = {}

    for d in positive_dirs:
        dp = Path(d)
        if not dp.exists():
            print(f"Warning: {dp} not found, skipping")
            continue
        label = f"positive:{dp.name}"
        print(f"\nScoring {label} ({dp})...")
        results = score_directory(engine, dp)
        all_scores[label] = results
        print(f"  Scored {len(results)} clips")

    for d in negative_dirs:
        dp = Path(d)
        if not dp.exists():
            print(f"Warning: {dp} not found, skipping")
            continue
        label = f"negative:{dp.name}"
        print(f"\nScoring {label} ({dp})...")
        results = score_directory(engine, dp)
        all_scores[label] = results
        print(f"  Scored {len(results)} clips")

    pos_scores = [s for label, results in all_scores.items() if label.startswith("positive") for _, s in results]
    neg_scores = [s for label, results in all_scores.items() if label.startswith("negative") for _, s in results]

    print(f"\n{'='*60}\nSCORE DISTRIBUTIONS\n{'='*60}")
    for label, results in all_scores.items():
        _print_dist(label, [s for _, s in results])
    _print_dist("ALL POSITIVES", pos_scores)
    _print_dist("ALL NEGATIVES", neg_scores)

    print(f"\n{'='*60}\nMETRICS BY THRESHOLD\n{'='*60}")
    print(f"{'Thresh':>8}  {'Recall':>8}  {'FPR':>8}  {'Precision':>10}  {'TP':>5}  {'FP':>5}  {'FN':>5}  {'TN':>5}")

    pos_arr, neg_arr = np.array(pos_scores), np.array(neg_scores)
    metrics_by_threshold = {}

    for t in THRESHOLDS:
        tp = int((pos_arr >= t).sum())
        fn = int((pos_arr < t).sum())
        fp = int((neg_arr >= t).sum())
        tn = int((neg_arr < t).sum())
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        print(f"{t:>8.2f}  {recall:>8.4f}  {fpr:>8.4f}  {precision:>10.4f}  {tp:>5}  {fp:>5}  {fn:>5}  {tn:>5}")
        metrics_by_threshold[str(t)] = {"recall": recall, "fpr": fpr, "precision": precision, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    result = {
        "model": str(model_path),
        "n_positive": len(pos_scores),
        "n_negative": len(neg_scores),
        "mean_positive": float(pos_arr.mean()) if len(pos_arr) else 0,
        "mean_negative": float(neg_arr.mean()) if len(neg_arr) else 0,
        "thresholds": metrics_by_threshold,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nMetrics saved to {output_path}")

    return result


def _print_dist(label: str, scores: list[float]):
    arr = np.array(scores)
    if len(arr) == 0:
        print(f"  {label}: (empty)")
        return
    print(f"  {label} (n={len(arr)}): mean={arr.mean():.4f} std={arr.std():.4f} min={arr.min():.4f} med={np.median(arr):.4f} max={arr.max():.4f}")
