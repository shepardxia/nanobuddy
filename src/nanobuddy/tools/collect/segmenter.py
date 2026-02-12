"""Segment continuous audio into positive clips + per-gap negative files."""

from dataclasses import dataclass

import numpy as np
import soundfile as sf

from .config import CollectConfig
from .validator import validate_clip


@dataclass
class HoldWindow:
    """A press-and-hold window marking a sample."""
    start_sample: int
    end_sample: int
    kind: str = "pos"  # "pos" or "neg"


@dataclass
class SegmentResult:
    """Results from segmenting a recording session."""
    positive_saved: int = 0
    positive_rejected: int = 0
    explicit_negative_saved: int = 0
    gap_negative_saved: int = 0
    negative_duration_s: float = 0.0
    reports: list = None

    def __post_init__(self):
        if self.reports is None:
            self.reports = []

    @property
    def negative_saved(self) -> int:
        return self.explicit_negative_saved + self.gap_negative_saved


def _count_existing(directory, prefix: str) -> int:
    """Count existing numbered files to continue numbering."""
    return len(list(directory.glob(f"{prefix}_*.wav")))


def segment_recording(
    audio: np.ndarray,
    holds: list[HoldWindow],
    cfg: CollectConfig,
) -> SegmentResult:
    """Segment a continuous recording into positive clips + negative files.

    Positives: extracted per hold window with keypress trimming + padding.
    Negatives: each gap between consecutive positives saved as its own file.
    Numbering continues from existing files in the output dirs.

    Args:
        audio: Full recording buffer (int16, mono).
        holds: List of hold windows from recorder.
        cfg: Collection config.

    Returns:
        SegmentResult with counts.
    """
    cfg.ensure_dirs()
    result = SegmentResult()
    sr = cfg.sample_rate
    trim_start = int(cfg.keypress_trim_start_ms * sr / 1000)
    trim_end = int(cfg.keypress_trim_end_ms * sr / 1000)

    # Continue numbering from existing files
    pos_idx = _count_existing(cfg.positive_dir, cfg.phrase_slug) + \
              _count_existing(cfg.rejected_dir, cfg.phrase_slug)
    neg_idx = _count_existing(cfg.negative_dir, "neg")

    pos_holds = [h for h in holds if h.kind == "pos"]
    neg_holds = [h for h in holds if h.kind == "neg"]

    # --- Positive clips (trimmed to exclude keypress sound) ---
    for hold in pos_holds:
        clip_start = max(0, hold.start_sample + trim_start)
        clip_end = min(len(audio), hold.end_sample - trim_end)
        if clip_end <= clip_start:
            continue

        clip = audio[clip_start:clip_end]
        if len(clip) == 0:
            continue

        report = validate_clip(clip, cfg)
        result.reports.append(report)

        pos_idx += 1
        if report.passed:
            fname = f"{cfg.phrase_slug}_{pos_idx:04d}.wav"
            sf.write(str(cfg.positive_dir / fname), clip, sr, subtype="PCM_16")
            result.positive_saved += 1
        else:
            reasons = "_".join(report.reject_reasons)
            fname = f"{cfg.phrase_slug}_{pos_idx:04d}_{reasons}.wav"
            sf.write(str(cfg.rejected_dir / fname), clip, sr, subtype="PCM_16")
            result.positive_rejected += 1

    # --- Explicit negatives (N-key holds, trimmed same way) ---
    for hold in neg_holds:
        clip_start = max(0, hold.start_sample + trim_start)
        clip_end = min(len(audio), hold.end_sample - trim_end)
        if clip_end <= clip_start:
            continue

        clip = audio[clip_start:clip_end]
        if len(clip) == 0:
            continue

        neg_idx += 1
        fname = f"neg_{neg_idx:04d}.wav"
        sf.write(str(cfg.negative_dir / fname), clip, sr, subtype="PCM_16")
        result.explicit_negative_saved += 1
        result.negative_duration_s += len(clip) / sr

    # --- Gap negatives: one file per gap between ALL hold windows ---
    exclusion_ranges = sorted(
        (h.start_sample, h.end_sample) for h in holds
    )
    min_neg_samples = int(0.3 * sr)  # at least 300ms

    gaps = []
    cursor = 0
    for es, ee in exclusion_ranges:
        if es > cursor:
            gaps.append((cursor, es))
        cursor = max(cursor, ee)
    if cursor < len(audio):
        gaps.append((cursor, len(audio)))

    for gap_start, gap_end in gaps:
        chunk = audio[gap_start:gap_end]
        if len(chunk) < min_neg_samples:
            continue
        neg_idx += 1
        fname = f"neg_{neg_idx:04d}.wav"
        sf.write(str(cfg.negative_dir / fname), chunk, sr, subtype="PCM_16")
        result.gap_negative_saved += 1
        result.negative_duration_s += len(chunk) / sr

    return result
