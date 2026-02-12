"""Audio clip quality validation."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .config import CollectConfig


class Severity(Enum):
    PASS = "PASS"
    WARN = "WARN"
    REJECT = "REJECT"


@dataclass
class ValidationResult:
    severity: Severity
    check: str
    detail: str


@dataclass
class ClipReport:
    duration_s: float
    rms_dbfs: float
    peak_dbfs: float
    speech_ms: int
    results: list[ValidationResult]

    @property
    def passed(self) -> bool:
        return all(r.severity != Severity.REJECT for r in self.results)

    @property
    def worst(self) -> Severity:
        if any(r.severity == Severity.REJECT for r in self.results):
            return Severity.REJECT
        if any(r.severity == Severity.WARN for r in self.results):
            return Severity.WARN
        return Severity.PASS

    @property
    def reject_reasons(self) -> list[str]:
        return [r.check for r in self.results if r.severity == Severity.REJECT]


def _rms_dbfs(audio: np.ndarray) -> float:
    """Compute RMS level in dBFS for int16 audio."""
    float_audio = audio.astype(np.float64) / 32768.0
    rms = np.sqrt(np.mean(float_audio ** 2))
    if rms < 1e-10:
        return -100.0
    return 20.0 * np.log10(rms)


def _peak_dbfs(audio: np.ndarray) -> float:
    """Compute peak level in dBFS for int16 audio."""
    peak = np.max(np.abs(audio.astype(np.float64))) / 32768.0
    if peak < 1e-10:
        return -100.0
    return 20.0 * np.log10(peak)


def _clip_fraction(audio: np.ndarray) -> float:
    """Fraction of samples at max int16 amplitude."""
    threshold = 32700  # near-max for int16
    return np.mean(np.abs(audio) >= threshold)


def _dc_offset(audio: np.ndarray) -> float:
    """DC offset as fraction of full scale."""
    return abs(np.mean(audio.astype(np.float64)) / 32768.0)


def _vad_frames(audio: np.ndarray, sample_rate: int, threshold: float = 0.5):
    """Classify 30ms frames as speech/non-speech using Silero VAD.

    Returns (speech_frames, noise_frames) as lists of int16 arrays.
    """
    from nanobuddy.vad import VAD

    vad = VAD()
    frame_samples = sample_rate * 30 // 1000  # 480 samples = 30ms

    speech_frames = []
    noise_frames = []

    for i in range(0, len(audio) - frame_samples + 1, frame_samples):
        frame = audio[i:i + frame_samples]
        prob = vad.predict(frame, frame_size=frame_samples)
        if prob >= threshold:
            speech_frames.append(frame)
        else:
            noise_frames.append(frame)

    return speech_frames, noise_frames


def _speech_ms_from_frames(speech_frames: list[np.ndarray]) -> int:
    """Speech duration in ms from pre-computed VAD speech frames (30ms each)."""
    return len(speech_frames) * 30


def _snr_from_frames(
    speech_frames: list[np.ndarray],
    noise_frames: list[np.ndarray],
) -> float:
    """Rough SNR estimate from pre-computed VAD frames: speech RMS vs noise RMS."""
    if not speech_frames or not noise_frames:
        return 100.0  # can't estimate, assume fine

    speech_rms = np.sqrt(np.mean(np.concatenate(speech_frames).astype(np.float64) ** 2))
    noise_rms = np.sqrt(np.mean(np.concatenate(noise_frames).astype(np.float64) ** 2))

    if noise_rms < 1e-10:
        return 100.0
    return 20.0 * np.log10(speech_rms / noise_rms)


def validate_clip(audio: np.ndarray, cfg: CollectConfig) -> ClipReport:
    """Run all quality checks on an audio clip (int16, 16kHz mono)."""
    results = []
    duration_s = len(audio) / cfg.sample_rate
    rms = _rms_dbfs(audio)
    peak = _peak_dbfs(audio)

    # Run VAD once, reuse for both speech duration and SNR checks
    speech_frames, noise_frames = _vad_frames(audio, cfg.sample_rate)
    speech_ms = _speech_ms_from_frames(speech_frames)

    if rms < cfg.min_rms_dbfs:
        results.append(ValidationResult(
            Severity.REJECT, "too_quiet",
            f"RMS {rms:.1f} dBFS < {cfg.min_rms_dbfs} dBFS"))

    if peak > cfg.max_peak_dbfs:
        results.append(ValidationResult(
            Severity.REJECT, "clipping_peak",
            f"Peak {peak:.1f} dBFS > {cfg.max_peak_dbfs} dBFS"))

    clip_frac = _clip_fraction(audio)
    if clip_frac > cfg.max_clip_fraction:
        results.append(ValidationResult(
            Severity.REJECT, "clipping_samples",
            f"{clip_frac:.3%} samples clipped > {cfg.max_clip_fraction:.3%}"))

    if speech_ms < cfg.min_speech_ms:
        results.append(ValidationResult(
            Severity.REJECT, "too_short",
            f"Speech {speech_ms}ms < {cfg.min_speech_ms}ms"))

    if duration_s > cfg.max_clip_seconds:
        results.append(ValidationResult(
            Severity.REJECT, "too_long",
            f"Duration {duration_s:.1f}s > {cfg.max_clip_seconds}s"))

    dc = _dc_offset(audio)
    if dc > cfg.dc_offset_threshold:
        results.append(ValidationResult(
            Severity.REJECT, "dc_offset",
            f"DC offset {dc:.4f} > {cfg.dc_offset_threshold}"))

    snr = _snr_from_frames(speech_frames, noise_frames)
    if snr < cfg.snr_warn_db:
        results.append(ValidationResult(
            Severity.WARN, "low_snr",
            f"Estimated SNR {snr:.1f} dB < {cfg.snr_warn_db} dB"))

    if not results:
        results.append(ValidationResult(Severity.PASS, "all_checks", "OK"))

    return ClipReport(
        duration_s=duration_s,
        rms_dbfs=rms,
        peak_dbfs=peak,
        speech_ms=speech_ms,
        results=results,
    )
