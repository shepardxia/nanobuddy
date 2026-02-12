"""Collection configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CollectConfig:
    """Configuration for wake word sample collection."""

    phrase: str = "hey clarvis"
    target_positive: int = 400

    def __post_init__(self):
        if self.target_positive < 1:
            raise ValueError(f"target_positive must be >= 1, got {self.target_positive}")
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"

    # Press-and-hold timing
    pre_pad_ms: int = 200
    post_pad_ms: int = 150
    min_hold_ms: int = 200
    keypress_trim_start_ms: int = 160  # trim after start keypress (key-down resonates longer)
    keypress_trim_end_ms: int = 100  # trim before end keypress

    # Auto-capture mode (hands-free recording from distance)
    auto_capture: bool = False
    auto_interval_s: float = 3.0
    auto_duration_s: float = 1.5

    # Validation thresholds
    min_rms_dbfs: float = -55.0
    max_peak_dbfs: float = -0.5
    max_clip_fraction: float = 0.001
    min_speech_ms: int = 200
    max_clip_seconds: float = 3.0
    dc_offset_threshold: float = 0.01
    snr_warn_db: float = 10.0

    # Output
    output_dir: str = "./collected"
    input_device: int | None = None

    @property
    def phrase_slug(self) -> str:
        return self.phrase.strip().lower().replace(" ", "_")

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def positive_dir(self) -> Path:
        return self.output_path / "positive"

    @property
    def negative_dir(self) -> Path:
        return self.output_path / "negative"

    @property
    def rejected_dir(self) -> Path:
        return self.output_path / "rejected"

    def ensure_dirs(self) -> None:
        for d in (self.positive_dir, self.negative_dir, self.rejected_dir):
            d.mkdir(parents=True, exist_ok=True)
