"""Typed dataclass config with YAML loading and dot-override support."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    architecture: str = "e_branchformer"
    embedding_dim: int = 1
    n_features: int = 16
    dropout: float = 0.1
    # Architecture-specific params forwarded as-is
    extra: dict = field(default_factory=dict)


@dataclass
class TrainingConfig:
    steps: int = 20000
    batch_size: int = 64
    learning_rate_max: float = 5e-5
    learning_rate_base: float = 5e-6
    lr_scheduler_type: str = "onecycle"
    optimizer_type: str = "adamw"
    weight_decay: float = 0.01
    loss_bias: float = 0.9
    early_stopping_patience: int | None = None
    validation_interval: int = 500
    checkpoint_interval: int = 1000
    checkpoint_limit: int = 3
    warmup_fraction: float = 0.15
    ema_alpha: float = 0.01
    seed: int = 10


@dataclass
class AugmentConfig:
    rounds: int = 3
    batch_size: int = 32
    gain_prob: float = 1.0
    min_gain_db: float = -6.0
    max_gain_db: float = 6.0
    pitch_prob: float = 0.3
    min_pitch_semitones: float = -2.0
    max_pitch_semitones: float = 2.0
    rir_prob: float = 0.2
    noise_prob: float = 0.3
    min_snr_db: float = 3.0
    max_snr_db: float = 30.0


@dataclass
class DataConfig:
    positive_dir: str = ""
    negative_dir: str = ""
    noise_dir: str = ""
    rir_dir: str = ""
    output_dir: str = "output"
    val_split: float = 0.1
    sample_rate: int = 16000
    clip_length_s: float = 1.5


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    onnx_opset_version: int = 17
    model_name: str = "wakeword"


def load_config(yaml_path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load config from YAML with optional dot-notation overrides.

    Example override: ``"training.steps=30000"``
    """
    path = Path(yaml_path)
    raw: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}

    # Apply dot-overrides
    for override in overrides or []:
        key, _, value = override.partition("=")
        parts = key.strip().split(".")
        target = raw
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        # Auto-cast
        target[parts[-1]] = _cast(value.strip())

    return _build_config(raw)


def _cast(value: str) -> Any:
    """Best-effort cast from string."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _build_config(raw: dict) -> Config:
    """Build Config from a flat/nested dict."""
    model_raw = raw.get("model", {})
    known_model = {f.name for f in ModelConfig.__dataclass_fields__.values() if f.name != "extra"}
    model_known = {k: v for k, v in model_raw.items() if k in known_model}
    model_extra = {k: v for k, v in model_raw.items() if k not in known_model}
    model = ModelConfig(**model_known, extra=model_extra)

    training = TrainingConfig(**{k: v for k, v in raw.get("training", {}).items()
                                  if k in TrainingConfig.__dataclass_fields__})
    augment = AugmentConfig(**{k: v for k, v in raw.get("augment", {}).items()
                                if k in AugmentConfig.__dataclass_fields__})
    data = DataConfig(**{k: v for k, v in raw.get("data", {}).items()
                          if k in DataConfig.__dataclass_fields__})

    return Config(
        model=model,
        training=training,
        augment=augment,
        data=data,
        onnx_opset_version=raw.get("onnx_opset_version", 17),
        model_name=raw.get("model_name", "wakeword"),
    )
