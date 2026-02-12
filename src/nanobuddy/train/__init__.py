"""NanoBuddy training subpackage.

Requires ``nanobuddy[train]`` extras (torch, torchaudio, etc.).
"""

from nanobuddy.train.config import Config, load_config

__all__ = ["ARCHITECTURES", "Config", "Trainer", "load_config"]


def __getattr__(name):
    if name == "ARCHITECTURES":
        from nanobuddy.train.architectures import ARCHITECTURES
        return ARCHITECTURES
    if name == "Trainer":
        from nanobuddy.train.trainer import Trainer
        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
