"""NanoBuddy — lightweight wake word detection."""

from nanobuddy.engine import WakeEngine

__all__ = ["WakeEngine", "WakeDetector"]


def __getattr__(name):
    if name == "WakeDetector":
        from nanobuddy.detector import WakeDetector
        return WakeDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
