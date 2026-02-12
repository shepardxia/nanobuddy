"""Path constants for bundled ONNX models.

Models are stored alongside this package. If missing, they're downloaded
on first access via ensure_model().
"""

from pathlib import Path

from nanobuddy.onnx_utils import ensure_model

_MODELS_DIR = Path(__file__).parent

DEFAULT_WAKE_WORD_MODEL = _MODELS_DIR / "wake_word" / "clarvis.onnx"


def mel_path() -> Path:
    return ensure_model("melspectrogram.onnx", _MODELS_DIR / "mel_spectrogram")


def embedding_path() -> Path:
    return ensure_model("embedding_model.onnx", _MODELS_DIR / "embedding")


def vad_path() -> Path:
    return ensure_model("silero_vad.onnx", _MODELS_DIR / "vad")
