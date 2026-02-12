"""Silero VAD wrapper — stateful ONNX voice activity detection."""

import numpy as np

from nanobuddy.models import vad_path
from nanobuddy.onnx_utils import create_session


class VAD:
    """Lightweight Silero VAD with persistent hidden state."""

    def __init__(self, *, model_path: str | None = None, providers: list | None = None):
        self.session = create_session(model_path or vad_path(), providers)
        self._sr = np.array(16000, dtype=np.int64)
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def predict(self, audio: np.ndarray, frame_size: int = 480) -> float:
        """Return speech probability for an audio chunk (int16, 16 kHz)."""
        chunks = [
            (audio[i : i + frame_size] / 32767.0).astype(np.float32)
            for i in range(0, audio.shape[0], frame_size)
        ]
        scores = []
        for chunk in chunks:
            out, self._h, self._c = self.session.run(
                None,
                {"input": chunk[None, :], "h": self._h, "c": self._c, "sr": self._sr},
            )
            scores.append(out[0][0])
        return float(np.mean(scores))

    def reset(self):
        """Reset hidden state."""
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
