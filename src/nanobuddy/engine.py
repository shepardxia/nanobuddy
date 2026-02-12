"""WakeEngine — streaming wake word inference with VAD gating and patience."""

from __future__ import annotations

import logging
import wave
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from nanobuddy.features import FeatureExtractor
from nanobuddy.onnx_utils import create_session, default_providers
from nanobuddy.vad import VAD

logger = logging.getLogger(__name__)


@dataclass
class _ModelState:
    """Inference state for the loaded model."""

    session: object  # onnxruntime.InferenceSession
    input_names: list[str]
    feature_length: int
    is_stateful: bool
    hidden: tuple[np.ndarray, np.ndarray] | None = None
    prediction_buffer: deque = field(default_factory=lambda: deque(maxlen=30))
    raw_score: float = 0.0


class WakeEngine:
    """Streaming wake word engine.

    Feed 80 ms audio chunks via ``predict()`` and get back a detection score.
    VAD gating skips model inference during silence. Patience filtering
    requires N consecutive above-threshold frames before triggering.

    Example::

        engine = WakeEngine(vad_threshold=0.3)
        score = engine.predict(chunk, threshold=0.5, patience=4)
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        vad_threshold: float = 0.0,
        providers: list | None = None,
    ):
        self._providers = providers or default_providers()
        self._features = FeatureExtractor(providers=self._providers)
        self._vad = VAD(providers=self._providers) if vad_threshold > 0 else None
        self._vad_threshold = vad_threshold

        # Load model
        if model_path is None:
            from nanobuddy.models import DEFAULT_WAKE_WORD_MODEL
            model_path = str(DEFAULT_WAKE_WORD_MODEL)
        session = create_session(model_path, self._providers)
        inputs = session.get_inputs()
        input_names = [inp.name for inp in inputs]
        self._model = _ModelState(
            session=session,
            input_names=input_names,
            feature_length=inputs[0].shape[1],
            is_stateful="hidden_in" in input_names,
        )

        # Post-processed score (returned to caller)
        self._score: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        audio: np.ndarray,
        *,
        threshold: float = 0.5,
        patience: int = 0,
        debounce_time: float = 0.0,
    ) -> float:
        """Run inference on an audio chunk (int16, 16 kHz).

        Returns 0.0 when not detected, or the raw model confidence when triggered.
        """
        ms = self._model

        # 1. Feature extraction
        n_ready = self._features(audio)
        if n_ready < 1280:
            return self._score

        # 2. VAD gate — skip model inference during silence
        if self._vad is not None:
            vad_score = self._vad.predict(audio)
            if vad_score < self._vad_threshold:
                ms.raw_score = 0.0
                ms.prediction_buffer.append(0.0)
                self._score = 0.0
                return self._score

        # 3. Model inference
        features = self._features.get_features(ms.feature_length)
        feed = {"input": features}

        if ms.is_stateful:
            h, c = ms.hidden or self._initial_state(ms.session)
            feed["hidden_in"] = h
            feed["cell_in"] = c
            out = ms.session.run(None, feed)
            score = out[0].item()
            ms.hidden = (out[1], out[2])
        else:
            out = ms.session.run(None, feed)
            score = out[0].item()

        ms.raw_score = score

        # Suppress warmup instability (first 5 frames)
        if len(ms.prediction_buffer) < 5:
            score = 0.0

        # 4. Post-processing (patience / debounce)
        score = self._apply_post_processing(score, patience, threshold, debounce_time, n_ready)

        # 5. Update buffer with RAW score, publish filtered score
        ms.prediction_buffer.append(ms.raw_score)
        self._score = score

        return self._score

    def predict_clip(
        self,
        clip: str | np.ndarray,
        chunk_size: int = 1280,
        **kwargs,
    ) -> list[float]:
        """Predict on a full audio clip by simulating a stream."""
        if isinstance(clip, str):
            with wave.open(clip, mode="rb") as f:
                if f.getframerate() != 16000 or f.getsampwidth() != 2 or f.getnchannels() != 1:
                    raise ValueError("Audio must be 16 kHz, 16-bit, mono WAV.")
                data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        elif isinstance(clip, np.ndarray):
            data = clip
        else:
            raise TypeError("`clip` must be a file path or numpy array.")

        return [self.predict(data[i : i + chunk_size], **kwargs) for i in range(0, len(data), chunk_size)]

    def reset(self):
        """Reset all internal state for a new session."""
        self._features.reset()
        if self._vad is not None:
            self._vad.reset()
        self._model.prediction_buffer.clear()
        self._model.hidden = None
        self._model.raw_score = 0.0
        self._score = 0.0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _initial_state(session) -> tuple[np.ndarray, np.ndarray]:
        h_input = next(i for i in session.get_inputs() if i.name == "hidden_in")
        c_input = next(i for i in session.get_inputs() if i.name == "cell_in")
        return (
            np.zeros(h_input.shape, dtype=np.float32),
            np.zeros(c_input.shape, dtype=np.float32),
        )

    def _apply_post_processing(
        self,
        score: float,
        patience: int,
        threshold: float,
        debounce_time: float,
        n_samples: int,
    ) -> float:
        if score == 0.0:
            return 0.0

        ms = self._model

        if patience > 0:
            if len(ms.prediction_buffer) < patience:
                return 0.0
            recent = np.array(
                list(ms.prediction_buffer)[-(patience - 1) :] + [score]
            )
            if (recent >= threshold).sum() < patience:
                return 0.0

        elif debounce_time > 0:
            frame_dur = n_samples / 16000.0
            if frame_dur <= 0:
                return score
            n_check = int(np.ceil(debounce_time / frame_dur))
            recent = np.array(list(ms.prediction_buffer))[-n_check:]
            if score >= threshold and (recent >= threshold).any():
                return 0.0

        return score
