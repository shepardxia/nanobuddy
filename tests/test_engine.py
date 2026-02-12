"""Tests for WakeEngine — streaming inference, VAD gating, patience."""

import numpy as np
import pytest

from nanobuddy import WakeEngine

CHUNK = 1280  # 80 ms @ 16 kHz


@pytest.fixture
def engine():
    return WakeEngine()


@pytest.fixture
def engine_with_vad():
    return WakeEngine(vad_threshold=0.3)


class TestWakeEngine:
    def test_silence_returns_zero(self, engine):
        chunk = np.zeros(CHUNK, dtype=np.int16)
        for _ in range(10):
            score = engine.predict(chunk)
        assert score == pytest.approx(0.0, abs=0.05)

    def test_reset_clears_state(self, engine):
        engine.predict(np.random.randint(-1000, 1000, CHUNK, dtype=np.int16))
        engine.reset()
        ms = engine._model
        assert len(ms.prediction_buffer) == 0
        assert ms.raw_score == 0.0
        assert ms.hidden is None

    def test_vad_gates_silence(self, engine_with_vad):
        chunk = np.zeros(CHUNK, dtype=np.int16)
        for _ in range(10):
            score = engine_with_vad.predict(chunk)
        assert score == 0.0

    def test_warmup_suppression(self, engine):
        chunk = np.random.randint(-5000, 5000, CHUNK, dtype=np.int16)
        for _ in range(5):
            score = engine.predict(chunk)
        assert score == 0.0

    def test_patience_blocks_isolated_frames(self, engine):
        chunk = np.zeros(CHUNK, dtype=np.int16)
        for _ in range(10):
            score = engine.predict(chunk, patience=10, threshold=0.5)
        assert score == 0.0

    def test_raw_scores_in_prediction_buffer(self, engine):
        """The upstream patience bug stored filtered zeros instead of raw scores."""
        chunk = np.random.randint(-3000, 3000, CHUNK, dtype=np.int16)
        for _ in range(10):
            engine.predict(chunk, patience=100, threshold=0.5)

        buffer_vals = list(engine._model.prediction_buffer)
        post_warmup = buffer_vals[5:]
        assert any(v != 0.0 for v in post_warmup), (
            "prediction_buffer should store raw scores, not patience-filtered zeros"
        )
