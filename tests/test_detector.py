"""Tests for WakeDetector lifecycle (without real audio device)."""

from nanobuddy.detector import WakeDetector


def test_constructor_accepts_all_params():
    detector = WakeDetector(
        model_path="fake.onnx",
        threshold=0.5,
        patience=4,
        vad_threshold=0.3,
        cooldown=2.0,
        input_device=None,
        on_detected=lambda: None,
    )
    assert not detector.is_running
