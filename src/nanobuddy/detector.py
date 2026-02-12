"""WakeDetector — background mic capture + WakeEngine + callbacks."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import numpy as np

from nanobuddy.engine import WakeEngine

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280  # 80 ms


class WakeDetector:
    """Background wake word detection with mic capture.

    Captures audio from the default (or specified) input device, feeds 80 ms
    chunks to a ``WakeEngine``, and fires ``on_detected`` when a wake word
    triggers above threshold with patience confirmation.

    Requires ``sounddevice`` (install with ``pip install nanobuddy[detect]``).

    Example::

        detector = WakeDetector(
            threshold=0.5, patience=4, vad_threshold=0.3,
            cooldown=2.0, on_detected=lambda: print("wake!"),
        )
        detector.start()
        # ... later ...
        detector.stop()
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        threshold: float = 0.5,
        patience: int = 4,
        vad_threshold: float = 0.0,
        cooldown: float = 2.0,
        input_device: int | str | None = None,
        on_detected: Callable | None = None,
        providers: list | None = None,
    ):
        self._model_path = model_path
        self._threshold = threshold
        self._patience = patience
        self._vad_threshold = vad_threshold
        self._cooldown = cooldown
        self._input_device = input_device
        self._on_detected = on_detected
        self._providers = providers

        self._engine: WakeEngine | None = None
        self._stream = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._data_ready = threading.Event()
        self._lock = threading.Lock()
        self._last_trigger = 0.0

        # Ring buffer (4 chunks deep)
        self._buf = np.zeros(CHUNK_SAMPLES * 4, dtype=np.int16)
        self._write_pos = 0
        self._read_pos = 0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Start background detection. Returns True if started."""
        if self.is_running:
            return True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop background detection and release resources."""
        self._stop_event.set()
        self._data_ready.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def pause(self) -> None:
        """Pause detection (stops mic and inference)."""
        self.stop()
        logger.info("WakeDetector paused")

    def resume(self) -> None:
        """Resume detection after pause."""
        if not self.is_running:
            self.start()
            logger.info("WakeDetector resumed")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run(self) -> None:
        import sounddevice as sd

        # Build engine
        self._engine = WakeEngine(
            self._model_path,
            vad_threshold=self._vad_threshold,
            providers=self._providers,
        )

        # Warmup
        self._engine.predict(np.zeros(CHUNK_SAMPLES, dtype=np.int16))
        self._engine.reset()
        logger.info("WakeDetector engine ready")

        # Parse device
        device = self._input_device
        if device is not None:
            try:
                device = int(device)
            except (ValueError, TypeError):
                pass

        # Open mic stream
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=CHUNK_SAMPLES,
                device=device,
                callback=self._audio_callback,
            )
            self._stream.start()
            logger.info("WakeDetector listening on device: %s", device or "default")
        except Exception as e:
            logger.error("Failed to open audio stream: %s", e)
            return

        while not self._stop_event.is_set():
            self._data_ready.wait(timeout=1.0)
            self._data_ready.clear()

            while not self._stop_event.is_set():
                with self._lock:
                    if self._write_pos - self._read_pos < CHUNK_SAMPLES:
                        break
                    start = self._read_pos % len(self._buf)
                    chunk = self._buf[start : start + CHUNK_SAMPLES].copy()
                    self._read_pos += CHUNK_SAMPLES

                score = self._engine.predict(
                    chunk, threshold=self._threshold, patience=self._patience
                )

                if score > 0 and time.monotonic() - self._last_trigger >= self._cooldown:
                    self._last_trigger = time.monotonic()
                    logger.info("Wake word detected (score=%.3f)", score)
                    if self._on_detected:
                        try:
                            self._on_detected()
                        except Exception as e:
                            logger.error("Detection callback error: %s", e)

        # Cleanup
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._engine = None
        logger.info("WakeDetector stopped")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning("Audio status: %s", status)
        audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        with self._lock:
            start = self._write_pos % len(self._buf)
            n = min(len(audio), CHUNK_SAMPLES)
            self._buf[start : start + n] = audio[:n]
            self._write_pos += n
            self._data_ready.set()
