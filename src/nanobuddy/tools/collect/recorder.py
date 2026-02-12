"""Core recording engine with Space-toggle marking."""

import sys
import select
import termios
import threading
import tty
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from .config import CollectConfig
from .segmenter import HoldWindow


@dataclass
class RecordingSession:
    """Result of a recording session."""
    audio: np.ndarray
    holds: list[HoldWindow]
    duration_s: float
    sample_rate: int


class KeyReader:
    """Non-blocking keyboard reader using raw terminal mode."""

    def __init__(self):
        self._fd = sys.stdin.fileno()
        self._old_settings = None

    def __enter__(self):
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setraw(self._fd)
        return self

    def __exit__(self, *args):
        if self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def read_key(self) -> str | None:
        """Non-blocking key read. Returns key char or None."""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


class Recorder:
    """Continuous audio recorder with Space-toggle marking.

    Audio streams continuously. Press Space to start marking a positive
    sample, press Space again to stop. Press Q to stop recording.
    Press R to undo the last mark (or cancel current mark).

    The full recording is saved — positives are extracted with keypress
    trimming, and everything else is saved as one raw negative file.
    """

    def __init__(
        self,
        cfg: CollectConfig,
        on_hold_start: Callable | None = None,
        on_hold_end: Callable[[HoldWindow, np.ndarray], None] | None = None,
        on_undo: Callable | None = None,
        on_tick: Callable[[float], None] | None = None,
        on_auto_flash: Callable[[str | None], None] | None = None,
    ):
        self.cfg = cfg
        self._on_hold_start = on_hold_start
        self._on_hold_end = on_hold_end
        self._on_undo = on_undo
        self._on_tick = on_tick
        self._on_auto_flash = on_auto_flash

        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._holds: list[HoldWindow] = []
        self._holding = False
        self._hold_kind: str = "pos"  # "pos" or "neg"
        self._hold_start_sample = 0
        self._total_samples = 0
        self._running = False

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback — runs in audio thread."""
        self._chunks.append(indata[:, 0].copy())
        with self._lock:
            self._total_samples += frames

    def _get_total_samples(self) -> int:
        with self._lock:
            return self._total_samples

    def record(self) -> RecordingSession:
        """Run the recording loop. Blocks until user presses Q."""
        sr = self.cfg.sample_rate
        min_hold_samples = int(self.cfg.min_hold_ms * sr / 1000)

        stream = sd.InputStream(
            samplerate=sr,
            channels=self.cfg.channels,
            dtype=self.cfg.dtype,
            blocksize=int(sr * 0.05),  # 50ms blocks
            device=self.cfg.input_device,
            callback=self._audio_callback,
        )

        self._running = True
        start_time = time.monotonic()

        with stream, KeyReader() as keys:
            if self.cfg.auto_capture:
                self._run_auto(keys, start_time, min_hold_samples)
            else:
                while self._running:
                    key = keys.read_key()
                    if key is not None:
                        self._process_key(key, min_hold_samples)

                    if self._on_tick:
                        self._on_tick(time.monotonic() - start_time)

                    time.sleep(0.02)  # 50Hz poll rate

        audio = np.concatenate(self._chunks) if self._chunks else np.array([], dtype=np.int16)

        return RecordingSession(
            audio=audio,
            holds=list(self._holds),
            duration_s=len(audio) / sr,
            sample_rate=sr,
        )

    def _process_key(self, key: str, min_hold_samples: int):
        """Handle a keypress."""
        if key in ("q", "Q", "\x03"):  # Q or Ctrl+C
            if self._holding:
                self._complete_hold(min_hold_samples)
            self._running = False

        elif key == " " or key in ("n", "N"):
            kind = "pos" if key == " " else "neg"
            if not self._holding:
                self._holding = True
                self._hold_kind = kind
                self._hold_start_sample = self._get_total_samples()
                if self._on_hold_start:
                    self._on_hold_start(kind)
            else:
                self._complete_hold(min_hold_samples)

        elif key in ("r", "R"):
            if self._holding:
                self._holding = False
                if self._on_undo:
                    self._on_undo()
            elif self._holds:
                self._holds.pop()
                if self._on_undo:
                    self._on_undo()

    def _complete_hold(self, min_hold_samples: int):
        """Complete a hold window."""
        self._holding = False
        current_samples = self._get_total_samples()
        hold_samples = current_samples - self._hold_start_sample

        if hold_samples < min_hold_samples:
            return  # too short, accidental tap

        hold = HoldWindow(
            start_sample=self._hold_start_sample,
            end_sample=current_samples,
            kind=self._hold_kind,
        )
        self._holds.append(hold)

        if self._on_hold_end:
            audio = np.concatenate(self._chunks) if self._chunks else np.array([], dtype=np.int16)
            pre_pad = int(self.cfg.pre_pad_ms * self.cfg.sample_rate / 1000)
            post_pad = int(self.cfg.post_pad_ms * self.cfg.sample_rate / 1000)
            clip_start = max(0, hold.start_sample - pre_pad)
            clip_end = min(len(audio), hold.end_sample + post_pad)
            self._on_hold_end(hold, audio[clip_start:clip_end])

    def _run_auto(self, keys, start_time: float, min_hold_samples: int):
        """Auto-capture loop: wait → flash GO → capture → flash STOP → repeat."""
        interval = self.cfg.auto_interval_s
        duration = self.cfg.auto_duration_s

        # Initial wait for user to get into position
        self._wait_with_keys(keys, start_time, interval)

        while self._running:
            # GO — start capturing
            if self._on_auto_flash:
                self._on_auto_flash("go")
            self._holding = True
            self._hold_start_sample = self._get_total_samples()
            if self._on_hold_start:
                self._on_hold_start()

            # Capture for duration
            self._wait_with_keys(keys, start_time, duration)
            if not self._running:
                break

            # Complete the hold
            self._complete_hold(min_hold_samples)

            # Check if target reached
            if len(self._holds) >= self.cfg.target_positive:
                if self._on_auto_flash:
                    self._on_auto_flash(None)
                self._running = False
                break

            # STOP flash
            if self._on_auto_flash:
                self._on_auto_flash("stop")
            self._wait_with_keys(keys, start_time, 0.5)
            if not self._running:
                break

            # Clear flash and wait for next capture
            if self._on_auto_flash:
                self._on_auto_flash(None)
            self._wait_with_keys(keys, start_time, max(0, interval - 0.5))

    def _wait_with_keys(self, keys, start_time: float, wait_seconds: float):
        """Wait for given duration while checking for quit key and ticking."""
        end = time.monotonic() + wait_seconds
        while time.monotonic() < end and self._running:
            key = keys.read_key()
            if key in ("q", "Q", "\x03"):
                if self._holding:
                    self._complete_hold(0)
                self._running = False
                return
            if self._on_tick:
                self._on_tick(time.monotonic() - start_time)
            time.sleep(0.02)

    @property
    def hold_count(self) -> int:
        return len(self._holds)

    @property
    def is_holding(self) -> bool:
        return self._holding
