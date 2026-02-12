"""Incremental mel-spectrogram and embedding extraction for streaming inference."""

import numpy as np
from collections import deque

from nanobuddy.models import mel_path, embedding_path
from nanobuddy.onnx_utils import create_session


class FeatureExtractor:
    """Streaming audio feature pipeline: raw PCM → mel-spectrogram → embeddings.

    Designed for 80 ms chunks (1280 samples @ 16 kHz). Maintains rolling buffers
    so callers just feed chunks and read features.
    """

    def __init__(self, *, providers: list | None = None):
        self._mel_session = create_session(mel_path(), providers)
        self._emb_session = create_session(embedding_path(), providers)

        # Rolling buffers
        self.raw_buffer: deque = deque(maxlen=16000 * 10)
        self.mel_buffer = np.ones((76, 32), dtype=np.float32)
        self._mel_max_len = 10 * 97  # ~10 s of mel frames
        self.feature_buffer = self._warm_embedding_buffer()
        self._feature_max_len = 120  # ~10 s of embedding frames

        # Accumulation state
        self._accumulated = 0
        self._remainder = np.empty(0, dtype=np.int16)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, audio: np.ndarray) -> int:
        """Feed a chunk of int16 audio. Returns samples processed (0 if buffering)."""
        return self._streaming_features(audio)

    def get_features(self, n_frames: int = 16) -> np.ndarray:
        """Return the last *n_frames* embedding frames as (1, n_frames, 96) float32."""
        return self.feature_buffer[-n_frames:, :][None, :].astype(np.float32)

    def reset(self):
        """Clear all internal state."""
        self.raw_buffer.clear()
        self.mel_buffer = np.ones((76, 32), dtype=np.float32)
        self._accumulated = 0
        self._remainder = np.empty(0, dtype=np.int16)
        self.feature_buffer = self._warm_embedding_buffer()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _warm_embedding_buffer(self) -> np.ndarray:
        """Produce an initial embedding from random noise so the buffer shape is correct."""
        noise = np.random.randint(-1000, 1000, 16000 * 4, dtype=np.int16)
        return self._embeddings_from_audio(noise)

    def _melspectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel-spectrogram from int16 PCM audio."""
        x = audio.astype(np.float32) if audio.dtype != np.float32 else audio
        x = x[None, :] if x.ndim == 1 else x
        out = self._mel_session.run(None, {"input": x})
        return np.squeeze(out[0]) / 10 + 2  # match TF hub transform

    def _embedding_predict(self, mel_window: np.ndarray) -> np.ndarray:
        """Run embedding model on a (batch, 76, 32, 1) mel window."""
        return self._emb_session.run(None, {"input_1": mel_window})[0].squeeze()

    def _embeddings_from_audio(self, audio: np.ndarray) -> np.ndarray:
        """Full mel → windowed embeddings for a complete audio buffer."""
        spec = self._melspectrogram(audio)
        windows = [spec[i : i + 76] for i in range(0, spec.shape[0], 8) if spec[i : i + 76].shape[0] == 76]
        if not windows:
            return np.empty((0, 96), dtype=np.float32)
        batch = np.expand_dims(np.array(windows), axis=-1).astype(np.float32)
        return self._embedding_predict(batch)

    def _streaming_melspectrogram(self, n_samples: int):
        """Incrementally extend mel buffer from latest raw audio."""
        tail = list(self.raw_buffer)[-(n_samples + 160 * 3) :]
        new_mel = self._melspectrogram(np.array(tail, dtype=np.int16))
        self.mel_buffer = np.vstack((self.mel_buffer, new_mel))
        if self.mel_buffer.shape[0] > self._mel_max_len:
            self.mel_buffer = self.mel_buffer[-self._mel_max_len :]

    def _streaming_features(self, x: np.ndarray) -> int:
        """Core streaming loop: buffer raw audio, emit features every 1280 samples."""
        # Prepend any leftover from the previous call
        if self._remainder.shape[0] > 0:
            x = np.concatenate((self._remainder, x))
            self._remainder = np.empty(0, dtype=np.int16)

        if self._accumulated + x.shape[0] >= 1280:
            remainder = (self._accumulated + x.shape[0]) % 1280
            if remainder:
                self.raw_buffer.extend(x[:-remainder].tolist())
                self._accumulated += x.shape[0] - remainder
                self._remainder = x[-remainder:]
            else:
                self.raw_buffer.extend(x.tolist())
                self._accumulated += x.shape[0]
        else:
            self._accumulated += x.shape[0]
            self.raw_buffer.extend(x.tolist())

        processed = 0
        if self._accumulated >= 1280 and self._accumulated % 1280 == 0:
            self._streaming_melspectrogram(self._accumulated)

            # Extract embeddings for each new 1280-sample chunk
            for i in np.arange(self._accumulated // 1280 - 1, -1, -1):
                ndx = -8 * i
                ndx = ndx if ndx != 0 else len(self.mel_buffer)
                window = self.mel_buffer[-76 + ndx : ndx].astype(np.float32)[None, :, :, None]
                if window.shape[1] == 76:
                    emb = self._embedding_predict(window)
                    self.feature_buffer = np.vstack((self.feature_buffer, emb))

            processed = self._accumulated
            self._accumulated = 0

        # Trim feature buffer
        if self.feature_buffer.shape[0] > self._feature_max_len:
            self.feature_buffer = self.feature_buffer[-self._feature_max_len :]

        return processed if processed else self._accumulated
