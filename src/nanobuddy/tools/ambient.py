"""Extract ambient noise from recordings using VAD + RMS filtering.

Scans WAV files frame-by-frame with Silero VAD (ONNX). Contiguous regions
where speech probability is low AND volume is above a floor are extracted
as ambient noise clips, suitable for use as background noise in training.

Uses onnxruntime directly -- no hey-buddy import required.
"""

import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
FRAME_MS = 30  # Silero VAD frame size

SILERO_VAD_URL = "https://huggingface.co/benjamin-paine/hey-buddy/resolve/main/pretrained/silero-vad.onnx"


class SileroVAD:
    """Minimal Silero VAD wrapper using onnxruntime directly."""

    def __init__(self, model_path: Path):
        from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel

        opts = SessionOptions()
        opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1

        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        self.session = InferenceSession(str(model_path), providers=providers, sess_options=opts)
        self.reset()

    def reset(self):
        """Reset hidden state between files."""
        self.h = np.zeros((2, 1, 64), dtype=np.float32)
        self.c = np.zeros((2, 1, 64), dtype=np.float32)

    def __call__(self, audio: np.ndarray, sample_rate: int = 16000) -> float:
        """Return speech probability for a single frame."""
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        out, self.h, self.c = self.session.run(None, {
            "input": audio.astype(np.float32),
            "h": self.h,
            "c": self.c,
            "sr": np.array([sample_rate], dtype=np.int64),
        })
        return float(out[0][0])


def _get_vad_model() -> SileroVAD:
    """Download Silero VAD ONNX model if needed, return loaded model."""
    cache_dir = Path.home() / ".cache" / "nanobuddy"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "silero-vad.onnx"

    if not model_path.exists():
        print(f"Downloading Silero VAD model to {model_path}...")
        urlretrieve(SILERO_VAD_URL, str(model_path))
        print("Done.")

    return SileroVAD(model_path)


def extract_ambient(
    input_dir: Path,
    output_dir: Path,
    vad_threshold: float = 0.3,
    rms_threshold_db: float = -40.0,
    min_segment_ms: int = 500,
    max_segment_ms: int = 5000,
    max_gap_frames: int = 5,
):
    """Extract non-speech ambient noise segments from WAV files.

    For each WAV, runs Silero VAD frame-by-frame. Regions where most
    frames have vad_prob <= vad_threshold AND rms_db >= rms_threshold_db
    are extracted as ambient noise clips. Brief speech bursts up to
    max_gap_frames (default 5 = 150ms) are tolerated within a segment
    rather than splitting it.
    """
    wavs = sorted(input_dir.glob("*.wav"))
    if not wavs:
        print(f"No .wav files found in {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input: {len(wavs)} files from {input_dir}")
    print(f"VAD threshold: <= {vad_threshold}")
    print(f"RMS threshold: >= {rms_threshold_db} dBFS")
    print(f"Segment length: {min_segment_ms}-{max_segment_ms}ms")
    print(f"Gap tolerance: {max_gap_frames} frames ({max_gap_frames * FRAME_MS}ms)")
    print(f"Output: {output_dir}")
    print()

    vad = _get_vad_model()
    frame_size = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples

    total_segments = 0
    total_duration_s = 0.0

    for wav_path in wavs:
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]

        # Resample if needed
        if sr != SAMPLE_RATE:
            import soxr
            audio = soxr.resample(audio, sr, SAMPLE_RATE)
            sr = SAMPLE_RATE

        # Reset VAD hidden state between files
        vad.reset()

        # Classify each frame: is it ambient (non-speech + audible)?
        ambient_mask = []
        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i : i + frame_size]
            vad_prob = vad(frame, sr)
            rms = float(np.sqrt(np.mean(frame ** 2)))
            rms_db = 20.0 * np.log10(max(rms, 1e-10))

            is_ambient = vad_prob <= vad_threshold and rms_db >= rms_threshold_db
            ambient_mask.append((i, is_ambient))

        # Find ambient regions with gap tolerance
        segments = []
        seg_start = None
        gap_count = 0  # consecutive non-ambient frames
        seg_end = None
        for sample_idx, is_ambient in ambient_mask:
            if is_ambient:
                if seg_start is None:
                    seg_start = sample_idx
                gap_count = 0
                seg_end = sample_idx + frame_size
            elif seg_start is not None:
                gap_count += 1
                if gap_count > max_gap_frames:
                    # Gap too long -- close this segment
                    dur_ms = (seg_end - seg_start) / sr * 1000
                    if min_segment_ms <= dur_ms <= max_segment_ms:
                        segments.append((seg_start, seg_end))
                    seg_start = None
                    gap_count = 0

        # Handle trailing segment
        if seg_start is not None and seg_end is not None:
            dur_ms = (seg_end - seg_start) / sr * 1000
            if min_segment_ms <= dur_ms <= max_segment_ms:
                segments.append((seg_start, seg_end))

        # Save segments
        for j, (s, e) in enumerate(segments):
            clip = audio[s:e]
            clip_int16 = np.clip(clip * 32768.0, -32768, 32767).astype(np.int16)
            out_name = f"ambient_{wav_path.stem}_{j:03d}.wav"
            sf.write(str(output_dir / out_name), clip_int16, sr)
            total_duration_s += len(clip) / sr

        if segments:
            print(f"  {wav_path.name}: {len(segments)} segments extracted")

        total_segments += len(segments)

    dur_m, dur_s = divmod(int(total_duration_s), 60)
    print(f"\nDone. {total_segments} ambient clips ({dur_m}m {dur_s:02d}s) saved to {output_dir}")


def run_ambient(input_dir: str, output_dir: str,
                vad_threshold: float = 0.3, rms_threshold: float = -40.0,
                min_segment: int = 500, max_segment: int = 5000):
    """CLI entry point for ambient subcommand."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    extract_ambient(
        input_dir,
        output_dir,
        vad_threshold=vad_threshold,
        rms_threshold_db=rms_threshold,
        min_segment_ms=min_segment,
        max_segment_ms=max_segment,
    )
