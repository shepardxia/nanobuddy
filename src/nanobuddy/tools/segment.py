"""Whisper-based word-level audio segmentation for negative sample extraction.

Takes raw audio recordings (any language), transcribes with faster-whisper using
word-level timestamps, and extracts individual word segments as WAV clips.

Key features:
  - Dual filtering: text substrings AND phoneme fragments of the target word
  - Configurable padding, minimum/maximum clip duration, confidence threshold
  - Automatic resampling to 16kHz mono
  - Resumable (continues numbering from existing files)
  - Transcript TSV log for review
"""

import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

from .config import SAMPLE_RATE, DEFAULT_SKIP_WORDS

try:
    import pronouncing
    HAS_PRONOUNCING = True
except ImportError:
    HAS_PRONOUNCING = False

PAD_MS = 80
MIN_PROBABILITY = 0.4


def _strip_stress(phoneme: str) -> str:
    return re.sub(r"\d", "", phoneme)


def _build_text_fragments(target_word: str, min_len: int = 3) -> set[str]:
    """Build set of text substrings of target (len >= min_len)."""
    frags = set()
    for i in range(len(target_word)):
        for j in range(i + min_len, len(target_word) + 1):
            frags.add(target_word[i:j])
    frags.discard(target_word)  # full word caught separately
    return frags


def is_target_fragment(word_text: str, target_word: str,
                       target_phonemes: list[str] | None,
                       text_fragments: set[str]) -> bool:
    """Check if a word is a phonetic or textual fragment of the target."""
    text = word_text.strip().lower()
    if text in text_fragments:
        return True
    if HAS_PRONOUNCING and target_phonemes:
        target_str = " ".join(target_phonemes)
        phones = pronouncing.phones_for_word(text)
        if phones:
            stripped = [_strip_stress(p) for p in phones[0].split()]
            if " ".join(stripped) in target_str:
                return True
    return False


def _save_group(group, audio, sr, pad_samples, total_samples,
                output_dir, prefix, idx, lang, max_samples=0):
    """Save a group of consecutive words as a single clip, hard-trimmed."""
    start_sample = max(0, int(group[0].start * sr) - pad_samples)
    end_sample = min(total_samples, int(group[-1].end * sr) + pad_samples)

    clip = audio[start_sample:end_sample]
    if max_samples > 0 and len(clip) > max_samples:
        clip = clip[:max_samples]
    clip_int16 = np.clip(clip * 32767, -32768, 32767).astype(np.int16)

    idx += 1
    fname = f"{prefix}_{idx:04d}.wav"
    sf.write(str(output_dir / fname), clip_int16, sr, subtype="PCM_16")

    words_text = " ".join(w.word.strip() for w in group)
    entry = {
        "idx": idx, "word": words_text, "lang": lang,
        "prob": f"{min(w.probability for w in group):.2f}",
        "start": f"{group[0].start:.3f}",
        "end": f"{group[-1].end:.3f}",
        "duration_ms": int((group[-1].end - group[0].start) * 1000),
        "clip_ms": int(len(clip) / sr * 1000),
        "file": fname,
        "source": "",
    }
    return idx, entry


def segment_file(
    audio_path: Path,
    model: WhisperModel,
    output_dir: Path,
    prefix: str,
    start_idx: int,
    target_word: str = "clarvis",
    target_phonemes: list[str] | None = None,
    skip_words: set[str] | None = None,
    min_clip_s: float = 0.5,
    max_clip_s: float = 2.0,
) -> tuple[int, list[dict]]:
    """Segment one audio file into phrase-level clips.

    Groups consecutive safe words until the span reaches min_clip_s,
    then cuts. Flushes the group if a skip/target word is encountered.

    Returns (next_idx, list of entry dicts).
    """
    if skip_words is None:
        skip_words = DEFAULT_SKIP_WORDS
    text_fragments = _build_text_fragments(target_word)

    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import soxr
        audio = soxr.resample(audio, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE

    segments, info = model.transcribe(audio, word_timestamps=True, vad_filter=False)
    detected_lang = info.language

    pad_samples = int(PAD_MS * sr / 1000)
    total_samples = len(audio)
    idx = start_idx
    log_entries = []

    # Collect all valid words, marking unsafe ones
    all_words = []
    for segment in segments:
        if segment.words is None:
            continue
        for word_info in segment.words:
            text = word_info.word.strip().lower()
            if word_info.probability < MIN_PROBABILITY:
                continue
            if not any(c.isalnum() for c in text):
                continue

            is_unsafe = (
                any(skip in text for skip in skip_words)
                or is_target_fragment(text, target_word, target_phonemes, text_fragments)
            )
            all_words.append((word_info, is_unsafe))

    # Group consecutive safe words into clips >= min_clip_s
    # Break on: unsafe words, large gaps (>0.5s), or max_clip_s reached
    max_gap_s = 0.5
    group = []

    max_clip_samples = int(max_clip_s * sr)

    def _flush():
        nonlocal idx
        if group and (group[-1].end - group[0].start) >= min_clip_s:
            idx, entry = _save_group(
                group, audio, sr, pad_samples, total_samples,
                output_dir, prefix, idx, detected_lang,
                max_samples=max_clip_samples)
            entry["source"] = audio_path.name
            log_entries.append(entry)

    for word_info, is_unsafe in all_words:
        if is_unsafe:
            _flush()
            group = []
            continue

        # Break on large gap between words
        if group and (word_info.start - group[-1].end) > max_gap_s:
            _flush()
            group = []

        group.append(word_info)
        if (group[-1].end - group[0].start) >= max_clip_s:
            _flush()
            group = []

    _flush()

    return idx, log_entries


def run_segment(input_path: str, output_dir: str = "./negative_real",
                target: str = "clarvis", prefix: str = "real",
                whisper_model: str = "base", skip_words: str | None = None):
    """CLI entry point for 'nanobuddy segment'."""
    from .adversarial import resolve_target_phonemes

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        audio_files = [input_path]
    elif input_path.is_dir():
        audio_files = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
        )
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)

    if not audio_files:
        print(f"No audio files found in {input_path}")
        sys.exit(1)

    target_phonemes = None
    try:
        target_phonemes = resolve_target_phonemes(target)
    except ValueError:
        print(f"Warning: '{target}' not in CMU dict, skipping phoneme filtering")

    skip_words_set = set(skip_words.split(",")) if skip_words else DEFAULT_SKIP_WORDS

    print(f"Loading Whisper model '{whisper_model}'...")
    model = WhisperModel(whisper_model, device="cpu", compute_type="int8")

    existing = list(output_dir.glob(f"{prefix}_*.wav"))
    start_idx = len(existing)
    if start_idx > 0:
        print(f"Continuing from {start_idx} existing {prefix}_* files")

    all_entries = []
    idx = start_idx
    for audio_path in audio_files:
        print(f"\nProcessing: {audio_path.name}")
        idx, entries = segment_file(
            audio_path, model, output_dir, prefix, idx,
            target_word=target, target_phonemes=target_phonemes,
            skip_words=skip_words_set,
        )
        all_entries.extend(entries)
        for e in entries:
            print(f"  {e['file']:20s}  {e['clip_ms']:4d}ms  p={e['prob']}  [{e['lang']}]  \"{e['word']}\"")

    log_path = output_dir / f"{prefix}_transcript.tsv"
    with open(log_path, "w") as f:
        f.write("idx\tfile\tword\tlang\tprob\tstart\tend\tword_ms\tclip_ms\tsource\n")
        for e in all_entries:
            f.write(f"{e['idx']}\t{e['file']}\t{e['word']}\t{e['lang']}\t{e['prob']}\t"
                    f"{e['start']}\t{e['end']}\t{e['duration_ms']}\t{e['clip_ms']}\t{e['source']}\n")

    print(f"\nDone: {len(all_entries)} word clips saved to {output_dir}")
    print(f"Transcript log: {log_path}")
