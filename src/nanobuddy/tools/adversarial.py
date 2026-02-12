"""Phoneme-based adversarial negative generation for wake word training.

Uses the `pronouncing` library (a clean Python wrapper around CMU Pronouncing
Dictionary) to find words and phrases that are phonetically similar to a target
wake word. This is more effective than random negative sampling because it
forces the model to learn fine-grained phonetic distinctions.

Pipeline:
  1. Resolve target word -> phonemes via pronouncing.phones_for_word()
  2. Direct trigram search from target phonemes -> top candidates
  3. Merge with optional hand-crafted seed words
  4. Expand seeds via rhymes + trigram search, score against target
  5. Generate multi-word pairs (first-half x second-half phonemes)
  6. Score-weighted TTS generation (higher phonetic overlap -> more samples)
"""

import logging
import re

import pronouncing

from .config import VOWELS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# -- Phoneme helpers ----------------------------------------------------------

def strip_stress(phoneme: str) -> str:
    """Remove stress markers (0/1/2) from a CMU phoneme."""
    return re.sub(r"\d", "", phoneme)


def get_phonemes(word: str) -> list[str]:
    """Look up phonemes for a word via the pronouncing library."""
    phones = pronouncing.phones_for_word(word)
    if not phones:
        return []
    return [strip_stress(p) for p in phones[0].split()]


def resolve_target_phonemes(target_word: str) -> list[str]:
    """Resolve a target word to its phoneme sequence.

    Falls back to manual lookup for words not in CMU dict (like "clarvis").
    """
    ph = get_phonemes(target_word)
    if ph:
        return ph
    # Common manual overrides for words not in CMU dict
    MANUAL = {
        "clarvis": ["K", "L", "AA", "R", "V", "IH", "S"],
    }
    if target_word.lower() in MANUAL:
        return MANUAL[target_word.lower()]
    raise ValueError(
        f"'{target_word}' not in CMU dict. Pass --phonemes explicitly, e.g.: "
        f"--phonemes K L AA R V IH S"
    )


def phrase_phonemes(phrase: str) -> list[str]:
    """Get combined phonemes for a multi-word phrase."""
    out = []
    for w in phrase.split():
        out.extend(get_phonemes(w))
    return out


def bigrams(ph: list[str]) -> set[tuple[str, str]]:
    """Extract phoneme bigrams from a phoneme list."""
    return {(ph[i], ph[i + 1]) for i in range(len(ph) - 1)}


def _phoneme_edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance on phoneme sequences."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def score_word(phonemes: list[str], target_phonemes_or_bigrams, target_phonemes: list[str] | None = None) -> int:
    """Score phonetic similarity: max_len - edit_distance.

    For backward compat, accepts either:
      score_word(ph, target_bigrams)            -- legacy bigram mode
      score_word(ph, target_phonemes_list)       -- edit distance mode
      score_word(ph, target_bigrams, target_ph)  -- edit distance mode (preferred)
    """
    if not phonemes:
        return 0
    # Determine target phoneme list
    if target_phonemes is not None:
        tgt = target_phonemes
    elif isinstance(target_phonemes_or_bigrams, list):
        tgt = target_phonemes_or_bigrams
    else:
        # Legacy bigram set -- fall back to bigram overlap
        if len(phonemes) < 2:
            return 0
        return len(target_phonemes_or_bigrams & bigrams(phonemes))

    dist = _phoneme_edit_distance(phonemes, tgt)
    return max(0, max(len(phonemes), len(tgt)) - dist)


def is_fragment(phonemes: list[str], target_str: str) -> bool:
    """True if candidate phonemes are a contiguous subsequence of target."""
    return " ".join(phonemes) in target_str


def trigram_regex(phonemes: list[str], start: int) -> str:
    """Build a pronouncing search regex for a 3-phoneme window."""
    return " ".join(
        p + r"[012]?" if p in VOWELS else p
        for p in phonemes[start:start + 3]
    )


def is_valid_word(word: str) -> bool:
    return 4 <= len(word) <= 12 and not word.endswith("'s") and not word.endswith("'")


# -- Pipeline stages ----------------------------------------------------------

def find_direct_matches(target_phonemes: list[str], target_bigrams: set,
                        target_str: str, top_n: int = 40) -> dict[str, int]:
    """Stage 1: Search CMU dict directly from target phoneme trigrams."""
    candidates = {}
    for i in range(len(target_phonemes) - 2):
        regex = trigram_regex(target_phonemes, i)
        for match in pronouncing.search(regex):
            if not is_valid_word(match):
                continue
            mp = get_phonemes(match)
            if not mp or is_fragment(mp, target_str):
                continue
            score = score_word(mp, target_phonemes)
            if score >= 1:
                candidates[match] = max(candidates.get(match, 0), score)

    ranked = sorted(candidates.items(), key=lambda x: -x[1])
    return dict(ranked[:top_n])


def expand_seeds(seeds: list[str], target_phonemes: list[str],
                 target_str: str) -> dict[str, int]:
    """Stage 3: Expand each seed via rhymes + trigram search."""
    candidates = {}
    for seed in seeds:
        phonemes = get_phonemes(seed)
        if not phonemes:
            continue
        for rhyme in pronouncing.rhymes(seed):
            if not is_valid_word(rhyme):
                continue
            rp = get_phonemes(rhyme)
            if rp and not is_fragment(rp, target_str):
                score = score_word(rp, target_phonemes)
                if score >= 1:
                    candidates[rhyme] = max(candidates.get(rhyme, 0), score)
        for i in range(len(phonemes) - 2):
            regex = trigram_regex(phonemes, i)
            for match in pronouncing.search(regex):
                if not is_valid_word(match):
                    continue
                mp = get_phonemes(match)
                if mp and not is_fragment(mp, target_str):
                    score = score_word(mp, target_phonemes)
                    if score >= 1:
                        candidates[match] = max(candidates.get(match, 0), score)
    return candidates


def generate_pairs(target_phonemes: list[str], target_bigrams: set,
                   target_str: str, min_combo_score: int = 3,
                   extra_words: list[str] | None = None) -> list[tuple[str, int, list]]:
    """Stage 4: Generate word pairs whose combined phonemes overlap with target.

    extra_words: additional words (hand seeds, discovered singles) to include
                 as pair components alongside CMU dict search results.
    """
    mid = len(target_phonemes) // 2 + 1
    first_half_bg = bigrams(target_phonemes[:mid])
    second_half_bg = bigrams(target_phonemes[mid - 1:])

    # Build search patterns from target phoneme windows
    def _search_patterns(phonemes: list[str]) -> list[str]:
        pats = []
        for i in range(len(phonemes) - 1):
            pat = " ".join(
                p + "[012]?" if p in VOWELS else p
                for p in phonemes[i:i + 2]
            )
            pats.append(pat)
        return pats

    first_words = {}
    for pat in _search_patterns(target_phonemes[:mid]):
        for w in pronouncing.search(pat):
            ph = get_phonemes(w)
            if not ph or len(w) < 3 or len(w) > 8 or len(ph) > 6:
                continue
            if is_fragment(ph, target_str):
                continue
            score = len(first_half_bg & bigrams(ph))
            if score >= 2:
                first_words[w] = ph

    second_words = {}
    for pat in _search_patterns(target_phonemes[mid - 1:]):
        for w in pronouncing.search(pat):
            ph = get_phonemes(w)
            if not ph or len(w) < 2 or len(w) > 8 or len(ph) > 5:
                continue
            if is_fragment(ph, target_str):
                continue
            score = len(second_half_bg & bigrams(ph))
            if score >= 1:
                second_words[w] = ph

    # Inject extra words (seeds, discovered singles) into both pools
    for w in (extra_words or []):
        if " " in w:
            continue
        ph = get_phonemes(w)
        if not ph or is_fragment(ph, target_str):
            continue
        if len(ph) <= 6 and len(first_half_bg & bigrams(ph)) >= 1:
            first_words.setdefault(w, ph)
        if len(ph) <= 5 and len(second_half_bg & bigrams(ph)) >= 1:
            second_words.setdefault(w, ph)

    pairs = []
    seen = set()
    for w1, ph1 in first_words.items():
        for w2, ph2 in second_words.items():
            combined = ph1 + ph2
            if len(combined) > 10:
                continue
            combo_score = score_word(combined, target_phonemes)
            if combo_score >= min_combo_score:
                phrase = f"{w1} {w2}"
                if phrase not in seen:
                    seen.add(phrase)
                    pairs.append((phrase, combo_score, combined))
    pairs.sort(key=lambda x: -x[1])
    return pairs


# -- Main pipeline ------------------------------------------------------------

# Default hand-crafted seeds for "clarvis" -- override with --seeds
CLARVIS_SEEDS = [
    "jarvis", "clovis", "travis", "davis", "elvis", "mavis",
    "harvest", "carving", "starving", "marvel",
    "service", "nervous", "serving", "swerving", "cervix", "novice",
    "clarify", "clarity", "clever", "clover", "artist",
]
CLARVIS_PHRASES = [
    "carve this", "clark is", "carved his",
    "carve us", "carve it", "starve us",
]


def run_pipeline(target_word: str, phonemes: list[str] | None = None,
                 hand_seeds: list[str] | None = None,
                 hand_phrases: list[str] | None = None,
                 min_single_score: int = 1, min_pair_score: int = 5,
                 max_pairs: int = 500, max_total: int = 0,
                 max_per_score: dict[int, int] | None = None) -> list[str]:
    """Run the full adversarial word discovery pipeline.

    Args:
        target_word: The wake word to generate adversarial samples for.
        phonemes: Explicit phoneme list. Auto-resolved if None.
        hand_seeds: Optional hand-crafted seed words. Uses built-in seeds
                    for "clarvis" if None.
        hand_phrases: Optional hand-crafted phrases.
        min_single_score: Minimum bigram overlap for single words.
        min_pair_score: Minimum bigram overlap for word pairs.
        max_pairs: Maximum number of word pairs to include.
        max_total: Maximum total words/phrases in output.

    Returns:
        List of adversarial words and phrases.
    """
    target_ph = phonemes or resolve_target_phonemes(target_word)
    target_bg = bigrams(target_ph)
    target_str = " ".join(target_ph)

    if hand_seeds is None and target_word.lower() == "clarvis":
        hand_seeds = CLARVIS_SEEDS
    if hand_phrases is None and target_word.lower() == "clarvis":
        hand_phrases = CLARVIS_PHRASES
    hand_seeds = hand_seeds or []
    hand_phrases = hand_phrases or []

    # Stage 1: Direct search
    print(f"Stage 1: Direct search from {' '.join(target_ph)}...")
    s1 = find_direct_matches(target_ph, target_bg, target_str, top_n=40)
    print(f"  {len(s1)} candidates\n")

    # Stage 2: Merge seeds
    merged_seeds = list(set(hand_seeds + list(s1.keys())))
    print(f"Stage 2: Merged {len(hand_seeds)} hand seeds + {len(s1)} stage1"
          f" -> {len(merged_seeds)} seeds\n")

    # Stage 3: Expand
    print("Stage 3: Expanding from all seeds...")
    expanded = expand_seeds(merged_seeds, target_ph, target_str)
    seed_set = {s.lower() for s in merged_seeds}
    expanded = {w: s for w, s in expanded.items() if w.lower() not in seed_set}
    expanded = dict(sorted(expanded.items(), key=lambda x: -x[1]))
    above = {w: s for w, s in expanded.items() if s >= min_single_score}
    print(f"  {len(expanded)} total, {len(above)} at score >= {min_single_score}\n")

    # Stage 4: Pairs -- feed all discovered singles + seeds as pair components
    all_singles = list(hand_seeds) + list(s1.keys()) + list(above.keys())
    print(f"Stage 4: Generating multi-word pairs (score >= {min_pair_score})...")
    print(f"  Injecting {len(all_singles)} discovered words as pair components")
    pairs = generate_pairs(target_ph, target_bg, target_str,
                           min_combo_score=min_pair_score,
                           extra_words=all_singles)
    print(f"  {len(pairs)} pairs found, keeping top {max_pairs}\n")

    # Build final list
    final = list(hand_phrases)
    seed_scored = [(s, score_word(get_phonemes(s), target_ph))
                   for s in hand_seeds if get_phonemes(s)]
    seed_scored.sort(key=lambda x: -x[1])
    for word, _ in seed_scored:
        if word not in final:
            final.append(word)
    for word in s1:
        if word not in final:
            final.append(word)
    for word in above:
        if word not in final:
            final.append(word)
    final_set = set(final)
    for phrase, score, _ in pairs[:max_pairs]:
        if phrase not in final_set:
            final.append(phrase)
            final_set.add(phrase)
    # Apply per-score caps with random sampling
    if max_per_score:
        import random as _rng
        _rng.seed(42)
        # Group by score
        buckets = {}
        for w in final:
            ph = phrase_phonemes(w) if " " in w else get_phonemes(w)
            s = score_word(ph, target_ph) if ph else 0
            buckets.setdefault(s, []).append(w)
        # Random sample capped buckets, keep uncapped as-is
        capped = []
        for s in sorted(buckets.keys(), reverse=True):
            words_in_bucket = buckets[s]
            cap = max_per_score.get(s)
            if cap is not None and len(words_in_bucket) > cap:
                words_in_bucket = _rng.sample(words_in_bucket, cap)
            capped.extend(words_in_bucket)
        final = capped

    if max_total and len(final) > max_total:
        final = final[:max_total]
    return final


def print_list(words: list[str], target_phonemes: list[str]):
    """Pretty-print the adversarial word list with scores."""
    singles = [w for w in words if " " not in w]
    phrases = [w for w in words if " " in w]

    print(f"\n{'='*60}")
    print(f"Final list: {len(words)} adversarial words/phrases")
    print(f"{'='*60}\n")

    print(f"-- Single words ({len(singles)}) --\n")
    for i, w in enumerate(singles):
        ph = get_phonemes(w)
        score = score_word(ph, target_phonemes) if ph else "?"
        print(f"  {i+1:3d}. {w:17s} score={score}  {' '.join(ph)}")

    print(f"\n-- Phrases ({len(phrases)}) --\n")
    for i, p in enumerate(phrases):
        ph = phrase_phonemes(p)
        score = score_word(ph, target_phonemes) if ph else "?"
        print(f"  {i+1:3d}. {p:25s} score={score}  {' '.join(ph)}")


# Score-weighted TTS: higher-scoring words get more voice variants
# Edit distance scoring: max ~7 for clarvis (7 phonemes)
SAMPLES_BY_SCORE = {7: 8, 6: 8, 5: 8, 4: 4, 3: 1, 2: 1, 1: 1, 0: 8}

# Top confusable prefixes for cartesian adversarial (exact + similar-sounding)
CARTESIAN_PREFIXES = ["hey", "hi", "ok", "yo", "sup", "oh", "uh", "aye", "go", "high"]

ALL_ENGINES = ["piper", "kokoro", "coqui"]


def generate_adversarial(target_word: str, output_dir: str,
                         engine: str = "piper",
                         phonemes: list[str] | None = None,
                         min_single_score: int = 1,
                         max_per_score: dict[int, int] | None = None,
                         cartesian: bool = False,
                         cartesian_prefixes: list[str] | None = None,
                         cartesian_top_n: int = 65):
    """Generate all adversarial negatives -- bare (score-weighted) + optional cartesian.

    Bare: all discovered words x SAMPLES_BY_SCORE multiplier.
    Cartesian: cartesian_prefixes x top_n highest-scoring words x 1 clip each.
    """
    if engine == "all":
        for eng in ALL_ENGINES:
            generate_adversarial(
                target_word, output_dir, engine=eng,
                phonemes=phonemes, min_single_score=min_single_score,
                max_per_score=max_per_score, cartesian=cartesian,
                cartesian_prefixes=cartesian_prefixes,
                cartesian_top_n=cartesian_top_n,
            )
        return

    from .generate import _get_generate_fn
    generate_samples = _get_generate_fn(engine)
    if cartesian_prefixes is None:
        cartesian_prefixes = CARTESIAN_PREFIXES

    # Run the discovery pipeline
    words = run_pipeline(
        target_word, phonemes=phonemes,
        min_single_score=min_single_score, max_total=0,
        max_per_score=max_per_score,
    )
    target_ph = phonemes or resolve_target_phonemes(target_word)

    # 1. Bare adversarial: all words x score-weighted multiplier
    text_list = []
    for w in words:
        ph = phrase_phonemes(w) if " " in w else get_phonemes(w)
        score = score_word(ph, target_ph) if ph else 0
        reps = SAMPLES_BY_SCORE.get(min(score, 4), 1)
        text_list.extend([w] * reps)

    bare_count = len(text_list)

    # 2. Cartesian: prefixes x top-N words x 1 clip
    cart_count = 0
    if cartesian:
        scored = []
        for w in words:
            ph = phrase_phonemes(w) if " " in w else get_phonemes(w)
            s = score_word(ph, target_ph) if ph else 0
            scored.append((w, s))
        scored.sort(key=lambda x: -x[1])
        top_words = [w for w, _ in scored[:cartesian_top_n]]

        for prefix in cartesian_prefixes:
            for word in top_words:
                text_list.append(f"{prefix} {word}")
        cart_count = len(cartesian_prefixes) * len(top_words)

    total = len(text_list)
    print(f"\n[{engine}] Generating {total} adversarial clips -> {output_dir}")
    print(f"  bare: {bare_count} ({len(words)} words, score-weighted)")
    if cartesian:
        print(f"  cartesian: {cart_count} ({len(cartesian_prefixes)} prefixes"
              f" x {min(cartesian_top_n, len(words))} words x 1)")

    generate_samples(
        text=text_list,
        output_dir=output_dir,
        max_samples=total,
        file_prefix="adv",
    )
    print(f"\n[{engine}] Done. {total} adversarial clips in {output_dir}")


def run_adversarial(target: str, output_dir: str = "./negative",
                    phonemes: list[str] | None = None,
                    min_score: int = 1, max_words: int = 0,
                    dry_run: bool = False, engine: str = "piper",
                    cartesian: bool = True):
    """CLI entry point for 'nanobuddy adversarial'."""
    max_per_score = {1: 1000, 2: 3000, 3: 2000}

    if dry_run:
        words = run_pipeline(
            target_word=target,
            phonemes=phonemes,
            min_single_score=min_score,
            max_total=max_words,
            max_per_score=max_per_score,
        )
        target_ph = phonemes or resolve_target_phonemes(target)
        print_list(words, target_ph)
        bare_total = sum(
            SAMPLES_BY_SCORE.get(min(score_word(
                phrase_phonemes(w) if " " in w else get_phonemes(w),
                target_ph), 7), 1)
            for w in words
        )
        cart_total = len(CARTESIAN_PREFIXES) * min(65, len(words)) if cartesian else 0
        print(f"\nDry run: would generate {bare_total + cart_total} clips"
              f" (bare: {bare_total}, cartesian: {cart_total})")
    else:
        generate_adversarial(
            target_word=target,
            output_dir=output_dir,
            engine=engine,
            phonemes=phonemes,
            min_single_score=min_score,
            max_per_score=max_per_score,
            cartesian=cartesian,
        )
