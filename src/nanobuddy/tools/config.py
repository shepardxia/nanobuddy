"""Shared constants and utilities."""

SAMPLE_RATE = 16000

VOWELS = frozenset(
    "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split()
)

# Words that should never appear in negative samples
DEFAULT_SKIP_WORDS = frozenset({"clarvis", "clarvus", "klarvis"})
