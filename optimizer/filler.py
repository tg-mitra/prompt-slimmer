"""Removes polite filler phrases that add tokens without adding meaning.

Phrases are grouped into tiers by how safe they are to strip. Tier 1 phrases
("could you please") are essentially always filler; higher tiers are softer
or more context-dependent, and only get removed as the configured
optimization level increases.
"""

from __future__ import annotations

import re

# (min_level_to_apply, phrase_pattern)
_FILLER_TIERS: list[tuple[int, str]] = [
    (1, r"\bcould you please\b"),
    (1, r"\bwould you please\b"),
    (1, r"\bcan you please\b"),
    (1, r"\bi would like you to\b"),
    (1, r"\bi'd like you to\b"),
    (1, r"\bit would be great if you (?:could|would)\b"),
    (1, r"\bkindly\b"),
    (2, r"\bplease\b"),
    (2, r"\bi was wondering if you could\b"),
    (2, r"\bif (?:it'?s )?possible,?\b"),
    (3, r"\bjust to let you know,?\b"),
    (3, r"\bplease note that\b"),
    (3, r"\bi think (?:that )?\b"),
    (4, r"\bbasically,?\b"),
    (4, r"\bactually,?\b"),
    (4, r"\bin order to\b"),
    (5, r"\bfeel free to\b"),
    (5, r"\bi just wanted to\b"),
    (6, r"\bhonestly,?\b"),
    (6, r"\bto be honest,?\b"),
    (7, r"\bthank you(?: so much| very much)?(?: in advance)?[,.]?\b"),
    (7, r"\bthanks(?: so much| in advance)?[,.]?\b"),
]

_WHITESPACE_RE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.!?])")
_SENTENCE_START_RE = re.compile(r"(^\s*|[.!?]\s+)([a-z])")


def remove_filler_phrases(text: str, level: int) -> str:
    if level <= 0:
        return text

    for min_level, pattern in _FILLER_TIERS:
        if level >= min_level:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _recapitalize_sentence_starts(text)
    return text.strip()


def _recapitalize_sentence_starts(text: str) -> str:
    """Removing a leading filler often leaves a lowercase word at the start
    of a sentence; capitalize it back."""
    return "\n".join(
        _SENTENCE_START_RE.sub(lambda m: m.group(1) + m.group(2).upper(), line)
        for line in text.split("\n")
    )
