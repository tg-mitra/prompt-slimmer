"""Detects and merges sentences that express the same instruction/constraint
in different words, e.g. "be concise", "keep the answer short", and "avoid
long explanations" all collapse down to a single kept instruction.

Generic sentence embeddings turn out to be unreliable for this: short
imperative paraphrases don't share enough vocabulary to score reliably
higher than unrelated short directives (empirically, "avoid long
explanations" vs. "do not reveal internal reasoning" can score *higher*
cosine similarity than true paraphrases like "avoid long explanations" vs.
"keep the answer short"). So the most common recurring constraint types are
recognized directly via keyword categories first; embeddings are only used
as a conservative fallback for directives that don't match a known category.
"""

from __future__ import annotations

import re

from .clustering import find_near_duplicates
from .embeddings import EmbeddingBackend
from .text_utils import SentenceDoc, is_directive_sentence

# Any two directive sentences matching the same category are treated as the
# same constraint, regardless of embedding similarity.
_INTENT_CATEGORIES: dict[str, str] = {
    "brevity": (
        r"\b(concise|brief|succinct|to the point|"
        r"avoid (?:long|lengthy|verbose)(?: explanations)?|"
        r"don'?t (?:be|write) (?:too )?(?:long|verbose)|"
        r"keep (?:it|the answer|the response) short|"
        r"no long explanations)\b"
    ),
    "format": (
        r"\b(json|xml|yaml|markdown|bullet points?|numbered list|table format|"
        r"structured (?:format|output)|"
        r"return (?:the )?(?:result|answer|output) (?:in|as))\b"
    ),
    "tone": r"\b(formal|casual|friendly|professional|polite|conversational)\s+tone\b",
    "citation": r"\b(cite|citation|sources?|references?)\b",
    "confidentiality": r"\b(do not|don'?t|never)\s+(?:reveal|disclose|share|expose)\b",
}

_COMPILED_CATEGORIES = {
    name: re.compile(pattern, re.IGNORECASE) for name, pattern in _INTENT_CATEGORIES.items()
}


def _categorize(sentence: str) -> str | None:
    for name, pattern in _COMPILED_CATEGORIES.items():
        if pattern.search(sentence):
            return name
    return None


def _threshold_for_level(level: int) -> float:
    # Conservative on purpose: this only applies to directives that didn't
    # match a known category, where we have no strong prior that they mean
    # the same thing.
    return max(0.75, 0.95 - level * 0.02)


def consolidate_instructions(
    text: str, level: int, backend: EmbeddingBackend | None = None
) -> str:
    if level <= 0:
        return text

    doc = SentenceDoc.from_text(text)
    sentences = doc.flatten()
    candidates = [i for i, s in enumerate(sentences) if is_directive_sentence(s)]
    if len(candidates) < 2:
        return text

    dropped: set[int] = set()

    categorized: dict[str, list[int]] = {}
    uncategorized: list[int] = []
    for i in candidates:
        category = _categorize(sentences[i])
        if category:
            categorized.setdefault(category, []).append(i)
        else:
            uncategorized.append(i)

    for indices in categorized.values():
        if len(indices) < 2:
            continue
        keeper = min(indices, key=lambda i: len(sentences[i]))
        dropped.update(i for i in indices if i != keeper)

    if len(uncategorized) >= 2:
        backend = backend or EmbeddingBackend()
        local_dropped = find_near_duplicates(
            [sentences[i] for i in uncategorized], backend, _threshold_for_level(level)
        )
        dropped.update(uncategorized[i] for i in local_dropped)

    return doc.to_text(dropped)
