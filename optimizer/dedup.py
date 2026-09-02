"""Deduplicates near-identical background/context sentences, e.g. "The
customer is unhappy because the delivery was delayed" and "The delayed
delivery made the customer dissatisfied" carry the same meaning -- only the
clearer one is kept.
"""

from __future__ import annotations

from .clustering import find_near_duplicates
from .embeddings import EmbeddingBackend
from .text_utils import SentenceDoc


def _threshold_for_level(level: int) -> float:
    # Higher level -> lower threshold -> more sentences are treated as
    # near-duplicates and removed.
    return max(0.5, 0.9 - level * 0.035)


def deduplicate_context(
    text: str, level: int, backend: EmbeddingBackend | None = None
) -> str:
    if level <= 0:
        return text

    doc = SentenceDoc.from_text(text)
    sentences = doc.flatten()
    if len(sentences) < 2:
        return text

    backend = backend or EmbeddingBackend()
    dropped = find_near_duplicates(sentences, backend, _threshold_for_level(level))

    return doc.to_text(dropped)
