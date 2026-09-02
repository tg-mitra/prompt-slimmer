"""Extractive semantic summarization for long background/context paragraphs.

Only touches paragraphs above a configured length threshold, and only keeps
the summary if it stays close enough (by embedding similarity) to the
original -- otherwise the paragraph is left untouched. This follows the
instructions' guidance to pair summarization with a similarity safety check,
since compression here risks dropping details.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import EmbeddingBackend
from .text_utils import split_paragraphs, split_sentences


def _keep_ratio_for_level(level: int) -> float:
    return max(0.3, 1.0 - level * 0.07)


def summarize_long_context(
    text: str,
    level: int,
    min_chars_to_trigger: int = 800,
    min_retained_similarity: float = 0.6,
    backend: EmbeddingBackend | None = None,
) -> str:
    if level <= 0:
        return text

    backend = backend or EmbeddingBackend()
    keep_ratio = _keep_ratio_for_level(level)

    out_paragraphs = []
    for paragraph in split_paragraphs(text):
        if len(paragraph) < min_chars_to_trigger:
            out_paragraphs.append(paragraph)
            continue

        sentences = split_sentences(paragraph)
        if len(sentences) < 4:
            out_paragraphs.append(paragraph)
            continue

        summary = _extractive_summary(sentences, keep_ratio, backend)
        if _retains_meaning(paragraph, summary, min_retained_similarity, backend):
            out_paragraphs.append(summary)
        else:
            out_paragraphs.append(paragraph)

    return "\n\n".join(out_paragraphs)


def _extractive_summary(sentences: list[str], keep_ratio: float, backend: EmbeddingBackend) -> str:
    n_keep = max(2, round(len(sentences) * keep_ratio))
    if n_keep >= len(sentences):
        return " ".join(sentences)

    vectors = backend.encode(sentences)
    centroid = vectors.mean(axis=0, keepdims=True)
    scores = cosine_similarity(vectors, centroid).flatten()

    top_indices = sorted(np.argsort(scores)[-n_keep:])
    return " ".join(sentences[i] for i in top_indices)


def _retains_meaning(
    original: str, summary: str, min_similarity: float, backend: EmbeddingBackend
) -> bool:
    vectors = backend.encode([original, summary])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return score >= min_similarity
