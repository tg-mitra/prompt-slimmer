"""Near-duplicate sentence detection shared by instruction consolidation and
context deduplication -- the two techniques differ only in which sentences
they consider as candidates, not in how duplicates are found and resolved.
"""

from __future__ import annotations

from .embeddings import EmbeddingBackend


def find_near_duplicates(
    sentences: list[str], backend: EmbeddingBackend, threshold: float
) -> set[int]:
    """Return the indices of sentences that should be dropped because a
    near-duplicate (cosine similarity >= threshold) is already kept.

    Between two near-duplicates, the shorter/clearer phrasing is kept.
    """
    if len(sentences) < 2:
        return set()

    sim = backend.similarity_matrix(sentences)
    alive = set(range(len(sentences)))
    dropped: set[int] = set()

    for a in range(len(sentences)):
        if a not in alive:
            continue
        for b in range(a + 1, len(sentences)):
            if b not in alive or sim[a][b] < threshold:
                continue
            loser = b if len(sentences[a]) <= len(sentences[b]) else a
            alive.discard(loser)
            dropped.add(loser)
            if loser == a:
                break

    return dropped
