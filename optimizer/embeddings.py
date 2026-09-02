"""Shared text-embedding backend used by the deduplication, instruction
consolidation, and semantic summarization modules.

Uses a sentence-transformers model for true semantic similarity. If the
optional model weights can't be loaded (no internet on first run, package
not installed, etc.), it transparently falls back to a TF-IDF + cosine
representation so the rest of the pipeline keeps working offline.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingBackend:
    _model_cache: dict[str, object] = {}
    _failed_models: set[str] = set()

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.model_name = model_name
        self._st_model = self._get_sentence_transformer(model_name)

    @classmethod
    def _get_sentence_transformer(cls, model_name: str):
        if model_name in cls._failed_models:
            return None
        if model_name in cls._model_cache:
            return cls._model_cache[model_name]
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
            cls._model_cache[model_name] = model
            return model
        except Exception:
            cls._failed_models.add(model_name)
            return None

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1))
        if self._st_model is not None:
            return np.asarray(self._st_model.encode(list(texts)))
        # TF-IDF fallback: fit fresh on each call since the vocabulary is
        # only meaningful within one comparison set.
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(texts).toarray()

    def similarity_matrix(self, texts: list[str]) -> np.ndarray:
        vectors = self.encode(texts)
        if vectors.shape[0] == 0:
            return np.zeros((0, 0))
        return cosine_similarity(vectors)
