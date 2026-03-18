"""
Embedding layer — wraps SentenceTransformers for consistent vector generation.
"""
from __future__ import annotations

import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from config import settings


class Embedder:
    """Singleton-style embedding wrapper so the model loads once."""

    def __init__(self, model_name: str = settings.embedding_model) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer = SentenceTransformer(model_name)
        self.dim: int = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """Return a 1-D float32 embedding for *text*."""
        return self._model.encode(text, normalize_embeddings=True).astype(np.float32)

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Return shape (N, dim) float32 array for a list of texts."""
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()
