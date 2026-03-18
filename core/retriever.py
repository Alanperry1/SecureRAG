"""
Vector-store retrieval layer using ChromaDB.
Supports ingesting documents and querying top-k chunks with their embeddings.
"""
from __future__ import annotations

import uuid
import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings
from dataclasses import dataclass, field
from typing import Optional

from config import settings
from core.embedder import get_embedder


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    embedding: np.ndarray
    score: float                    # cosine similarity to query
    metadata: dict = field(default_factory=dict)
    flagged: bool = False
    flag_reason: Optional[str] = None


class Retriever:
    """ChromaDB-backed retriever that also returns raw embeddings."""

    def __init__(self) -> None:
        self._embedder = get_embedder()
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, texts: list[str], metadatas: Optional[list[dict]] = None) -> list[str]:
        """Embed and store *texts*.  Returns assigned chunk IDs."""
        if not texts:
            return []

        embeddings = self._embedder.embed_batch(texts)
        ids = [str(uuid.uuid4()) for _ in texts]
        metas = metadatas or [{} for _ in texts]

        self._col.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metas,
        )
        return ids

    # ── Querying ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = settings.top_k) -> list[RetrievedChunk]:
        """Return top-k chunks with their raw embeddings for security scoring."""
        q_emb = self._embedder.embed(query)

        results = self._col.query(
            query_embeddings=[q_emb.tolist()],
            n_results=min(top_k, self._col.count()),
            include=["documents", "embeddings", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        for i, doc in enumerate(results["documents"][0]):
            dist = results["distances"][0][i]          # ChromaDB returns L2 distance for cosine space
            sim = 1.0 - dist / 2.0                     # convert to cosine similarity ∈ [0,1]
            emb = np.array(results["embeddings"][0][i], dtype=np.float32)
            chunks.append(
                RetrievedChunk(
                    chunk_id=results["ids"][0][i],
                    text=doc,
                    embedding=emb,
                    score=float(sim),
                    metadata=results["metadatas"][0][i] or {},
                )
            )
        return chunks

    def count(self) -> int:
        return self._col.count()
