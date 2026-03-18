"""
Data Poisoning Detector
=======================
Detects poisoned chunks in the retrieval set using:

1. Cosine outlier scoring     — chunks with embeddings far from the centroid
                                of retrieved set are statistical outliers
2. Query–chunk semantic drift — chunk that is anomalously unlike the query
                                relative to peers (z-score)
3. Intra-set similarity gap   — poisoned chunks often cluster separately from
                                the legitimate set (gap statistic)
4. Content–embedding mismatch — re-embed the chunk and compare: if stored
                                embedding diverges from fresh embedding, the
                                chunk may have been tampered with in the DB

Returns a PoisoningReport per chunk.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from config import settings
from core.embedder import get_embedder


@dataclass
class PoisoningReport:
    chunk_id: str
    text: str
    is_poisoned: bool
    risk_score: float               # composite [0, 1]
    cosine_outlier_score: float     # how far from centroid
    query_drift_zscore: float       # how far from query relative to peers
    embedding_tamper_score: float   # fresh vs stored embedding divergence
    details: dict = field(default_factory=dict)


class PoisoningDetector:
    """
    Operates on the *full set* of retrieved chunks simultaneously to detect
    statistical outliers that signal injected / poisoned content.
    """

    def __init__(
        self,
        z_threshold: float = settings.cosine_outlier_z_threshold,
        risk_threshold: float = settings.poison_score_threshold,
    ) -> None:
        self._z_thresh = z_threshold
        self._risk_thresh = risk_threshold
        self._embedder = get_embedder()

    # ── Public API ────────────────────────────────────────────────────────────

    def score_batch(
        self,
        chunk_ids: list[str],
        texts: list[str],
        embeddings: list[np.ndarray],           # stored embeddings from vector DB
        query_embedding: np.ndarray,
    ) -> list[PoisoningReport]:
        """Score all *chunks* together — batch context is required for z-scoring."""
        if not texts:
            return []

        emb_matrix = np.stack(embeddings, axis=0)           # (N, D)

        # ── 1. Centroid outlier score ─────────────────────────────────────────
        centroid = emb_matrix.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-9)
        cosine_to_centroid = emb_matrix @ centroid           # cosine sim, (N,)

        # ── 2. Query-drift z-score ────────────────────────────────────────────
        query_sims = emb_matrix @ query_embedding            # (N,)
        q_mean = query_sims.mean()
        q_std = query_sims.std() + 1e-9
        query_zscores = (query_sims - q_mean) / q_std       # negative = outlier

        # ── 3. Intra-set average similarity ──────────────────────────────────
        intra_sim_matrix = emb_matrix @ emb_matrix.T        # (N, N)
        np.fill_diagonal(intra_sim_matrix, np.nan)
        intra_avg = np.nanmean(intra_sim_matrix, axis=1)    # (N,)

        # ── 4. Embedding tamper detection ─────────────────────────────────────
        fresh_embs = self._embedder.embed_batch(texts)       # re-embed from text
        tamper_scores = 1.0 - np.clip(
            np.sum(emb_matrix * fresh_embs, axis=1), -1.0, 1.0
        )  # 0 = identical, 2 = opposite

        reports: list[PoisoningReport] = []
        for i in range(len(texts)):
            cosine_score = float(1.0 - cosine_to_centroid[i])          # 0=centroid, 1=outlier
            drift_z = float(-query_zscores[i])                          # high=low sim to query
            tamper = float(np.clip(tamper_scores[i] / 0.05, 0, 1))     # 0.05 cos diff → full score

            # Centroid z-score for outlier flag
            cent_mean = cosine_to_centroid.mean()
            cent_std = cosine_to_centroid.std() + 1e-9
            cent_z = (cosine_to_centroid[i] - cent_mean) / cent_std

            intra_score = float(1.0 - np.clip(intra_avg[i], 0, 1))     # low intra-sim = outlier

            # Composite (weighted)
            composite = float(np.clip(
                0.35 * cosine_score +
                0.30 * min(max(drift_z, 0) / self._z_thresh, 1.0) +
                0.20 * tamper +
                0.15 * intra_score,
                0.0, 1.0,
            ))

            reports.append(PoisoningReport(
                chunk_id=chunk_ids[i],
                text=texts[i],
                is_poisoned=composite >= self._risk_thresh,
                risk_score=composite,
                cosine_outlier_score=cosine_score,
                query_drift_zscore=float(drift_z),
                embedding_tamper_score=tamper,
                details={
                    "centroid_zscore": round(float(cent_z), 4),
                    "intra_avg_sim": round(float(intra_avg[i]), 4),
                    "query_cosine_sim": round(float(query_sims[i]), 4),
                },
            ))
        return reports
