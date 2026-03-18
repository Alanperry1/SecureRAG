"""
Confidence Gate
===============
Prevents low-confidence responses from being surfaced to the user.

Inspired by calibration / uncertainty quantification work:
  - Ensemble agreement score (how consistent are the top-k chunks?)
  - Coverage check (do we have enough supporting evidence?)
  - Score distribution sharpness (peaked or flat retrieval score curve?)
  - Abstention policy: if confidence < threshold, return a "I don't know"

Returns a GateDecision containing:
  - should_answer  : bool
  - confidence     : float [0, 1]
  - reason         : human-readable explanation
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from config import settings


@dataclass
class GateDecision:
    should_answer: bool
    confidence: float
    reason: str


class ConfidenceGate:
    """
    Determines whether retrieved evidence is sufficient to answer confidently.
    Uses only the clean (non-flagged) chunks that survived security filtering.
    """

    def __init__(
        self,
        confidence_threshold: float = settings.confidence_threshold,
        min_supporting_chunks: int = settings.min_supporting_chunks,
    ) -> None:
        self._conf_thresh = confidence_threshold
        self._min_chunks = min_supporting_chunks

    def evaluate(
        self,
        clean_scores: list[float],
        clean_embeddings: list[np.ndarray],
        query_embedding: np.ndarray,
    ) -> GateDecision:
        """
        Parameters
        ----------
        clean_scores      : retrieval cosine similarities for surviving chunks
        clean_embeddings  : embeddings of surviving chunks
        query_embedding   : query embedding for cross-check
        """
        if len(clean_scores) < self._min_chunks:
            return GateDecision(
                should_answer=False,
                confidence=0.0,
                reason=f"Only {len(clean_scores)} clean chunk(s) survived security "
                       f"filtering (minimum required: {self._min_chunks}).",
            )

        scores = np.array(clean_scores, dtype=np.float32)

        # ── 1. Mean retrieval confidence ─────────────────────────────────────
        mean_score = float(scores.mean())

        # ── 2. Score distribution sharpness (high std = one chunk dominates) ─
        # We prefer a peaked top-score rather than flat mediocre scores.
        top_score = float(scores.max())
        score_sharpness = float(top_score - scores.mean())  # gap: top vs avg

        # ── 3. Intra-chunk agreement (ensemble consistency) ───────────────────
        if len(clean_embeddings) >= 2:
            emb_matrix = np.stack(clean_embeddings, axis=0)           # (N, D)
            sim_matrix = emb_matrix @ emb_matrix.T
            np.fill_diagonal(sim_matrix, np.nan)
            agreement = float(np.nanmean(sim_matrix))                 # avg pairwise sim
        else:
            agreement = 1.0   # single chunk — no disagreement possible

        # ── 4. Query–context alignment ────────────────────────────────────────
        emb_matrix = np.stack(clean_embeddings, axis=0)
        query_alignment = float((emb_matrix @ query_embedding).mean())

        # Composite confidence
        confidence = (
            0.40 * mean_score        +
            0.25 * query_alignment   +
            0.20 * agreement         +
            0.15 * min(score_sharpness / 0.3, 1.0)
        )
        confidence = float(np.clip(confidence, 0.0, 1.0))

        if confidence < self._conf_thresh:
            return GateDecision(
                should_answer=False,
                confidence=confidence,
                reason=(
                    f"Confidence {confidence:.2%} below threshold {self._conf_thresh:.0%}. "
                    f"Retrieved evidence may be insufficient or conflicting."
                ),
            )

        return GateDecision(
            should_answer=True,
            confidence=confidence,
            reason=f"Confidence {confidence:.2%} — {len(clean_scores)} clean chunks support the answer.",
        )
