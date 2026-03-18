"""
Security Orchestrator
=====================
Runs the full three-stage defence on a batch of retrieved chunks:

  Stage 1 — Injection Detection   (per-chunk, parallel-friendly)
  Stage 2 — Poisoning Detection   (batch, requires full set context)
  Stage 3 — Anomaly Scoring       (per-chunk PyTorch autoencoder)

Output: SecurityReport summarising which chunks survived and why.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import settings
from core.retriever import RetrievedChunk
from core.security.injection_detector import InjectionDetector, InjectionReport
from core.security.poisoning_detector import PoisoningDetector, PoisoningReport
from core.security.anomaly_scorer import AnomalyScorer
from core.security.confidence_gate import ConfidenceGate, GateDecision

log = logging.getLogger(__name__)


@dataclass
class ChunkVerdict:
    chunk: RetrievedChunk
    injection: InjectionReport
    poisoning: PoisoningReport
    anomaly_score: float
    is_anomalous: bool
    safe: bool                  # True only if all three checks pass


@dataclass
class SecurityReport:
    query: str
    total_chunks: int
    safe_chunks: int
    flagged_chunks: int
    verdicts: list[ChunkVerdict]
    gate_decision: Optional[GateDecision]
    latency_ms: float
    summary: dict = field(default_factory=dict)

    @property
    def clean_chunks(self) -> list[RetrievedChunk]:
        return [v.chunk for v in self.verdicts if v.safe]


class SecurityOrchestrator:
    """Runs all three defence stages and the confidence gate."""

    def __init__(self) -> None:
        self._injection = InjectionDetector()
        self._poisoning = PoisoningDetector()
        self._anomaly = AnomalyScorer.load_or_init()
        self._gate = ConfidenceGate()

    def inspect(
        self,
        query: str,
        query_embedding: np.ndarray,
        chunks: list[RetrievedChunk],
    ) -> SecurityReport:
        t0 = time.perf_counter()

        if not chunks:
            return SecurityReport(
                query=query, total_chunks=0, safe_chunks=0, flagged_chunks=0,
                verdicts=[], gate_decision=None, latency_ms=0.0,
            )

        chunk_ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        embeddings = [c.embedding for c in chunks]

        # ── Stage 1: Injection detection ─────────────────────────────────────
        injection_reports: list[InjectionReport] = [
            self._injection.score(c.chunk_id, c.text, query_embedding)
            for c in chunks
        ]

        # ── Stage 2: Poisoning detection (batch) ─────────────────────────────
        poisoning_reports: list[PoisoningReport] = self._poisoning.score_batch(
            chunk_ids, texts, embeddings, query_embedding
        )

        # ── Stage 3: Autoencoder anomaly scoring ─────────────────────────────
        emb_matrix = np.stack(embeddings, axis=0)
        anomaly_scores: np.ndarray = self._anomaly.score_batch(emb_matrix)

        # ── Combine verdicts ──────────────────────────────────────────────────
        verdicts: list[ChunkVerdict] = []
        for i, chunk in enumerate(chunks):
            inj = injection_reports[i]
            poi = poisoning_reports[i]
            anom_score = float(anomaly_scores[i])
            is_anom = anom_score > settings.anomaly_recon_threshold

            safe = not (inj.is_injected or poi.is_poisoned or is_anom)

            if not safe:
                reasons = []
                if inj.is_injected:
                    reasons.append(f"injection({inj.risk_score:.2f})")
                if poi.is_poisoned:
                    reasons.append(f"poisoning({poi.risk_score:.2f})")
                if is_anom:
                    reasons.append(f"anomaly({anom_score:.4f})")
                log.warning("Chunk %s FLAGGED: %s", chunk.chunk_id, ", ".join(reasons))
                chunk.flagged = True
                chunk.flag_reason = " | ".join(reasons)

            verdicts.append(ChunkVerdict(
                chunk=chunk,
                injection=inj,
                poisoning=poi,
                anomaly_score=anom_score,
                is_anomalous=is_anom,
                safe=safe,
            ))

        # ── Confidence gate ───────────────────────────────────────────────────
        clean = [v.chunk for v in verdicts if v.safe]
        gate_decision: Optional[GateDecision] = None
        if clean:
            gate_decision = self._gate.evaluate(
                clean_scores=[c.score for c in clean],
                clean_embeddings=[c.embedding for c in clean],
                query_embedding=query_embedding,
            )

        latency_ms = (time.perf_counter() - t0) * 1000
        flagged = sum(1 for v in verdicts if not v.safe)

        return SecurityReport(
            query=query,
            total_chunks=len(chunks),
            safe_chunks=len(clean),
            flagged_chunks=flagged,
            verdicts=verdicts,
            gate_decision=gate_decision,
            latency_ms=latency_ms,
            summary={
                "injection_flagged": sum(1 for r in injection_reports if r.is_injected),
                "poisoning_flagged": sum(1 for r in poisoning_reports if r.is_poisoned),
                "anomaly_flagged": int((anomaly_scores > settings.anomaly_recon_threshold).sum()),
            },
        )
