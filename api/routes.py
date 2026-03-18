"""
API Routes
"""
from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import APIRouter, HTTPException

from api.schemas import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse,
    ChunkSummary, SecuritySummary,
    HealthResponse, CacheStatsResponse,
)
from cache.redis_cache import ChunkCache
from core.pipeline import SecureRAGPipeline

log = logging.getLogger(__name__)
router = APIRouter()


@lru_cache(maxsize=1)
def _get_pipeline() -> SecureRAGPipeline:
    return SecureRAGPipeline()


@lru_cache(maxsize=1)
def _get_cache() -> ChunkCache:
    return ChunkCache()


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    pipeline = _get_pipeline()
    cache = _get_cache()
    return HealthResponse(
        status="ok",
        total_chunks_in_db=pipeline._retriever.count(),
        cache_stats=cache.stats(),
    )


# ── Ingest ────────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, tags=["data"])
def ingest(body: IngestRequest):
    """Embed and store documents into the vector store."""
    pipeline = _get_pipeline()
    try:
        ids = pipeline.ingest(body.texts, body.metadatas)
    except Exception as exc:
        log.exception("Ingest error")
        raise HTTPException(status_code=500, detail=str(exc))
    return IngestResponse(chunk_ids=ids, count=len(ids))


# ── Query ─────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, tags=["rag"])
def query(body: QueryRequest):
    """
    Run the full SecureRAG pipeline:
      retrieve → security scan → confidence gate → LLM → response
    """
    pipeline = _get_pipeline()
    try:
        result = pipeline.query(body.question)
    except Exception as exc:
        log.exception("Query error")
        raise HTTPException(status_code=500, detail=str(exc))

    report = result.security_report
    gate = report.gate_decision
    sec_summary = report.summary

    return QueryResponse(
        query=result.query,
        answer=result.answer,
        aborted=result.aborted,
        abort_reason=result.abort_reason,
        sources=[
            ChunkSummary(**s) for s in result.sources
        ],
        security=SecuritySummary(
            total_chunks=report.total_chunks,
            safe_chunks=report.safe_chunks,
            flagged_chunks=report.flagged_chunks,
            injection_flagged=sec_summary.get("injection_flagged", 0),
            poisoning_flagged=sec_summary.get("poisoning_flagged", 0),
            anomaly_flagged=sec_summary.get("anomaly_flagged", 0),
            latency_ms=round(report.latency_ms, 2),
            confidence=round(gate.confidence, 4) if gate else None,
            gate_passed=gate.should_answer if gate else None,
        ),
    )


# ── Cache management ──────────────────────────────────────────────────────────

@router.get("/cache/stats", response_model=CacheStatsResponse, tags=["meta"])
def cache_stats():
    cache = _get_cache()
    return CacheStatsResponse(**cache.stats())


@router.delete("/cache", tags=["meta"])
def clear_cache():
    """Flush all cached flagged chunk IDs."""
    _get_cache().clear()
    return {"message": "Cache cleared."}
