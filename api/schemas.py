"""
API Schemas — Pydantic models for request/response validation.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Documents to embed and store")
    metadatas: Optional[list[dict]] = Field(None, description="Optional metadata per document")


class IngestResponse(BaseModel):
    chunk_ids: list[str]
    count: int


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2048)


class ChunkSummary(BaseModel):
    chunk_id: str
    score: float
    text_preview: str


class SecuritySummary(BaseModel):
    total_chunks: int
    safe_chunks: int
    flagged_chunks: int
    injection_flagged: int
    poisoning_flagged: int
    anomaly_flagged: int
    latency_ms: float
    confidence: Optional[float]
    gate_passed: Optional[bool]


class QueryResponse(BaseModel):
    query: str
    answer: str
    aborted: bool
    abort_reason: Optional[str]
    sources: list[ChunkSummary]
    security: SecuritySummary


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    total_chunks_in_db: int
    cache_stats: dict


# ── Cache ─────────────────────────────────────────────────────────────────────

class CacheStatsResponse(BaseModel):
    backend: str
    memory_entries: int
    redis_entries: Optional[int]
