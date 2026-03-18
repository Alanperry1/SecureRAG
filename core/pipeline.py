"""
SecureRAG Pipeline
==================
Full pipeline:  Query → Retrieval → Security Layer → LLM → Response

The pipeline:
1. Embeds the query
2. Retrieves top-k chunks from ChromaDB
3. Checks Redis cache for previously flagged chunk IDs
4. Runs the SecurityOrchestrator on un-cached chunks
5. Writes newly flagged chunks to Redis cache
6. Runs the ConfidenceGate
7. Feeds clean chunks to the LLM (OpenAI via LangChain)
8. Returns a PipelineResult
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings
from core.embedder import get_embedder
from core.retriever import Retriever, RetrievedChunk
from core.security.orchestrator import SecurityOrchestrator, SecurityReport
from cache.redis_cache import ChunkCache

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    query: str
    answer: str
    security_report: SecurityReport
    sources: list[dict] = field(default_factory=list)
    aborted: bool = False
    abort_reason: Optional[str] = None


_SYSTEM_PROMPT = """You are a helpful, precise assistant.
Answer ONLY based on the provided context.
If the context is insufficient, say so clearly.
Do NOT invent information."""


class SecureRAGPipeline:
    """End-to-end SecureRAG pipeline."""

    def __init__(self) -> None:
        self._embedder = get_embedder()
        self._retriever = Retriever()
        self._security = SecurityOrchestrator()
        self._cache = ChunkCache()

        if settings.openai_api_key:
            self._llm = ChatOpenAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                openai_api_key=settings.openai_api_key,
            )
        else:
            self._llm = None
            log.warning("No OPENAI_API_KEY set — LLM calls will be stubbed.")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, texts: list[str], metadatas: Optional[list[dict]] = None) -> list[str]:
        """Insert documents into the vector store."""
        return self._retriever.ingest(texts, metadatas)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, question: str) -> PipelineResult:
        # 1. Embed query
        q_emb: np.ndarray = self._embedder.embed(question)

        # 2. Retrieve chunks
        chunks: list[RetrievedChunk] = self._retriever.retrieve(question)

        if not chunks:
            return PipelineResult(
                query=question,
                answer="No documents found in the knowledge base.",
                security_report=self._security.inspect(question, q_emb, []),
                aborted=True,
                abort_reason="empty_retrieval",
            )

        # 3. Cache pre-filter — skip re-scoring known bad chunks
        cached_bad_ids = self._cache.get_flagged_ids([c.chunk_id for c in chunks])
        pre_flagged: list[RetrievedChunk] = []
        to_inspect: list[RetrievedChunk] = []
        for chunk in chunks:
            if chunk.chunk_id in cached_bad_ids:
                chunk.flagged = True
                chunk.flag_reason = "cached:previously_flagged"
                pre_flagged.append(chunk)
            else:
                to_inspect.append(chunk)

        # 4. Security inspection
        report: SecurityReport = self._security.inspect(question, q_emb, to_inspect)

        # Merge pre-flagged back for complete report accounting
        # (they don't affect the security report verdicts, treated as extra flagged)
        total_flagged = report.flagged_chunks + len(pre_flagged)

        # 5. Write newly flagged chunks to cache
        newly_flagged_ids = [
            v.chunk.chunk_id for v in report.verdicts if not v.safe
        ]
        if newly_flagged_ids:
            self._cache.set_flagged(newly_flagged_ids)

        # 6. Confidence gate
        gate = report.gate_decision
        if gate is not None and not gate.should_answer:
            return PipelineResult(
                query=question,
                answer=f"I'm not confident enough to answer. {gate.reason}",
                security_report=report,
                aborted=True,
                abort_reason="confidence_gate",
            )

        clean_chunks = report.clean_chunks
        if not clean_chunks:
            return PipelineResult(
                query=question,
                answer="All retrieved context was flagged as potentially malicious. Refusing to answer.",
                security_report=report,
                aborted=True,
                abort_reason="all_chunks_flagged",
            )

        # 7. Build context and call LLM
        context = "\n\n---\n\n".join(
            f"[Source {i+1}]\n{c.text}" for i, c in enumerate(clean_chunks)
        )
        answer = self._call_llm(question, context)

        sources = [
            {"chunk_id": c.chunk_id, "score": round(c.score, 4), "text_preview": c.text[:150]}
            for c in clean_chunks
        ]

        return PipelineResult(
            query=question,
            answer=answer,
            security_report=report,
            sources=sources,
        )

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call_llm(self, question: str, context: str) -> str:
        if self._llm is None:
            return f"[STUB — no LLM configured]\n\nContext:\n{context}"

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ]
        response = self._llm.invoke(messages)
        return response.content
