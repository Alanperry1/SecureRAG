"""
Redis Cache — Flagged Chunk Store
==================================
Caches chunk IDs that have been flagged as injected or poisoned so that:
  1. Subsequent queries don't pay the re-scoring latency cost.
  2. Repeated retrieval of poisoned chunks is detected immediately.

Falls back to an in-memory dict when Redis is unavailable.
"""
from __future__ import annotations

import logging
from typing import Optional

from config import settings

log = logging.getLogger(__name__)

_REDIS_AVAILABLE = False
_redis_client = None

if settings.redis_enabled:
    try:
        import redis
        _redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
            socket_connect_timeout=1,
        )
        _redis_client.ping()
        _REDIS_AVAILABLE = True
        log.info("Redis connected at %s:%s", settings.redis_host, settings.redis_port)
    except Exception as exc:
        log.warning("Redis unavailable (%s) — using in-memory fallback cache.", exc)


_FLAGGED_KEY_PREFIX = "securerag:flagged:"


class ChunkCache:
    """Thread-safe flagged-chunk cache with Redis backend and memory fallback."""

    def __init__(self) -> None:
        self._memory: dict[str, bool] = {}   # fallback

    # ── Write ─────────────────────────────────────────────────────────────────

    def set_flagged(self, chunk_ids: list[str], ttl: int = settings.redis_ttl_seconds) -> None:
        for cid in chunk_ids:
            if _REDIS_AVAILABLE and _redis_client:
                try:
                    _redis_client.setex(_FLAGGED_KEY_PREFIX + cid, ttl, "1")
                    continue
                except Exception as exc:
                    log.warning("Redis write failed for %s: %s", cid, exc)
            self._memory[cid] = True

    # ── Read ──────────────────────────────────────────────────────────────────

    def is_flagged(self, chunk_id: str) -> bool:
        if _REDIS_AVAILABLE and _redis_client:
            try:
                return bool(_redis_client.exists(_FLAGGED_KEY_PREFIX + chunk_id))
            except Exception:
                pass
        return self._memory.get(chunk_id, False)

    def get_flagged_ids(self, chunk_ids: list[str]) -> set[str]:
        """Return subset of *chunk_ids* that are cached as flagged."""
        if not chunk_ids:
            return set()

        if _REDIS_AVAILABLE and _redis_client:
            try:
                keys = [_FLAGGED_KEY_PREFIX + cid for cid in chunk_ids]
                results = _redis_client.mget(keys)
                return {cid for cid, val in zip(chunk_ids, results) if val}
            except Exception as exc:
                log.warning("Redis mget failed: %s", exc)

        return {cid for cid in chunk_ids if self._memory.get(cid)}

    # ── Maintenance ───────────────────────────────────────────────────────────

    def clear(self) -> None:
        self._memory.clear()
        if _REDIS_AVAILABLE and _redis_client:
            try:
                keys = _redis_client.keys(_FLAGGED_KEY_PREFIX + "*")
                if keys:
                    _redis_client.delete(*keys)
            except Exception as exc:
                log.warning("Redis clear failed: %s", exc)

    def stats(self) -> dict:
        memory_count = len(self._memory)
        redis_count: Optional[int] = None
        if _REDIS_AVAILABLE and _redis_client:
            try:
                redis_count = len(_redis_client.keys(_FLAGGED_KEY_PREFIX + "*"))
            except Exception:
                pass
        return {
            "backend": "redis" if _REDIS_AVAILABLE else "memory",
            "memory_entries": memory_count,
            "redis_entries": redis_count,
        }
