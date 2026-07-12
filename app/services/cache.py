"""Redis-backed cache for full RAG query responses.

Caching is a pure optimization layered on top of `RagPipeline`, never a
hard dependency: if Redis is unreachable or `ENABLE_QUERY_CACHE=false`,
`clear()` here becomes a silent no-op (see `NullQueryCache`) rather than
breaking document re-ingestion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import redis

from app.config import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Namespaced + versioned so a future change to what's cached can bump this
# prefix to invalidate every old entry, rather than risk deserializing
# stale/incompatible JSON.
_KEY_PREFIX = "rag:query:v1:"


class QueryCache(ABC):
    """Interface for caching query responses. Add a new backend (e.g. an
    in-memory dict for tests) by subclassing this."""

    @abstractmethod
    def clear(self) -> None:
        """Drops every cached answer. Use case: called by
        `scripts/ingest_documents.py` after (re-)ingesting documents, so
        visitors don't keep getting answers grounded in since-replaced
        content until the TTL happens to expire."""


class RedisQueryCache(QueryCache):
    """`QueryCache` backed by Redis - the default when `ENABLE_QUERY_CACHE`
    is true and Redis is reachable (see `get_query_cache`)."""

    def __init__(self, settings: Settings):
        self._client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        # `from_url` doesn't actually open a connection - ping eagerly so
        # `get_query_cache` can fail fast (and fall back to `NullQueryCache`)
        # if Redis isn't reachable, rather than silently discovering that
        # on the first real request.
        self._client.ping()

    def clear(self) -> None:
        try:
            keys = list(self._client.scan_iter(match=f"{_KEY_PREFIX}*"))
            if keys:
                self._client.delete(*keys)
        except redis.RedisError:
            logger.warning("Redis unavailable for cache clear; skipping", exc_info=True)


class NullQueryCache(QueryCache):
    """No-op cache used when caching is disabled (`ENABLE_QUERY_CACHE=false`)
    or Redis couldn't be reached at startup. Keeps `RagPipeline` unaware of
    whether caching is actually active - it always calls `clear()`, and
    this backend just makes that a no-op."""

    def clear(self) -> None:
        return None


def get_query_cache(settings: Settings) -> QueryCache:
    """Factory that returns a working `RedisQueryCache` if caching is
    enabled and Redis is reachable right now, or a `NullQueryCache`
    otherwise - callers never need to check which one they got."""
    if not settings.enable_query_cache:
        return NullQueryCache()
    try:
        return RedisQueryCache(settings)
    except redis.RedisError:
        logger.warning(
            "Could not connect to Redis at %s - query caching disabled",
            settings.redis_url,
        )
        return NullQueryCache()
