"""Redis-backed cache for full RAG query responses.

Caching happens at the "question -> `QueryResponse`" level, rather than
just caching the embedding or the LLM call individually - so a cache hit
skips BOTH the embedding call and the LLM call, the two things that
actually cost quota/money per `RagPipeline.answer_query` (see
`rag_pipeline.py`). This matters most for the handful of common questions
every visitor tends to ask (e.g. the chat UI's suggestion buttons like
"What are Rahul's skills?") - after the first person asks one, everyone
else gets an instant, free answer until the cache entry expires.

Cache keys are derived from the normalized question text + `top_k`, so
`"What is RAG?"`, `"what is rag?"`, and `"  What is RAG?  "` all hit the
same entry - this is a shared cache across all visitors (not per-user),
since the answer for a given question doesn't depend on who's asking.

Caching is a pure optimization layered on top of `RagPipeline`, never a
hard dependency: if Redis is unreachable or `ENABLE_QUERY_CACHE=false`,
every lookup/write here becomes a silent no-op (see `NullQueryCache`)
rather than breaking the `/api/chat/query` request.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import redis

from app.config import Settings
from app.models.schemas import QueryResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Namespaced + versioned so a future change to what's cached (e.g. adding
# more fields to `QueryResponse`) can bump this prefix to invalidate every
# old entry, rather than risk deserializing stale/incompatible JSON.
_KEY_PREFIX = "rag:query:v1:"


class QueryCache(ABC):
    """Interface for caching `RagPipeline.answer_query` results. Add a new
    backend (e.g. an in-memory dict for tests) by subclassing this."""

    @abstractmethod
    def get(self, question: str, top_k: int) -> QueryResponse | None:
        """Returns the cached response for `question`/`top_k`, or `None`
        on a cache miss (including "Redis is down" - callers can't tell
        the difference, and shouldn't need to: either way, just run the
        pipeline normally)."""

    @abstractmethod
    def set(self, question: str, top_k: int, response: QueryResponse) -> None:
        """Stores `response` for `question`/`top_k`, to be returned by a
        future `get()` call until it expires. Use case: called once per
        cache miss, right after `RagPipeline.answer_query` generates a
        fresh answer."""

    @abstractmethod
    def clear(self) -> None:
        """Drops every cached answer. Use case: called by
        `scripts/ingest_documents.py` after (re-)ingesting documents, so
        visitors don't keep getting answers grounded in since-replaced
        content until the TTL happens to expire."""


def _cache_key(question: str, top_k: int) -> str:
    """Builds a stable cache key from the question + top_k. The question
    is lowercased and has its whitespace collapsed first, so trivially
    different phrasing (extra spaces, different casing) still counts as
    the "same" question; it's then hashed (rather than embedded directly
    in the key) so arbitrarily long/weird question text never produces an
    oversized or invalid Redis key."""
    normalized = " ".join(question.strip().lower().split())
    digest = hashlib.sha256(f"{normalized}|{top_k}".encode("utf-8")).hexdigest()
    return f"{_KEY_PREFIX}{digest}"


class RedisQueryCache(QueryCache):
    """`QueryCache` backed by Redis - the default when `ENABLE_QUERY_CACHE`
    is true and Redis is reachable (see `get_query_cache`)."""

    def __init__(self, settings: Settings):
        self._client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self._ttl_seconds = settings.query_cache_ttl_seconds
        # `from_url` doesn't actually open a connection - ping eagerly so
        # `get_query_cache` can fail fast (and fall back to `NullQueryCache`)
        # if Redis isn't reachable, rather than silently discovering that
        # on the first real request.
        self._client.ping()

    def get(self, question: str, top_k: int) -> QueryResponse | None:
        try:
            raw = self._client.get(_cache_key(question, top_k))
        except redis.RedisError:
            # Redis dying mid-run shouldn't take the API down with it -
            # log once and behave exactly like a cache miss.
            logger.warning("Redis unavailable for cache lookup; skipping cache", exc_info=True)
            return None
        if raw is None:
            return None
        try:
            return QueryResponse.model_validate_json(raw)
        except ValueError:
            # e.g. a leftover entry from an incompatible schema version -
            # treat as a miss rather than crashing the request.
            logger.warning("Discarding malformed cache entry", exc_info=True)
            return None

    def set(self, question: str, top_k: int, response: QueryResponse) -> None:
        try:
            self._client.set(
                _cache_key(question, top_k),
                response.model_dump_json(),
                ex=self._ttl_seconds,
            )
        except redis.RedisError:
            logger.warning("Redis unavailable for cache write; skipping cache", exc_info=True)

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
    whether caching is actually active - it always calls `get`/`set`, and
    this backend just makes both do nothing."""

    def get(self, question: str, top_k: int) -> QueryResponse | None:
        return None

    def set(self, question: str, top_k: int, response: QueryResponse) -> None:
        return None

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
