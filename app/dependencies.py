"""FastAPI dependency providers (a.k.a. the app's dependency-injection wiring).

Route handlers ask for services (like `RagPipeline`) via FastAPI's
`Depends(...)` mechanism instead of constructing them inline. That makes it
easy to swap implementations later (e.g. in tests, via
`app.dependency_overrides`) without touching route code.

Every provider below is wrapped in `@lru_cache`, so the first call builds
the object (connecting to Postgres, creating the Gemini client, etc.) and
every subsequent call - across every request - reuses that same instance.
This avoids reconnecting to the database or re-authenticating with Gemini
on every single API call.
"""

from functools import lru_cache

from app.config import Settings, get_settings
from app.core.rag_pipeline import RagPipeline
from app.services.cache import QueryCache, get_query_cache
from app.services.document_store import DocumentStore
from app.services.embeddings import EmbeddingProvider, get_embedding_provider
from app.services.llm import get_llm_client
from app.services.vector_store import VectorStoreBackend, get_vector_store


@lru_cache
def get_document_store() -> DocumentStore:
    """Provides the singleton `DocumentStore`, which tracks (in a small
    JSON file) which documents have been ingested, their status, and how
    many chunks they produced. Used by the ingestion script and the RAG
    pipeline to support idempotent re-ingestion and clean deletion."""
    settings = get_settings()
    return DocumentStore(settings.documents_index_path)


@lru_cache
def get_embedding_provider_instance() -> EmbeddingProvider:
    """Provides the singleton embedding provider (Gemini or OpenAI,
    depending on `settings.embedding_provider`). Shared by both the vector
    store (which needs it to satisfy LangChain's `PGVector` constructor)
    and the RAG pipeline (which uses it to embed queries/chunks directly)."""
    return get_embedding_provider(get_settings())


@lru_cache
def get_vector_store_instance() -> VectorStoreBackend:
    """Provides the singleton vector store backend (Postgres/pgvector or
    Qdrant, depending on `settings.vector_store_provider`), opening its
    connection on first use. Cached so the app keeps a single connection
    alive instead of reconnecting per request."""
    settings = get_settings()
    return get_vector_store(settings, get_embedding_provider_instance())


@lru_cache
def get_query_cache_instance() -> QueryCache:
    """Provides the singleton query cache (Redis-backed if reachable and
    `ENABLE_QUERY_CACHE=true`, otherwise a no-op fallback - see
    `services/cache.py`). Shared by the RAG pipeline (to check/populate it
    per question) and the ingestion script (to clear it after content
    changes)."""
    return get_query_cache(get_settings())


@lru_cache
def get_rag_pipeline() -> RagPipeline:
    """Provides the singleton `RagPipeline`, the main orchestrator that
    ties together settings, embeddings, the vector store, the LLM client,
    the document store, and the query cache. This is the object route
    handlers (and the ingestion script) actually call to ingest documents
    or answer questions - see `core/rag_pipeline.py`."""
    settings: Settings = get_settings()
    return RagPipeline(
        settings=settings,
        embedding_provider=get_embedding_provider_instance(),
        vector_store=get_vector_store_instance(),
        llm_client=get_llm_client(settings),
        document_store=get_document_store(),
        query_cache=get_query_cache_instance(),
    )
