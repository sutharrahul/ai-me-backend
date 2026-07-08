"""FastAPI dependency providers (the app's dependency-injection wiring).

Route handlers ask for services via FastAPI's `Depends(...)` mechanism
instead of constructing them inline. Each provider below is wrapped in
`@lru_cache`, so the first call builds the object (connecting to Qdrant,
creating the Gemini client, etc.) and every subsequent call — across every
request — reuses that same singleton instance.
"""

from functools import lru_cache

from app.config import Settings, get_settings
from app.core.rag_pipeline import RagPipeline
from app.services.cache import QueryCache, get_query_cache
from app.services.embeddings import GeminiEmbeddingProvider, get_embedding_provider
from app.services.llm import get_llm_client
from app.services.vector_store import QdrantVectorStore, get_vector_store


@lru_cache
def get_embedding_provider_instance() -> GeminiEmbeddingProvider:
    """Singleton embedding provider. Shared by the vector store and the RAG
    pipeline so both always use the same configured model and API key."""
    return get_embedding_provider(get_settings())


@lru_cache
def get_vector_store_instance() -> QdrantVectorStore:
    """Singleton Qdrant vector store. Opens the connection on first use and
    reuses it across all requests."""
    return get_vector_store(get_settings())


@lru_cache
def get_query_cache_instance() -> QueryCache:
    """Singleton query cache (Redis-backed if reachable and
    `ENABLE_QUERY_CACHE=true`, otherwise a silent no-op fallback)."""
    return get_query_cache(get_settings())


@lru_cache
def get_rag_pipeline() -> RagPipeline:
    """Singleton `RagPipeline` — the main orchestrator used by the chat
    endpoint to answer questions."""
    settings: Settings = get_settings()
    return RagPipeline(
        settings=settings,
        embedding_provider=get_embedding_provider_instance(),
        vector_store=get_vector_store_instance(),
        llm_client=get_llm_client(settings),
        query_cache=get_query_cache_instance(),
    )
