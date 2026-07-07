"""Embedding provider abstraction.

An "embedding" is a list of floats representing the meaning of a piece of
text, used to find semantically similar chunks via vector search. Wrapping
the embedding call behind the `EmbeddingProvider` interface makes it easy
to swap providers without touching the rest of the pipeline - `RagPipeline`
and `VectorStore` only ever call `.embed_documents()`/`.embed_query()`,
never a provider-specific API directly.

`EmbeddingProvider` also doubles as a LangChain-compatible `Embeddings`
object (it implements the same `embed_documents` / `embed_query` method
signatures), so instances can be passed straight into
`langchain_postgres.PGVector` (see `vector_store.py`) without an extra
adapter class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import Settings


class EmbeddingProvider(ABC):
    """Interface every embedding backend must implement. Add a new
    provider (e.g. Cohere) by subclassing this and implementing both
    methods, then wiring it into `get_embedding_provider` below."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents/chunks. Use case: called once per
        document during ingestion, embedding all of its chunks in a single
        batch call (cheaper and faster than one call per chunk)."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string. Use case: called once per user
        question in `RagPipeline.answer_query`, so it can be compared
        against the stored chunk embeddings via vector similarity search."""


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Embeddings backed by Google's Gemini embedding models via
    LangChain. This is the only supported provider (`embedding_provider="gemini"`
    in config)."""

    def __init__(self, settings: Settings):
        client_kwargs: dict[str, str] = {"model": settings.gemini_embedding_model}
        api_key = settings.resolved_google_api_key()
        # Only pass google_api_key when non-empty. Passing "" overrides the
        # SDK's own GOOGLE_API_KEY/GEMINI_API_KEY env lookup, which breaks
        # deploy setups that inject the key only via platform env vars.
        if api_key:
            client_kwargs["google_api_key"] = api_key
        self._client = GoogleGenerativeAIEmbeddings(**client_kwargs)
        # NOTE: `output_dimensionality` must be passed per-call, not to the
        # constructor - the constructor silently ignores it.
        self._output_dimensionality = settings.embedding_dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Guard against an empty batch: Gemini's API doesn't need to be
        # called (and some clients error) if there's nothing to embed -
        # e.g. a document that produced zero chunks.
        if not texts:
            return []
        return self._client.embed_documents(
            texts,
            # RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY tells Gemini which side
            # of a search this text represents, which can improve
            # relevance since the model optimizes each embedding slightly
            # differently depending on its role.
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=self._output_dimensionality,
        )

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(
            text,
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=self._output_dimensionality,
        )


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Factory that returns the configured embedding provider. Currently
    only Gemini is supported. See `dependencies.py` for where this is called."""
    return GeminiEmbeddingProvider(settings)
