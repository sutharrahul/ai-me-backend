"""Embedding provider abstraction.

An "embedding" is a list of floats representing the meaning of a piece of
text, used to find semantically similar chunks via vector search. Wrapping
the embedding call behind the `EmbeddingProvider` interface makes it easy
to swap providers (Gemini, OpenAI, Cohere, local models, etc.) without
touching the rest of the pipeline - `RagPipeline` and `VectorStore` only
ever call `.embed_documents()`/`.embed_query()`, never a provider-specific
API directly.

`EmbeddingProvider` also doubles as a LangChain-compatible `Embeddings`
object (it implements the same `embed_documents` / `embed_query` method
signatures), so instances can be passed straight into
`langchain_postgres.PGVector` (see `vector_store.py`) without an extra
adapter class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from openai import OpenAI

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
    LangChain. This is the default provider (`embedding_provider="gemini"`
    in config)."""

    def __init__(self, settings: Settings):
        self._client = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.google_api_key,
        )
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


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embeddings backed by OpenAI's embedding models. Used when
    `embedding_provider="openai"` in config - e.g. if you don't have a
    Gemini key, or prefer OpenAI's models."""

    def __init__(self, settings: Settings):
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        # OpenAI's embeddings API has no separate "query" mode like
        # Gemini's task_type, so a single query is just embedded the same
        # way as a one-item batch of documents.
        return self.embed_documents([text])[0]


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embeddings backed by a local Ollama model via LangChain. Used when
    `embedding_provider="ollama"` in config - handy for testing without an
    API key/cost. Requires `ollama serve` running locally with the model
    already pulled (e.g. `ollama pull nomic-embed-text`)."""

    def __init__(self, settings: Settings):
        self._client = OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Factory that returns the configured embedding provider. This is the
    single place that decides Gemini vs OpenAI vs Ollama based on
    `settings.embedding_provider`, so the rest of the app never needs an
    if/else on the provider name - see `dependencies.py`."""
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(settings)
    if settings.embedding_provider == "ollama":
        return OllamaEmbeddingProvider(settings)
    return GeminiEmbeddingProvider(settings)
