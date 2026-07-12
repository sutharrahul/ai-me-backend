"""Embedding provider for the portfolio chatbot.

Wraps Google Gemini's embedding model (via LangChain), or a local Ollama
embedding model for dev (see `settings.embedding_provider`), and exposes the
two methods the RAG pipeline needs: `embed_documents` for batch ingestion
and `embed_query` for query-time similarity search.

`GeminiEmbeddingProvider` also satisfies LangChain's `Embeddings` duck-type
interface (same method signatures), so it can be passed straight into any
LangChain vector-store constructor without an adapter.
"""

from __future__ import annotations

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings

from app.config import Settings


class GeminiEmbeddingProvider:
    """Embeddings backed by Google's Gemini embedding model via LangChain, or
    by a local Ollama embedding model when
    `settings.embedding_provider == "ollama"`."""

    def __init__(self, settings: Settings):
        self._use_ollama = settings.embedding_provider == "ollama"
        if self._use_ollama:
            # Ollama models have a fixed embedding size (no output_dimensionality
            # knob) — nomic-embed-text produces 768 dims, matching the default
            # `embedding_dimensions`, so the existing Qdrant collection still works.
            self._client = OllamaEmbeddings(
                model=settings.ollama_embedding_model,
                base_url=settings.ollama_base_url,
            )
        else:
            client_kwargs: dict[str, str] = {"model": settings.gemini_embedding_model}
            api_key = settings.resolved_google_api_key()
            # Only pass google_api_key when non-empty. Passing "" overrides the
            # SDK's own GOOGLE_API_KEY/GEMINI_API_KEY env lookup, which breaks
            # deploy setups that inject the key only via platform env vars.
            if api_key:
                client_kwargs["google_api_key"] = api_key
            self._client = GoogleGenerativeAIEmbeddings(**client_kwargs)
            # NOTE: `output_dimensionality` must be passed per-call, not to the
            # constructor — the constructor silently ignores it.
            self._output_dimensionality = settings.embedding_dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks. Called once per document during
        ingestion, embedding all chunks in a single batch API call."""
        if not texts:
            return []
        if self._use_ollama:
            return self._client.embed_documents(texts)
        return self._client.embed_documents(
            texts,
            # RETRIEVAL_DOCUMENT tells Gemini this text will be stored and
            # searched against, which lets the model optimise the embedding
            # for retrieval rather than for semantic similarity alone.
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=self._output_dimensionality,
        )

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user question. Called once per chat request so it
        can be compared against stored chunk embeddings via vector search."""
        if self._use_ollama:
            return self._client.embed_query(text)
        return self._client.embed_query(
            text,
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=self._output_dimensionality,
        )


def get_embedding_provider(settings: Settings) -> GeminiEmbeddingProvider:
    """Returns the embedding provider. Used by `dependencies.py` for DI."""
    return GeminiEmbeddingProvider(settings)
