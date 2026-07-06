"""Persistent vector store abstraction, backed by either PostgreSQL
(pgvector) or Qdrant.

This is where chunk text + embeddings actually live, and where similarity
search happens at query time. `VectorStoreBackend` is the interface both
implementations satisfy (mirroring `EmbeddingProvider`/`LLMClient` in
`embeddings.py`/`llm.py`), so `RagPipeline` never needs to know which
concrete database is active - only `get_vector_store` below (and
`dependencies.py`, which calls it) decides that, based on
`settings.vector_store_provider`.

Embeddings are computed upstream (see `embeddings.py`) and passed in
directly via `add_chunks`, so neither backend is ever asked to embed
anything itself here.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_postgres import PGVector
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from app.config import Settings
from app.services.embeddings import EmbeddingProvider


@dataclass
class Chunk:
    """A single piece of a document, ready to be embedded and stored.
    Produced by `RagPipeline.ingest_document` after splitting a document's
    text with `text_splitter.split_text`."""

    id: str
    document_id: str
    filename: str
    chunk_index: int
    content: str


@dataclass
class ScoredChunk(Chunk):
    """A `Chunk` returned from a similarity search, with a `score`
    (0 = unrelated, 1 = identical) indicating how relevant it is to the
    query. Used by `llm.py` to build the context sent to the model and by
    `schemas.Source` to report citations back to the frontend."""

    score: float


class VectorStoreBackend(ABC):
    """Interface every vector store backend must implement. Add a new
    backend (e.g. Pinecone, Weaviate) by subclassing this and wiring it
    into `get_vector_store` below."""

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Persists a batch of chunks and their pre-computed embeddings.
        Use case: called once per document at the end of
        `RagPipeline.ingest_document`, after all of that document's chunks
        have been embedded together. `chunks` and `embeddings` must be the
        same length and in the same order."""

    @abstractmethod
    def query(self, embedding: list[float], top_k: int = 4) -> list[ScoredChunk]:
        """Finds the `top_k` chunks most similar to `embedding`. Use case:
        called once per user question in `RagPipeline.answer_query`, after
        the question itself has been embedded."""

    @abstractmethod
    def delete_chunks(self, ids: list[str]) -> None:
        """Deletes chunks by their exact ids. Use case: called by
        `RagPipeline.delete_document` when removing/re-ingesting a
        document - the caller must know all the chunk ids up front (which
        it can, since they're deterministic - see `Chunk.id`)."""


class PostgresVectorStore(VectorStoreBackend):
    """Vector store backed by PostgreSQL + pgvector via LangChain's
    `PGVector`, which handles the SQL/schema details (tables, the `vector`
    column, indexes) for us - this class just adapts our own
    `Chunk`/`ScoredChunk` types to/from LangChain's `Document` type so the
    rest of the app never has to import LangChain directly. This is the
    default backend (`vector_store_provider="postgres"` in config)."""

    def __init__(self, settings: Settings, embedding_provider: EmbeddingProvider):
        # `PGVector` is typed to accept a LangChain `Embeddings` object, but
        # only calls `.embed_query`/`.embed_documents` on it - our
        # `EmbeddingProvider` already implements that same interface, so it
        # can be passed straight through.
        self._store = PGVector(
            embeddings=embedding_provider,  # type: ignore[arg-type]
            collection_name=settings.vector_collection_name,
            connection=settings.database_url,
            embedding_length=settings.embedding_dimensions,
            # Store metadata as JSONB rather than fixed columns, so we can
            # add new metadata fields (per chunk) later without a schema
            # migration.
            use_jsonb=True,
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._store.add_embeddings(
            texts=[c.content for c in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "document_id": c.document_id,
                    "filename": c.filename,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
            # Using our own deterministic ids (`{document_id}-{index}`,
            # see rag_pipeline.py) instead of letting PGVector generate
            # random ones lets us delete a document's chunks later without
            # having to look them up first.
            ids=[c.id for c in chunks],
        )

    def query(self, embedding: list[float], top_k: int = 4) -> list[ScoredChunk]:
        results = self._store.similarity_search_with_score_by_vector(
            embedding, k=top_k
        )

        scored_chunks: list[ScoredChunk] = []
        for doc, distance in results:
            # Cosine distance (0 = identical, 2 = opposite) -> similarity in [0, 1].
            score = max(0.0, 1 - distance / 2)
            metadata = doc.metadata
            scored_chunks.append(
                ScoredChunk(
                    id=doc.id or "",
                    document_id=metadata["document_id"],
                    filename=metadata["filename"],
                    chunk_index=metadata["chunk_index"],
                    content=doc.page_content,
                    score=score,
                )
            )
        return scored_chunks

    def delete_chunks(self, ids: list[str]) -> None:
        if not ids:
            return
        self._store.delete(ids=ids)


class QdrantVectorStoreBackend(VectorStoreBackend):
    """Vector store backed by Qdrant via its low-level `QdrantClient`
    (rather than LangChain's `QdrantVectorStore` wrapper), since embeddings
    are already computed upstream and this way we can upsert/query
    pre-computed vectors directly instead of letting a wrapper re-embed
    text itself. Used when `vector_store_provider="qdrant"` in config -
    e.g. for local testing via `docker compose up -d qdrant`.

    Qdrant point ids must be an unsigned int or a UUID, but our own chunk
    ids are strings like `{document_id}-{index}` - `_point_id` derives a
    deterministic UUID5 from that string so the same chunk id always maps
    to the same point, letting `delete_chunks` regenerate ids to delete
    without needing a separate id-mapping lookup. The original chunk id is
    also kept in the payload for completeness/debugging.
    """

    def __init__(self, settings: Settings):
        self._client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            check_compatibility=False
        )
        self._collection_name = settings.vector_collection_name
        self._ensure_collection(settings.embedding_dimensions)

    def _ensure_collection(self, dimensions: int) -> None:
        """Creates the collection on first use if it doesn't exist yet,
        so a fresh Qdrant container (no manual setup) works out of the
        box - mirroring how `PGVector` auto-creates its tables."""
        if not self._client.collection_exists(self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=dimensions, distance=qdrant_models.Distance.COSINE
                ),
            )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        points = [
            qdrant_models.PointStruct(
                id=self._point_id(chunk.id),
                vector=vector,
                payload={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                },
            )
            for chunk, vector in zip(chunks, embeddings)
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)

    def query(self, embedding: list[float], top_k: int = 4) -> list[ScoredChunk]:
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=embedding,
            limit=top_k,
            with_payload=True,
        ).points

        scored_chunks: list[ScoredChunk] = []
        for point in results:
            payload = point.payload or {}
            scored_chunks.append(
                ScoredChunk(
                    id=payload.get("chunk_id", str(point.id)),
                    document_id=payload["document_id"],
                    filename=payload["filename"],
                    chunk_index=payload["chunk_index"],
                    content=payload["content"],
                    # Qdrant's cosine "score" from `query_points` is
                    # already a similarity (higher = more similar), unlike
                    # pgvector's distance - no conversion needed.
                    score=point.score,
                )
            )
        return scored_chunks

    def delete_chunks(self, ids: list[str]) -> None:
        if not ids:
            return
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=qdrant_models.PointIdsList(
                points=[self._point_id(chunk_id) for chunk_id in ids]
            ),
        )


def get_vector_store(
    settings: Settings, embedding_provider: EmbeddingProvider
) -> VectorStoreBackend:
    """Factory that returns the configured vector store backend. This is
    the single place that decides Postgres vs Qdrant based on
    `settings.vector_store_provider`, mirroring `get_embedding_provider` in
    `embeddings.py` and `get_llm_client` in `llm.py` - see
    `dependencies.py` for where it's called."""
    if settings.vector_store_provider == "qdrant":
        return QdrantVectorStoreBackend(settings)
    return PostgresVectorStore(settings, embedding_provider)
