"""Qdrant-backed vector store for the portfolio chatbot.

Chunk text and pre-computed embeddings are stored in Qdrant and searched
at query time. Embeddings are computed upstream (see `embeddings.py`) and
passed in directly via `add_chunks`, so Qdrant is never asked to embed
anything itself.

Qdrant point IDs must be unsigned ints or UUIDs. Our chunk IDs are strings
like `{filename}-{index}`, so `_point_id` derives a deterministic UUID5
from that string — the same chunk ID always maps to the same point, which
lets `delete_chunks` regenerate point IDs without a separate lookup table.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from app.config import Settings


@dataclass
class Chunk:
    """A single piece of a document, ready to be embedded and stored.
    Produced by `RagPipeline.ingest_document` after splitting the document's
    text with `text_splitter.split_text`."""

    id: str
    document_id: str
    filename: str
    chunk_index: int
    content: str


@dataclass
class ScoredChunk(Chunk):
    """A `Chunk` returned from a similarity search, with a relevance `score`
    (higher = more similar). Used by `llm.py` to build the context block and
    by `schemas.Source` to report citations to the frontend."""

    score: float


class QdrantVectorStore:
    """Vector store backed by Qdrant via its low-level `QdrantClient`.

    Uses the client directly (rather than LangChain's wrapper) so
    pre-computed embeddings can be upserted and queried without the wrapper
    re-embedding text a second time.
    """

    def __init__(self, settings: Settings):
        self._client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value() or None,
            check_compatibility=False,
        )
        self._collection_name = settings.vector_collection_name
        self._ensure_collection(settings.embedding_dimensions)

    def _ensure_collection(self, dimensions: int) -> None:
        """Creates the Qdrant collection on first use if it doesn't exist,
        so a fresh Qdrant container works out of the box without manual setup."""
        if not self._client.collection_exists(self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=dimensions, distance=qdrant_models.Distance.COSINE
                ),
            )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """Derives a deterministic UUID5 from a string chunk ID so Qdrant's
        UUID-typed point IDs stay stable across re-ingestion."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Upserts a batch of chunks and their pre-computed embeddings."""
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
        """Finds the `top_k` chunks most similar to `embedding`.
        Called once per user question in `RagPipeline.answer_query`."""
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
                    # Qdrant's cosine score from `query_points` is already a
                    # similarity value (higher = more similar), unlike
                    # pgvector's cosine distance — no conversion needed.
                    score=point.score,
                )
            )
        return scored_chunks

    def delete_chunks(self, ids: list[str]) -> None:
        """Deletes chunks by their string chunk IDs. Used by the ingestion
        script when force-re-ingesting a file to clear stale vectors first."""
        if not ids:
            return
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=qdrant_models.PointIdsList(
                points=[self._point_id(chunk_id) for chunk_id in ids]
            ),
        )


def get_vector_store(settings: Settings) -> QdrantVectorStore:
    """Returns the Qdrant vector store. Used by `dependencies.py` for DI."""
    return QdrantVectorStore(settings)
