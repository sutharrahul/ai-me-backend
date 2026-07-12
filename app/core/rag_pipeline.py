"""Orchestrates document ingestion for the RAG pipeline.

`RagPipeline` ties together every service:
- `document_loader` + `text_splitter` to turn a file into text chunks
- `embeddings` to turn text chunks into vectors
- `vector_store` to persist / search those vectors
- `cache` to invalidate cached query responses after re-ingestion

The CLI ingestion script (`scripts/ingest_documents.py`) uses `ingest_document`,
`delete_document_chunks`, and `clear_query_cache`.
"""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.services.cache import QueryCache
from app.services.document_loader import load_text
from app.services.embeddings import GeminiEmbeddingProvider
from app.services.text_splitter import split_text
from app.services.vector_store import Chunk, QdrantVectorStore
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RagPipeline:
    def __init__(
        self,
        settings: Settings,
        embedding_provider: GeminiEmbeddingProvider,
        vector_store: QdrantVectorStore,
        query_cache: QueryCache,
    ):
        self._settings = settings
        self._embeddings = embedding_provider
        self._vector_store = vector_store
        self._cache = query_cache

    # ------------------------------------------------------------------
    # Ingestion (CLI only — not exposed via the API)
    # ------------------------------------------------------------------

    def ingest_document(self, file_path: Path) -> None:
        """Load a document from disk, chunk it, embed it, and upsert it into
        the vector store. Filename is used as the document identifier so
        chunk IDs are deterministic and stable across re-ingestion runs."""
        filename = file_path.name

        try:
            text = load_text(file_path)
            chunks_text = split_text(
                text,
                chunk_size=self._settings.chunk_size,
                chunk_overlap=self._settings.chunk_overlap,
            )

            if not chunks_text:
                logger.warning("No extractable text found in %s — skipping", filename)
                return

            embeddings = self._embeddings.embed_documents(chunks_text)

            chunks = [
                Chunk(
                    id=f"{filename}-{idx}",
                    document_id=filename,
                    filename=filename,
                    chunk_index=idx,
                    content=chunk,
                )
                for idx, chunk in enumerate(chunks_text)
            ]

            self._vector_store.add_chunks(chunks, embeddings)

            logger.info(
                "Ingested '%s' → %d chunks stored",
                filename,
                len(chunks),
            )

        except Exception:
            logger.exception("Failed to ingest %s", filename)

    def delete_document_chunks(self, filename: str, chunk_count: int) -> None:
        """Remove all vector chunks for a document identified by `filename`.
        Called by the ingestion script before force-re-ingesting a file."""
        ids = [f"{filename}-{i}" for i in range(chunk_count)]
        self._vector_store.delete_chunks(ids)
        logger.info("Deleted %d chunks for '%s'", len(ids), filename)

    def clear_query_cache(self) -> None:
        """Clear every cached query response. Called by the ingestion script
        after content changes so stale cached answers are invalidated."""
        self._cache.clear()
