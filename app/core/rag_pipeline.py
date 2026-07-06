"""Orchestrates document ingestion and retrieval-augmented generation.

`RagPipeline` is the single object that ties every service together:
- `document_loader` + `text_splitter` to turn a file into text chunks
- `embeddings` to turn text into vectors
- `vector_store` to persist/search those vectors
- `llm` to turn (question + retrieved chunks) into an answer
- `document_store` to track ingestion status/metadata per document

Both the API (`api/routes/chat.py`) and the CLI ingestion script
(`scripts/ingest_documents.py`) go through this class rather than calling
the individual services directly, so ingestion/query logic lives in one
place and behaves identically no matter which entrypoint triggered it.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from app.config import Settings
from app.models.schemas import DocumentStatus, QueryResponse, Source
from app.services.cache import QueryCache
from app.services.document_loader import load_text
from app.services.document_store import DocumentStore
from app.services.embeddings import EmbeddingProvider
from app.services.llm import LLMClient
from app.services.text_splitter import split_text
from app.services.vector_store import Chunk, VectorStoreBackend
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RagPipeline:
    def __init__(
        self,
        settings: Settings,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStoreBackend,
        llm_client: LLMClient,
        document_store: DocumentStore,
        query_cache: QueryCache,
    ):
        # All dependencies are injected (rather than constructed here) so
        # this class stays easy to unit test and so `dependencies.py` can
        # control which concrete provider (Gemini vs OpenAI, etc.) is used.
        self._settings = settings
        self._embeddings = embedding_provider
        self._vector_store = vector_store
        self._llm = llm_client
        self._documents = document_store
        self._cache = query_cache

    def ingest_document(self, document_id: str, filename: str, file_path: Path) -> None:
        """Loads a local file's text and ingests it via `ingest_text`.

        Use case: called once per file, by `scripts/ingest_documents.py`
        for files dropped into `data/seed/`. Extracting the text
        (`document_loader.load_text`, supports .txt/.md/.pdf) is the only
        thing this does beyond `ingest_text` - see that method for the
        chunk/embed/store steps, shared with `/api/documents/ingest-url`
        (which already has its text, fetched from a remote API).

        Any exception is caught and logged rather than raised, because this
        is often called from a batch script looping over many files, where
        one bad file shouldn't crash the whole run.
        """
        try:
            text = load_text(file_path)
        except Exception:
            logger.exception("Failed to read %s", filename)
            self._documents.update_status(document_id, DocumentStatus.FAILED)
            return
        self.ingest_text(document_id, filename, text)

    def ingest_text(self, document_id: str, filename: str, text: str) -> None:
        """Chunks, embeds, and stores a document's already-extracted text.

        Use case: the shared second half of ingestion, used both by
        `ingest_document` (text read from a local file) and
        `api/routes/documents.py`'s `/ingest-url` endpoint (text fetched
        from a remote API).

        Steps:
        1. Split the text into overlapping chunks small enough to embed
           and to fit as LLM context later (`text_splitter.split_text`).
        2. Embed every chunk in one batch call (cheaper/faster than one
           call per chunk).
        3. Store the chunks + embeddings in the vector store, tagged with
           `document_id`/`filename`/`chunk_index` metadata so sources can
           be traced back to the original document later.
        4. Update the document's status in `DocumentStore` to `ready` (or
           `failed`/left as `processing` if something goes wrong), so
           callers can poll/inspect ingestion state.

        Any exception is caught and logged rather than raised - see
        `ingest_document` above for why.
        """
        try:
            chunks_text = split_text(
                text,
                chunk_size=self._settings.chunk_size,
                chunk_overlap=self._settings.chunk_overlap,
            )

            if not chunks_text:
                # e.g. an empty file, or a PDF with no extractable text.
                self._documents.update_status(document_id, DocumentStatus.FAILED)
                logger.warning("No extractable text in %s", filename)
                return

            embeddings = self._embeddings.embed_documents(chunks_text)
            chunks = [
                # `id` is deterministic (`{document_id}-{index}`) so
                # `delete_document` below can reconstruct every chunk id for
                # a document without needing to query the vector store first.
                Chunk(
                    id=f"{document_id}-{idx}",
                    document_id=document_id,
                    filename=filename,
                    chunk_index=idx,
                    content=chunk_text,
                )
                for idx, chunk_text in enumerate(chunks_text)
            ]
            self._vector_store.add_chunks(chunks, embeddings)
            self._documents.update_status(
                document_id, DocumentStatus.READY, chunk_count=len(chunks)
            )
            logger.info("Ingested %s into %d chunks", filename, len(chunks))
        except Exception:
            logger.exception("Failed to ingest %s", filename)
            self._documents.update_status(document_id, DocumentStatus.FAILED)

    def delete_document(self, document_id: str) -> None:
        """Removes a document's chunks from the vector store and its entry
        from the document index. Use case: re-ingesting a file that
        changed (delete the old version's chunks, then ingest fresh) - see
        how `scripts/ingest_documents.py` calls this before re-ingesting.

        Chunk ids are reconstructed from `chunk_count` (rather than looked
        up) because pgvector's `delete()` only supports deleting by exact
        id, not by a metadata filter like "all chunks for this document".
        """
        document = self._documents.get(document_id)
        if document is not None and document.chunk_count:
            ids = [f"{document_id}-{idx}" for idx in range(document.chunk_count)]
            self._vector_store.delete_chunks(ids)
        self._documents.delete(document_id)

    def clear_query_cache(self) -> None:
        """Drops every cached answer. Use case: called by
        `scripts/ingest_documents.py` after (re-)ingesting documents, so
        visitors stop getting answers grounded in since-replaced content
        the moment new content goes live, rather than waiting up to
        `query_cache_ttl_seconds` for stale entries to expire on their
        own."""
        self._cache.clear()

    def answer_query(self, question: str, top_k: int) -> QueryResponse:
        """Answers a question using retrieval-augmented generation.

        Use case: this is what `POST /api/chat/query` calls for every
        message sent from the chat UI.

        0. Check the query cache (`services/cache.py`) first, keyed on the
           normalized question + top_k. On a hit, return immediately -
           skipping both the embedding call and the LLM call below, which
           are the only two things that actually cost API quota here.
        1. Embed the question with the same embedding model used for
           ingestion (embeddings must come from the same model/space to be
           comparable).
        2. Retrieve the `top_k` most similar chunks from the vector store.
        3. Ask the LLM to answer using only those chunks as context
           (grounding the answer, reducing hallucination).
        4. Cache the freshly generated answer, then return it plus the
           source chunks used, so the UI can show "N sources" and let the
           user inspect them.
        """
        cached = self._cache.get(question, top_k)
        if cached is not None:
            logger.info("Cache hit for question (top_k=%d)", top_k)
            return cached

        query_embedding = self._embeddings.embed_query(question)
        chunks = self._vector_store.query(query_embedding, top_k=top_k)
        answer = self._llm.generate_answer(question, chunks)
        sources = [
            Source(
                document_id=c.document_id,
                filename=c.filename,
                chunk_index=c.chunk_index,
                content=c.content,
                score=c.score,
            )
            for c in chunks
        ]
        response = QueryResponse(answer=answer, sources=sources)
        self._cache.set(question, top_k, response)
        return response


def new_document_id() -> str:
    """Generates a short, URL-safe, unique-enough id for a newly ingested
    document (12 hex chars from a UUID4). Used as the `document_id` that
    ties together the `DocumentStore` entry, the vector store's chunk ids
    (`{document_id}-{index}`), and any future API responses referencing it."""
    return uuid.uuid4().hex[:12]
