"""Simple JSON-backed store for document metadata.

This tracks *metadata about* ingested documents (id, filename, status,
chunk count, created_at) - not the document content/chunks themselves,
which live in the vector store (`vector_store.py`). It's what lets
`scripts/ingest_documents.py` answer "has this file already been
ingested?" and lets `RagPipeline.delete_document` know how many chunk ids
to delete.

For a boilerplate this avoids pulling in a full database. Swap this out
for Postgres/SQLite-backed persistence as the project grows (e.g. a
`documents` table in the same Postgres instance already used for vectors).
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.models.schemas import DocumentItem, DocumentStatus


class DocumentStore:
    def __init__(self, index_path: Path):
        # `index_path` is the JSON file on disk (see
        # `settings.documents_index_path`, defaults to data/documents.json).
        # A `threading.Lock` guards every read-modify-write so concurrent
        # requests (or a background ingestion task running alongside a
        # request) don't corrupt the file with interleaved writes.
        self._path = index_path
        self._lock = threading.Lock()
        if not self._path.exists():
            self._write([])

    def list_all(self) -> list[DocumentItem]:
        """Returns every tracked document. Use case: the ingestion script
        uses this to check whether a given filename has already been
        ingested before deciding whether to (re-)process it."""
        return [DocumentItem(**item) for item in self._read()]

    def get(self, document_id: str) -> DocumentItem | None:
        """Looks up a single document by id, or `None` if it doesn't
        exist. Use case: `RagPipeline.delete_document` calls this to find
        out how many chunks a document has before deleting them."""
        for item in self._read():
            if item["id"] == document_id:
                return DocumentItem(**item)
        return None

    def create(self, document_id: str, filename: str) -> DocumentItem:
        """Registers a new document with status `PROCESSING` and
        `chunk_count=0`, before ingestion actually runs. Use case: called
        right before `RagPipeline.ingest_document`, so there's always a
        record even if ingestion later fails partway through."""
        doc = DocumentItem(
            id=document_id,
            filename=filename,
            status=DocumentStatus.PROCESSING,
            chunk_count=0,
            created_at=datetime.now(timezone.utc),
        )
        with self._lock:
            items = self._read()
            items.append(json.loads(doc.model_dump_json()))
            self._write(items)
        return doc

    def update_status(
        self, document_id: str, status: DocumentStatus, chunk_count: int | None = None
    ) -> None:
        """Updates a document's status (and optionally its chunk count)
        after ingestion finishes. Use case: `RagPipeline.ingest_document`
        calls this with `READY` + the final chunk count on success, or
        `FAILED` (leaving chunk_count untouched) on error."""
        with self._lock:
            items = self._read()
            for item in items:
                if item["id"] == document_id:
                    item["status"] = status.value
                    if chunk_count is not None:
                        item["chunk_count"] = chunk_count
            self._write(items)

    def delete(self, document_id: str) -> None:
        """Removes a document's metadata entry entirely. Use case: called
        by `RagPipeline.delete_document` after its vector chunks have
        already been removed from the vector store, so the index doesn't
        reference chunks that no longer exist."""
        with self._lock:
            items = [i for i in self._read() if i["id"] != document_id]
            self._write(items)

    def _read(self) -> list[dict]:
        """Loads the raw list of document records from disk. Returns an
        empty list if the file doesn't exist yet (shouldn't normally
        happen since `__init__` creates it, but kept defensive)."""
        if not self._path.exists():
            return []
        with self._path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, items: list[dict]) -> None:
        """Overwrites the JSON file with the given list of records.
        `default=str` handles non-JSON-native types (like `datetime`)
        that might slip through without being pre-serialized."""
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, default=str)
