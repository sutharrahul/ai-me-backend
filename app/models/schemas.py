"""Pydantic request/response schemas shared across API routes.

FastAPI uses these models to: validate incoming request bodies, serialize
outgoing responses to JSON, and auto-generate the OpenAPI docs at /docs.
Keep API-facing shapes here rather than reusing internal dataclasses (like
`Chunk`/`ScoredChunk` in `vector_store.py`), so the public API contract can
evolve independently of internal implementation details.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl


class DocumentStatus(str, Enum):
    """Lifecycle states for an ingested document, tracked in
    `DocumentStore`. `PROCESSING` is set immediately when ingestion starts,
    then flips to `READY` on success or `FAILED` on error - see
    `RagPipeline.ingest_document`."""

    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DocumentItem(BaseModel):
    """Metadata for one ingested document (not its content/chunks, which
    live in the vector store). Used by `DocumentStore` to persist the
    index and by the ingestion script to check what's already been
    ingested and how many chunks it produced."""

    id: str
    filename: str
    status: DocumentStatus
    chunk_count: int = 0
    created_at: datetime


class IngestUrlRequest(BaseModel):
    """Request body for `POST /api/documents/ingest-url`. Fetches `url`
    (sending `api_key` in the `header_name` request header) and ingests
    the response body as a new document, the same way a file dropped into
    `data/seed/` would be."""

    url: HttpUrl
    header_name: str = Field(..., min_length=1, max_length=200)
    api_key: str = Field(..., min_length=1, max_length=1000)
    filename: str = Field(..., min_length=1, max_length=255)


class QueryRequest(BaseModel):
    """Request body for `POST /api/chat/query`. `question` is required and
    non-empty, and capped at 1000 characters so a huge pasted block of text
    can't blow up the prompt size (and cost) sent to the LLM; `top_k`
    controls how many retrieved chunks are used as context (higher = more
    context but a longer/noisier prompt to the LLM, capped at 20 to keep
    prompts reasonably sized)."""

    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=20)


class Source(BaseModel):
    """One retrieved chunk used to answer a question, returned alongside
    the answer so the frontend can show "N sources" and let the user
    inspect exactly which content backed the response (see
    `SourcesList.tsx` on the frontend)."""

    document_id: str
    filename: str
    chunk_index: int
    content: str
    score: float


class QueryResponse(BaseModel):
    """Response body for `POST /api/chat/query`: the generated answer plus
    the list of sources that were retrieved and passed to the LLM as
    context."""

    answer: str
    sources: list[Source]


class HealthResponse(BaseModel):
    """Response body for `GET /api/health` - a minimal payload confirming
    the API process is up and reporting which app/environment it is."""

    status: str
    app_name: str
    environment: str
