"""Pydantic request/response schemas for the portfolio chatbot API.

FastAPI uses these models to validate incoming request bodies, serialize
outgoing responses to JSON, and auto-generate the OpenAPI docs at /docs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for `POST /api/chat/query`.

    `question` is capped at 1 000 characters so a huge pasted block of text
    can't blow up the prompt size (and cost) sent to the LLM. `top_k`
    controls how many retrieved chunks are used as context (capped at 20 to
    keep prompts reasonably sized).
    """
    session_id: str
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=20)


class Source(BaseModel):
    """One retrieved chunk used to answer a question, returned alongside the
    answer so the frontend can show "N sources" and let the user inspect
    exactly which content backed the response."""

    document_id: str
    filename: str
    chunk_index: int
    content: str
    score: float


class QueryResponse(BaseModel):
    """Response body for `POST /api/chat/query`: the generated answer plus
    the list of sources that were retrieved and passed to the LLM as context."""

    answer: str
    sources: list[Source]


class HealthResponse(BaseModel):
    """Response body for `GET /api/health` — confirms the API process is up."""

    status: str
    app_name: str
    environment: str
