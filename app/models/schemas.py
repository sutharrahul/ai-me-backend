"""Pydantic request/response schemas for the portfolio chatbot API.

FastAPI uses these models to validate incoming request bodies, serialize
outgoing responses to JSON, and auto-generate the OpenAPI docs at /docs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for `POST /api/chat/query`.

    `question` is capped at 1 000 characters so a huge pasted block of text
    can't blow up the prompt size (and cost) sent to the LLM. `top_k`
    controls how many retrieved chunks are used as context (capped at 20 to
    keep prompts reasonably sized).
    """
    session_id: str
    chat_id: str
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


class TokenEvent(BaseModel):
    """One line of the `POST /api/chat/query` stream: a chunk of the answer
    as the model generates it."""

    type: Literal["token"] = "token"
    content: str


class DoneEvent(BaseModel):
    """Final line of the `POST /api/chat/query` stream, once generation and
    persistence finish. `chat_id` always echoes back the request's
    `chat_id` (a conversation id) unchanged - it never rotates, even once
    the conversation spans multiple storage chunks (see `CHUNK_SIZE` in
    `app/db/db_query.py`)."""

    type: Literal["done"] = "done"
    chat_id: str
    chat_title: str | None
    sources: list[Source]


class ErrorEvent(BaseModel):
    """Sent in place of `DoneEvent` if answering fails partway through the
    stream - by then a 200 has already been sent, so failure can't be
    reported as an HTTP error status."""

    type: Literal["error"] = "error"
    message: str


class ChatHistoryMessage(BaseModel):
    """One stored message, as persisted in `Chat.messages`."""

    role: str
    message: str
    created_at: str


class ChatSummary(BaseModel):
    """One conversation's metadata (one row per conversation, not per
    storage chunk), used to list a session's recent conversations in the
    sidebar without fetching every message. `chat_title` is always a
    non-empty string here - `list_recent_chats` (see `app/db/db_query.py`)
    excludes untitled conversations."""

    chat_id: str
    chat_title: str
    created_at: datetime


class ChatDetailResponse(BaseModel):
    """Response body for `GET /api/chat/{chat_id}`: a conversation's most
    recent chunk, used to open a chat picked from the sidebar's history
    list. `previous_chunk_id` is set when the conversation has older
    chunks - pass it to `GET /chat/chunk/{chunk_id}` to load them (see
    `ChunkMessagesResponse`)."""

    chat_id: str
    chat_title: str | None
    messages: list[ChatHistoryMessage]
    previous_chunk_id: str | None


class ChunkMessagesResponse(BaseModel):
    """Response body for `GET /api/chat/chunk/{chunk_id}`: one older chunk's
    messages, fetched on demand when the user clicks "Load older messages".
    `previous_chunk_id` is set if there's still an even older chunk to load."""

    messages: list[ChatHistoryMessage]
    previous_chunk_id: str | None


class HealthResponse(BaseModel):
    """Response body for `GET /api/health` — confirms the API process is up."""

    status: str
    app_name: str
    environment: str
