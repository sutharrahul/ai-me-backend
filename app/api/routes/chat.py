"""Chat/query endpoint for the portfolio chatbot.

Validates incoming requests, classifies intent and (when needed) retrieves
context, then streams the generated answer back to the client so it can be
rendered token-by-token, persisting the exchange once the stream ends.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from slowapi.util import get_remote_address

from app.config import get_settings
from app.core.rate_limiter import limiter
from app.models.schemas import (
    ChatDetailResponse,
    ChatSummary,
    ChunkMessagesResponse,
    DoneEvent,
    ErrorEvent,
    QueryRequest,
    Source,
    TokenEvent,
)
from app.utils.logger import get_logger
from sqlalchemy.orm import Session
from app.db.db_connection import get_db
from app.db.db_query import (
    get_active_chunk,
    get_chunk_by_chunk_id,
    list_recent_chats,
    store_exchange,
)
from app.graph.node import classify_intent, llm_bot, retrieve_chunks, route_intent
from app.services.llm import fix_first_person
from app.graph.system_prompt import (
    GREETING_SYSTEM_PROMPT,
    SMALL_TALK_SYSTEM_PROMPT,
    UNKNOWN_SYSTEM_PROMPT,
)

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)

# Built from `settings.daily_question_limit` (default 20) rather than a
# hardcoded string, so changing DAILY_QUESTION_LIMIT in .env is enough to
# retune it - no code change needed. `slowapi`'s decorator needs this
# string at import time, so it reads settings once, at module load.
_DAILY_LIMIT = f"{get_settings().daily_question_limit}/day"

# System prompt for every route except "retrieve_chunks", which needs the
# retrieved chunks folded into the prompt (see `stream_answer` in
# `app/services/llm.py`) rather than a fixed system prompt.
_ROUTE_SYSTEM_PROMPTS = {
    "greeting": GREETING_SYSTEM_PROMPT,
    "small_talks": SMALL_TALK_SYSTEM_PROMPT,
    "unknown": UNKNOWN_SYSTEM_PROMPT,
}

# How many of the conversation's most recent messages (across both roles)
# to feed back into classification/generation, so short follow-ups like
# "yes" or "tell me more" are understood in context. Kept small - it's
# added to every classification and generation call in the exchange, so a
# larger window means slower, pricier requests for context that's rarely
# still relevant much further back.
HISTORY_MESSAGE_LIMIT = 6

# Safety-net cap (characters) for buffering the answer's opening before the
# first bullet point - see the intro-buffering loop in `query()` below. Most
# answers reach a bullet or paragraph break well before this; it only
# matters for a bullet-free answer, where it bounds how long streaming is
# delayed before flushing regardless.
INTRO_BUFFER_MAX_CHARS = 300


def _intro_is_complete(buffer: str) -> bool:
    """True once `buffer` has reached a safe point to stop buffering and
    flush - the start of a markdown bullet list or a paragraph break.
    Every observed case of the model drifting into third person (see
    `fix_first_person`) happens in the answer's opening sentence(s), never
    inside a bullet list, so once we're past that opening we can stream
    the rest through untouched."""
    return "\n*" in buffer or "\n-" in buffer or "\n\n" in buffer


@router.post("/query")
@limiter.limit(_DAILY_LIMIT)
def query(
    request: Request,
    payload: QueryRequest,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """
    Answers a user's question, streaming the answer back as newline-delimited
    JSON - one `TokenEvent`/`DoneEvent`/`ErrorEvent` (see `app/models/schemas.py`)
    per line - so the frontend can render it token-by-token instead of
    waiting for the full answer.

    Classification and retrieval are single, fast calls made up front; only
    the final answer generation is actually streamed. The full answer is
    persisted via `store_exchange` once the stream ends.
    """
    ip_address = get_remote_address(request)

    def event_stream():
        try:
            active_chunk = get_active_chunk(db, conversation_id=payload.chat_id)
            history = (
                active_chunk.messages[-HISTORY_MESSAGE_LIMIT:] if active_chunk else []
            )

            state = {"user_query": payload.question, "history": history}
            classify_intent(state)
            route = route_intent(state)

            sources: list[Source] = []
            if route == "retrieve_chunks":
                state["top_k"] = payload.top_k
                retrieve_chunks(state)
                sources = [
                    Source(
                        document_id=chunk.document_id,
                        filename=chunk.filename,
                        chunk_index=chunk.chunk_index,
                        content=chunk.content,
                        score=chunk.score,
                    )
                    for chunk in state["chunks"]
                ]
                token_iter = llm_bot.stream_answer(
                    question=payload.question, chunks=state["chunks"], history=history
                )
            else:
                token_iter = llm_bot.stream_response(
                    user_query=payload.question,
                    system_prompt=_ROUTE_SYSTEM_PROMPTS[route],
                    history=history,
                )

            # The model occasionally drifts into third person ("Rahul
            # has...") despite PORTFOLIO_SYSTEM_PROMPT's instructions,
            # confined to the answer's opening sentence(s) - buffer just
            # that opening (not visibly streamed yet), run it through
            # `fix_first_person` once complete, then stream everything
            # after it through untouched at full token-by-token speed.
            full_answer = ""
            intro_buffer = ""
            intro_flushed = False

            for token in token_iter:
                if not intro_flushed:
                    intro_buffer += token
                    if (
                        _intro_is_complete(intro_buffer)
                        or len(intro_buffer) > INTRO_BUFFER_MAX_CHARS
                    ):
                        corrected = fix_first_person(intro_buffer)
                        full_answer += corrected
                        yield TokenEvent(content=corrected).model_dump_json() + "\n"
                        intro_flushed = True
                else:
                    full_answer += token
                    yield TokenEvent(content=token).model_dump_json() + "\n"

            if not intro_flushed:
                # The answer ended before any checkpoint was reached (e.g.
                # a short reply with no bullets at all) - flush it now.
                corrected = fix_first_person(intro_buffer)
                full_answer += corrected
                yield TokenEvent(content=corrected).model_dump_json() + "\n"

            saved_chunk = store_exchange(
                db=db,
                session_id=payload.session_id,
                conversation_id=payload.chat_id,
                user_query=payload.question,
                llm_response=full_answer,
                llm_client=llm_bot,
                ip_address=ip_address,
            )

            yield DoneEvent(
                chat_id=payload.chat_id,
                chat_title=saved_chunk.chat_title,
                sources=sources,
            ).model_dump_json() + "\n"

        except ChatGoogleGenerativeAIError as exc:
            logger.exception("Gemini API error (top_k=%d)", payload.top_k)
            if getattr(exc.__cause__, "code", None) == 429:
                message = "The AI assistant has hit its usage limit for now. Please try again in a bit."
            else:
                message = "The AI assistant is temporarily unavailable. Please try again shortly."
            yield ErrorEvent(message=message).model_dump_json() + "\n"
        except Exception as exc:  # pragma: no cover - surfaced to the client
            logger.exception("Failed to answer question (top_k=%d)", payload.top_k)
            yield ErrorEvent(message=str(exc)).model_dump_json() + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@router.get("/list", response_model=list[ChatSummary])
def list_chats(session_id: str, db: Session = Depends(get_db)) -> list[ChatSummary]:
    """Returns one entry per conversation from the last 7 days, newest
    first, for the sidebar's history list - a conversation spanning many
    storage chunks (see `CHUNK_SIZE` in `app/db/db_query.py`) still lists
    as a single item. Declared before `/{chat_id}` and `/chunk/{chunk_id}`
    so it isn't shadowed by those path parameters."""
    conversations = list_recent_chats(db, session_id)

    return [
        ChatSummary(
            chat_id=conversation.conversation_id,
            chat_title=conversation.chat_title,
            created_at=conversation.created_at,
        )
        for conversation in conversations
    ]


@router.get("/chunk/{chunk_id}", response_model=ChunkMessagesResponse)
def get_chunk(chunk_id: str, db: Session = Depends(get_db)) -> ChunkMessagesResponse:
    """Returns one older chunk's messages - used when the user clicks
    "Load older messages" while viewing a conversation. Declared before
    `/{chat_id}` so `/chunk/...` isn't captured by that path parameter."""
    chunk = get_chunk_by_chunk_id(db, chunk_id)

    if chunk is None:
        raise HTTPException(status_code=404, detail="Chunk not found")

    return ChunkMessagesResponse(
        messages=chunk.messages,
        previous_chunk_id=chunk.previous_chunk_id,
    )


@router.get("/{chat_id}", response_model=ChatDetailResponse)
def get_chat(chat_id: str, db: Session = Depends(get_db)) -> ChatDetailResponse:
    """Returns a conversation's most recent chunk - used to open a chat
    picked from the sidebar's history list. `chat_id` is the conversation
    id (see `ChatSummary`)."""
    chunk = get_active_chunk(db, conversation_id=chat_id)

    if chunk is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    return ChatDetailResponse(
        chat_id=chat_id,
        chat_title=chunk.chat_title,
        messages=chunk.messages,
        previous_chunk_id=chunk.previous_chunk_id,
    )
