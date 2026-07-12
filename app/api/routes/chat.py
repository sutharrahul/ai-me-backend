"""Chat/query endpoint for the portfolio chatbot.

Validates incoming requests, invokes the LangGraph workflow, and returns the
generated answer with its supporting sources.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from slowapi.util import get_remote_address

from app.config import get_settings
from app.core.rate_limiter import limiter
from app.models.schemas import QueryRequest, QueryResponse, Source
from app.utils.logger import get_logger
from sqlalchemy.orm import Session
from app.db.db_connection import get_db
from app.graph.graph import graph

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)

# Built from `settings.daily_question_limit` (default 20) rather than a
# hardcoded string, so changing DAILY_QUESTION_LIMIT in .env is enough to
# retune it - no code change needed. `slowapi`'s decorator needs this
# string at import time, so it reads settings once, at module load.
_DAILY_LIMIT = f"{get_settings().daily_question_limit}/day"


@router.post("/query", response_model=QueryResponse)
@limiter.limit(_DAILY_LIMIT)
def query(
    request: Request,
    payload: QueryRequest,
    db: Session = Depends(get_db),
) -> QueryResponse:
    """
    Answers a user's question using the LangGraph workflow.

    The graph handles intent classification, retrieval (when needed), response
    generation, and chat persistence before returning the answer and sources.

    Unexpected errors are logged and returned as HTTP 500 responses.
    """
    try:
        result = graph.invoke(
            {
                "db": db,
                "session_id": payload.session_id,
                "user_query": payload.question,
                "top_k": payload.top_k,
                "ip_address": get_remote_address(request),
            }
        )

        sources = [
            Source(
                document_id=chunk.document_id,
                filename=chunk.filename,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                score=chunk.score,
            )
            for chunk in result.get("chunks", [])
        ]

        return QueryResponse(
            answer=result["llm_response"],
            sources=sources,
        )

    except ChatGoogleGenerativeAIError as exc:
        logger.exception("Gemini API error (top_k=%d)", payload.top_k)
        if getattr(exc.__cause__, "code", None) == 429:
            raise HTTPException(
                status_code=429,
                detail="The AI assistant has hit its usage limit for now. Please try again in a bit.",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail="The AI assistant is temporarily unavailable. Please try again shortly.",
        ) from exc
    except Exception as exc:  # pragma: no cover - surfaced to the client
        logger.exception("Failed to answer question (top_k=%d)", payload.top_k)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
