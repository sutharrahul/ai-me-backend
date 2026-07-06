"""Chat/query endpoint - the core RAG API that the frontend chat UI calls.

This is intentionally a single, simple endpoint: given a question, run the
full retrieve-then-generate pipeline and return an answer plus the sources
used. All the actual RAG logic lives in `RagPipeline`
(`app/core/rag_pipeline.py`) - this route just validates the request,
delegates to the pipeline, and translates failures into HTTP errors.
"""

from fastapi import APIRouter, Depends, HTTPException, Request

from app.config import get_settings
from app.core.rag_pipeline import RagPipeline
from app.core.rate_limiter import limiter
from app.dependencies import get_rag_pipeline
from app.models.schemas import QueryRequest, QueryResponse
from app.utils.logger import get_logger

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
    pipeline: RagPipeline = Depends(get_rag_pipeline),
) -> QueryResponse:
    """Answers a user's question using retrieval-augmented generation.

    Flow: embed the question -> find the most similar chunks in the vector
    store -> ask the LLM to answer using only those chunks as context ->
    return the answer plus the sources so the UI can show citations.

    Rate limited per-IP (see `core/rate_limiter.py`) to `daily_question_limit`
    questions/day, since there's no auth to otherwise gate usage per-user -
    `request` (the raw Starlette request) is what `@limiter.limit(...)`
    inspects to identify the caller's IP; `payload` is the validated
    request body.

    Any exception from the pipeline (e.g. Postgres unreachable, Gemini API
    error) is caught here and turned into a 500 with the error message,
    rather than letting FastAPI's default handler return an opaque error -
    this makes debugging config/connectivity issues much easier from the
    frontend or curl.
    """
    try:
        return pipeline.answer_query(payload.question, top_k=payload.top_k)
    except Exception as exc:  # pragma: no cover - surfaced to the client
        logger.exception("Failed to answer question (top_k=%d)", payload.top_k)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
