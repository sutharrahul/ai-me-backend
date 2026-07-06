"""Document ingestion endpoint - fetches a document's content from an
external API and runs it through the same ingestion pipeline used for
files dropped into `data/seed/`.

`ingest-url` is deliberately generic about the source: any URL that
returns the document's text (plain text, markdown, JSON, etc.) works, as
long as the caller supplies whatever header the source's API needs for
auth (e.g. `Authorization: Bearer <key>` or a custom `X-API-Key` header).
"""

from fastapi import APIRouter, Depends, HTTPException
import requests

from app.core.rag_pipeline import RagPipeline, new_document_id
from app.dependencies import get_document_store, get_rag_pipeline
from app.models.schemas import DocumentItem, DocumentStatus, IngestUrlRequest
from app.services.document_store import DocumentStore
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/ingest-url", response_model=DocumentItem)
def ingest_url(
    payload: IngestUrlRequest,
    store: DocumentStore = Depends(get_document_store),
    pipeline: RagPipeline = Depends(get_rag_pipeline),
) -> DocumentItem:
    """Fetches `payload.url` (sending `payload.api_key` in the
    `payload.header_name` request header), then chunks/embeds/stores the
    response body as a new document via `RagPipeline.ingest_text` - the
    same pipeline used for local files, just sourced from a remote API
    instead of `data/seed/`.
    """
    try:
        response = requests.get(
            str(payload.url),
            headers={payload.header_name: payload.api_key},
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Failed to fetch ingest-url source %s: %s", payload.url, exc)
        raise HTTPException(
            status_code=502, detail=f"Failed to fetch URL: {exc}"
        ) from exc

    text = response.text
    if not text.strip():
        raise HTTPException(status_code=422, detail="Fetched content is empty")

    document_id = new_document_id()
    store.create(document_id, payload.filename)
    pipeline.ingest_text(document_id, payload.filename, text)

    document = store.get(document_id)
    if document is None:  # pragma: no cover - defensive, shouldn't happen
        raise HTTPException(status_code=500, detail="Ingestion failed unexpectedly")
    if document.status == DocumentStatus.FAILED:
        raise HTTPException(
            status_code=422,
            detail="Ingestion failed - no extractable text or an embedding error occurred",
        )
    return document
