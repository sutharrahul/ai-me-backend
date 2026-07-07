#!/usr/bin/env python3
"""Ingests every document in `data/seed/` into the vector store.

Drop any `.txt`, `.md`, or `.pdf` file into `backend/data/seed/` (e.g. your
"about me" bio, resume, project write-ups) and run this script to chunk,
embed, and store them so the chat assistant can answer questions grounded
in that content. Already-ingested files are skipped unless `--force` is
passed, so it's safe to re-run after adding new files.

This is the CLI entrypoint that replaces manual API-driven document
upload: instead of calling an `/api/documents/upload` endpoint, you just
manage files on disk and run this script whenever you add/change one.

Usage:
    python scripts/ingest_documents.py [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allows `from app...` imports to work when this script is run directly
# (e.g. `python scripts/ingest_documents.py`) rather than as part of the
# installed `app` package, by putting the backend/ root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.core.rag_pipeline import RagPipeline, new_document_id  # noqa: E402
from app.dependencies import get_document_store, get_rag_pipeline  # noqa: E402
from app.services.document_loader import SUPPORTED_EXTENSIONS  # noqa: E402
from app.services.document_store import DocumentStore  # noqa: E402
from app.utils.logger import get_logger, setup_logging  # noqa: E402

setup_logging()
logger = get_logger(__name__)


def _ingest_file(
    file_path: Path, store: DocumentStore, pipeline: RagPipeline, force: bool
) -> bool:
    """Ingests a single file unless it's already tracked (and `force` is
    false), in which case it logs and does nothing. Returns whether the
    file was actually (re-)ingested, so `main` knows whether the query
    cache needs clearing afterward."""
    # Idempotency check: match by filename (not path) so re-running the
    # script after the container restarts doesn't re-embed unchanged
    # files every time.
    existing = [doc for doc in store.list_all() if doc.filename == file_path.name]
    if existing and not force:
        logger.info(
            "'%s' is already ingested (id=%s, status=%s). Use --force to re-ingest.",
            file_path.name,
            existing[0].id,
            existing[0].status,
        )
        return False

    # On a forced re-ingest (or if a previous partial ingest left stale
    # chunks), clear out the old version's chunks/metadata first so we
    # don't end up with duplicate/orphaned vectors.
    for doc in existing:
        pipeline.delete_document(doc.id)

    document_id = new_document_id()
    store.create(document_id, file_path.name)
    pipeline.ingest_document(document_id, file_path.name, file_path)

    refreshed = store.get(document_id)
    logger.info(
        "Ingested '%s' -> status=%s, chunks=%s",
        file_path.name,
        refreshed.status if refreshed else "unknown",
        refreshed.chunk_count if refreshed else "?",
    )
    return True


def main() -> None:
    """Entry point: scans `data/seed/` for supported files and ingests any
    that aren't already tracked in the `DocumentStore` (or all of them, if
    `--force` is passed). For each file this creates a fresh document
    record and runs it through `RagPipeline.ingest_document`, then logs
    the resulting status/chunk count so you can confirm it worked."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest files even if already present in the vector store.",
    )
    args = parser.parse_args()

    settings = get_settings()
    # Fail fast with a clear message rather than letting the first Gemini/
    # OpenAI API call fail with a cryptic auth error deep in the pipeline.
    # Ollama needs no API key (it's local), so it's exempt from this check.
    if settings.embedding_provider == "openai" and not settings.openai_api_key.get_secret_value().strip():
        logger.error(
            "OPENAI_API_KEY is not set. Configure backend/.env before ingesting."
        )
        raise SystemExit(1)
    if settings.embedding_provider == "gemini" and not settings.resolved_google_api_key():
        logger.error(
            "GOOGLE_API_KEY is not set. Configure backend/.env before ingesting."
        )
        raise SystemExit(1)

    # Only pick up files with extensions `document_loader.py` knows how to
    # parse; anything else in the folder (e.g. a stray .DS_Store) is
    # silently ignored.
    files = sorted(
        p for p in settings.seed_dir.glob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        logger.error(
            "No documents found in %s (supported: %s)",
            settings.seed_dir,
            ", ".join(sorted(SUPPORTED_EXTENSIONS)),
        )
        raise SystemExit(1)

    store = get_document_store()
    pipeline = get_rag_pipeline()

    # An explicit loop (not `any(_ingest_file(...) for ...)`) is
    # deliberate here: `any()` short-circuits on the first `True`, which
    # would skip ingesting every file after the first newly-ingested one.
    ingested_any = False
    for file_path in files:
        if _ingest_file(file_path, store, pipeline, args.force):
            ingested_any = True

    if ingested_any:
        # Content changed - drop every cached answer (see
        # `services/cache.py`) so visitors don't keep getting responses
        # grounded in the old version of a document until the cache
        # entry happens to expire on its own (`QUERY_CACHE_TTL_SECONDS`).
        pipeline.clear_query_cache()
        logger.info("Cleared the query cache after ingesting new/updated content.")


if __name__ == "__main__":
    main()
