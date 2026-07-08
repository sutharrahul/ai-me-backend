#!/usr/bin/env python3
"""Ingests every document in `data/seed/` into the Qdrant vector store.

Drop any `.md` or `.pdf` file into `backend/data/seed/` (e.g. your "about
me" bio, résumé, project write-ups) and run this script to chunk, embed,
and store them so the portfolio chatbot can answer questions grounded in
that content.

Chunk IDs are deterministic (`{filename}-{index}`), so running the script
without `--force` on an already-ingested file simply overwrites the same
Qdrant points — safe but redundant. Use `--force` to make the intent
explicit and clear stale chunks before re-embedding (useful when you change
`CHUNK_SIZE` or significantly edit a file).

Usage:
    python scripts/ingest_documents.py [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Puts the backend/ root on sys.path so `from app...` imports work when this
# script is run directly (`python scripts/ingest_documents.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.dependencies import get_rag_pipeline  # noqa: E402
from app.services.document_loader import SUPPORTED_EXTENSIONS  # noqa: E402
from app.utils.logger import get_logger, setup_logging  # noqa: E402

setup_logging()
logger = get_logger(__name__)


def _ingest_file(file_path: Path, pipeline, force: bool) -> bool:
    """Ingests a single file. With `--force`, deletes old chunks first so
    stale vectors don't accumulate. Returns True if the file was ingested
    (so the caller knows whether to clear the query cache afterward)."""
    filename = file_path.name

    if force:
        # Estimate how many chunks the old version produced so we can delete
        # them before re-ingesting. We don't know the exact count, so we
        # delete up to a generous upper bound using the vector store directly.
        # In practice, calling add_chunks with the same deterministic IDs
        # does an upsert, so this is only strictly necessary when chunk_size
        # changes and old chunks with higher indices become orphaned.
        settings = get_settings()
        # Compute a safe upper bound: file_size / (chunk_size - chunk_overlap)
        size = file_path.stat().st_size
        approx_max_chunks = max(
            1, size // max(1, settings.chunk_size - settings.chunk_overlap) + 2
        )
        pipeline.delete_document_chunks(filename, approx_max_chunks)
        logger.info("Cleared old chunks for '%s' (force re-ingest)", filename)

    pipeline.ingest_document(file_path)
    return True


def main() -> None:
    """Scans `data/seed/` for supported files and ingests them into Qdrant.
    For each file, chunks the text, generates Gemini embeddings, and upserts
    the vectors. After any ingestion the query cache is cleared so visitors
    immediately get answers grounded in the updated content."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing chunks and re-ingest even if the file was already ingested.",
    )
    args = parser.parse_args()

    settings = get_settings()

    if not settings.resolved_google_api_key():
        logger.error(
            "GOOGLE_API_KEY (or GEMINI_API_KEY) is not set. "
            "Add it to backend/.env before ingesting."
        )
        raise SystemExit(1)

    files = sorted(
        p for p in settings.seed_dir.glob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        logger.error(
            "No documents found in %s (supported: %s)",
            settings.seed_dir,
            ", ".join(sorted(SUPPORTED_EXTENSIONS)),
        )
        raise SystemExit(1)

    logger.info(
        "Found %d file(s) in %s: %s",
        len(files),
        settings.seed_dir,
        ", ".join(f.name for f in files),
    )

    pipeline = get_rag_pipeline()

    ingested_any = False
    for file_path in files:
        if _ingest_file(file_path, pipeline, args.force):
            ingested_any = True

    if ingested_any:
        # Clear every cached answer so visitors don't keep getting responses
        # grounded in the old version of a document until the TTL expires.
        pipeline.clear_query_cache()
        logger.info("Query cache cleared after ingesting new/updated content.")


if __name__ == "__main__":
    main()
