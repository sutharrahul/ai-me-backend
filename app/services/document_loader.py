"""Extracts raw text content from seed document files.

This is the first step of ingestion (`RagPipeline.ingest_document`): before
anything can be chunked or embedded, we need plain text out of whatever
file format the user dropped into `data/seed/`. Add support for a new file
type by extending `SUPPORTED_EXTENSIONS` and `load_text` below.
"""

from __future__ import annotations

from pathlib import Path

# File extensions the pipeline knows how to read. Used both here (to route
# to the right parser) and by `scripts/ingest_documents.py` (to decide
# which files in data/seed/ to pick up).
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


class UnsupportedFileTypeError(ValueError):
    """Raised when `load_text` is given a file extension that isn't in
    `SUPPORTED_EXTENSIONS`. Use case: catch this specifically if you want
    to skip/report bad files without crashing a batch ingestion run."""

    pass


def load_text(path: Path) -> str:
    """Reads a file and returns its plain-text content, dispatching on the
    file extension. `.txt`/`.md` are read directly as UTF-8 text (ignoring
    any decoding errors rather than crashing on odd bytes); `.pdf` is
    delegated to `_load_pdf`. Raises `UnsupportedFileTypeError` for
    anything else - add a new `elif` branch here to support more formats
    (e.g. `.docx`)."""
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        return _load_pdf(path)

    raise UnsupportedFileTypeError(
        f"Unsupported file type '{suffix}'. Supported types: "
        f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def _load_pdf(path: Path) -> str:
    """Extracts text from every page of a PDF and joins them with blank
    lines in between (so `text_splitter.split_text`'s paragraph-boundary
    splitting still works sensibly across page breaks). Imports `pypdf`
    lazily (inside the function) so the rest of the app doesn't pay the
    import cost unless a PDF is actually being ingested."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)
