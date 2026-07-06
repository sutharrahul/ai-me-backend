"""Lightweight recursive character text splitter.

Splits text into overlapping chunks, preferring to break on paragraph,
line, sentence, then word boundaries before falling back to raw
character slicing. This mirrors the common LangChain splitter behavior
without adding an extra dependency.

Why chunk at all: embedding models and LLM context windows work best with
small, focused pieces of text rather than a whole document at once.
Why overlap: without it, a sentence that gets cut across a chunk boundary
could lose meaning in both halves; overlap repeats a bit of the previous
chunk's tail so nearby context isn't lost during retrieval.
"""

from __future__ import annotations

# Tried in order: split on blank lines (paragraphs) first, then single
# newlines, then sentence-ending periods, then spaces, and finally an
# empty string means "just slice by raw characters" as a last resort.
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def split_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    separators: list[str] | None = None,
) -> list[str]:
    """Splits `text` into a list of chunks of at most ~`chunk_size`
    characters, with `chunk_overlap` characters repeated between
    consecutive chunks. Use case: called once per document during
    ingestion (`RagPipeline.ingest_document`) before each chunk is
    embedded and stored. Returns an empty list for blank/whitespace-only
    input (e.g. an empty file)."""
    if not text.strip():
        return []

    separators = separators or DEFAULT_SEPARATORS
    chunks = _split_recursive(text, chunk_size, separators)
    return _merge_with_overlap(chunks, chunk_size, chunk_overlap)


def _split_recursive(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Recursively breaks `text` into pieces no larger than `chunk_size`,
    trying each separator in order and only recursing into a smaller
    separator when a piece is still too big. This produces pieces that
    respect natural text boundaries (paragraphs/sentences/words) as much
    as possible instead of cutting mid-word. `_merge_with_overlap` then
    reassembles these pieces into final chunks close to `chunk_size`."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separator = separators[0]
    remaining_separators = separators[1:]

    if separator == "":
        # Last resort: no more separators to try, so hard-slice by
        # character count.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(separator)
    pieces: list[str] = []
    for part in parts:
        if not part:
            continue
        if len(part) > chunk_size and remaining_separators:
            # This piece is still too big even after splitting on the
            # current separator - recurse with the next, finer-grained
            # separator (e.g. paragraph -> sentence).
            pieces.extend(_split_recursive(part, chunk_size, remaining_separators))
        else:
            pieces.append(part)
    return pieces


def _merge_with_overlap(
    pieces: list[str], chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Greedily packs the small `pieces` (paragraphs/sentences/words) from
    `_split_recursive` back together into chunks as close to `chunk_size`
    as possible, carrying `chunk_overlap` trailing characters of each
    chunk over into the start of the next one for context continuity."""
    if not pieces:
        return []

    chunks: list[str] = []
    current = ""

    for piece in pieces:
        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= chunk_size:
            # Still fits within the target size - keep accumulating.
            current = candidate
            continue

        if current:
            chunks.append(current)
        # start new chunk, carrying over the overlap tail of the previous chunk
        overlap_tail = current[-chunk_overlap:] if chunk_overlap else ""
        current = f"{overlap_tail} {piece}".strip() if overlap_tail else piece

        # a single piece may still exceed chunk_size; hard-split it
        while len(current) > chunk_size:
            chunks.append(current[:chunk_size])
            current = current[chunk_size - chunk_overlap :]

    if current:
        chunks.append(current)

    return [c.strip() for c in chunks if c.strip()]
