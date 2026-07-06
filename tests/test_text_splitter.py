"""Tests for `services.text_splitter.split_text` - pure logic with no
external dependencies, so these run fast and guard against regressions in
chunk sizing/overlap behavior used during document ingestion."""

from app.services.text_splitter import split_text


def test_split_text_empty():
    """Blank/whitespace-only input should produce zero chunks rather than
    a chunk containing nothing meaningful."""
    assert split_text("") == []
    assert split_text("   ") == []


def test_split_text_short_returns_single_chunk():
    """Text that already fits within `chunk_size` shouldn't be split at
    all - it comes back as a single, unmodified chunk."""
    text = "This is a short piece of text."
    chunks = split_text(text, chunk_size=100, chunk_overlap=10)
    assert chunks == [text]


def test_split_text_respects_chunk_size():
    """Longer text should be broken into multiple chunks, each staying
    close to `chunk_size` (small slack allowed since splitting prefers
    word boundaries over exact character cutoffs)."""
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 100 + 20  # allow small slack from word boundaries


def test_split_text_produces_overlap():
    """Sanity check that splitting multi-paragraph text with a small
    chunk size still produces more than one chunk (overlap logic doesn't
    collapse everything back into one)."""
    text = "\n\n".join(f"Paragraph number {i} with some content." for i in range(20))
    chunks = split_text(text, chunk_size=80, chunk_overlap=20)
    assert len(chunks) > 1
