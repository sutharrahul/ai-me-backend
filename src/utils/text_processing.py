"""Text processing utilities for RAG pipeline."""

import re
from typing import List

def clean_text(text: str) -> str:
    """Remove extra whitespace and normalize text."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into chunks of specified size."""
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(' '.join(current_chunk)) >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
