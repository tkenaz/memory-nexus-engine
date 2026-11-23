"""Markdown-aware recursive text chunker for Memory Nexus.

Splits long texts into focused chunks, each getting its own embedding.
Pure module with no DB dependencies.
"""
import re
from dataclasses import dataclass
from typing import List, Optional

from config.settings import settings


@dataclass
class Chunk:
    """A single chunk of text with its position metadata."""
    index: int
    text: str
    char_offset: int  # offset in original text


def needs_chunking(content: str) -> bool:
    """Check if content needs to be chunked."""
    return len(content) > settings.CHUNK_MAX_SIZE


def chunk_text(
    content: str,
    max_size: Optional[int] = None,
    overlap: Optional[int] = None,
    min_size: Optional[int] = None,
) -> List[Chunk]:
    """Split content into chunks. Returns empty list if chunking not needed.

    Strategy (recursive):
    1. Split by ## headers
    2. If section > max_size → split by ### headers
    3. If still > max_size → split by paragraph (\\n\\n)
    4. Last resort → split by sentences
    5. Overlap: prepend last ~overlap chars from previous chunk

    None means "use the setting"; overlap=0 disables overlap. A segment shorter
    than min_size is merged into the previous chunk, or, if it comes first, into
    the next one, so no text is lost. char_offset is the position of the chunk's
    own text (without the overlap prefix) in the original content.
    """
    max_size = settings.CHUNK_MAX_SIZE if max_size is None else max_size
    overlap = settings.CHUNK_OVERLAP if overlap is None else overlap
    min_size = settings.CHUNK_MIN_SIZE if min_size is None else min_size

    if len(content) <= max_size:
        return []

    raw_segments = _split_recursive(content, max_size)

    # Build chunks with overlap
    chunks: List[Chunk] = []
    search_from = 0
    prev_tail = ""
    pending = ""          # a too-small leading segment, carried into the next chunk
    pending_offset = 0

    for segment in raw_segments:
        segment = segment.strip()
        if not segment:
            continue
        found = content.find(segment, search_from)
        position = found if found >= 0 else search_from
        search_from = position + len(segment)

        if len(segment) < min_size:
            if chunks:
                chunks[-1].text = chunks[-1].text + "\n\n" + segment
            elif pending:
                pending = pending + "\n\n" + segment
            else:
                pending, pending_offset = segment, position
            continue

        if pending:
            segment = pending + "\n\n" + segment
            position = pending_offset
            pending = ""

        if chunks and prev_tail:
            chunk_text_val = f"...{prev_tail}\n\n{segment}"
        else:
            chunk_text_val = segment

        chunks.append(Chunk(index=len(chunks), text=chunk_text_val, char_offset=position))
        if overlap > 0:
            prev_tail = segment[-overlap:] if len(segment) > overlap else segment

    if pending:  # every segment was too small: keep the text as one chunk
        chunks.append(Chunk(index=len(chunks), text=pending, char_offset=pending_offset))

    # Re-index after possible merges
    for i, chunk in enumerate(chunks):
        chunk.index = i

    return chunks


def _split_recursive(text: str, max_size: int) -> List[str]:
    """Recursively split text using markdown hierarchy."""
    if len(text) <= max_size:
        return [text]

    # Level 1: ## headers
    parts = _split_by_pattern(text, r'\n(?=## )')
    if len(parts) > 1:
        result = []
        for part in parts:
            result.extend(_split_recursive(part, max_size))
        return result

    # Level 2: ### headers
    parts = _split_by_pattern(text, r'\n(?=### )')
    if len(parts) > 1:
        result = []
        for part in parts:
            result.extend(_split_recursive(part, max_size))
        return result

    # Level 3: paragraphs
    parts = _split_by_pattern(text, r'\n\n+')
    if len(parts) > 1:
        return _merge_small_parts(parts, max_size)

    # Level 4: sentences
    parts = _split_by_sentences(text)
    if len(parts) > 1:
        return _merge_small_parts(parts, max_size)

    # Can't split further — return as-is (will be oversized)
    return [text]


def _split_by_pattern(text: str, pattern: str) -> List[str]:
    """Split text by regex pattern, keeping delimiters with the following part."""
    parts = re.split(pattern, text)
    return [p for p in parts if p.strip()]


def _split_by_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Handle common sentence endings, including Russian
    sentences = re.split(r'(?<=[.!?。])\s+', text)
    return [s for s in sentences if s.strip()]


def _merge_small_parts(parts: List[str], max_size: int) -> List[str]:
    """Merge consecutive small parts to stay close to max_size."""
    merged: List[str] = []
    current = ""

    for part in parts:
        candidate = (current + "\n\n" + part).strip() if current else part
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = part

    if current:
        merged.append(current)

    return merged
