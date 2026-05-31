"""Markdown-aware recursive text chunker for Memory Nexus.

Splits long texts into focused chunks, each getting its own embedding.
Pure module with no DB dependencies.
"""
import re
from dataclasses import dataclass
from typing import List

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
    max_size: int = 0,
    overlap: int = 0,
    min_size: int = 0,
) -> List[Chunk]:
    """Split content into chunks. Returns empty list if chunking not needed.

    Strategy (recursive):
    1. Split by ## headers
    2. If section > max_size → split by ### headers
    3. If still > max_size → split by paragraph (\\n\\n)
    4. Last resort → split by sentences
    5. Overlap: prepend last ~overlap chars from previous chunk
    """
    max_size = max_size or settings.CHUNK_MAX_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP
    min_size = min_size or settings.CHUNK_MIN_SIZE

    if len(content) <= max_size:
        return []

    raw_segments = _split_recursive(content, max_size)

    # Build chunks with overlap
    chunks: List[Chunk] = []
    offset = 0
    prev_tail = ""

    for i, segment in enumerate(raw_segments):
        segment = segment.strip()
        if not segment or len(segment) < min_size:
            # Too small — merge with previous chunk if possible
            if chunks:
                chunks[-1] = Chunk(
                    index=chunks[-1].index,
                    text=chunks[-1].text + "\n\n" + segment,
                    char_offset=chunks[-1].char_offset,
                )
            offset += len(segment)
            continue

        if i > 0 and prev_tail:
            chunk_text_val = f"...{prev_tail}\n\n{segment}"
        else:
            chunk_text_val = segment

        chunks.append(Chunk(index=len(chunks), text=chunk_text_val, char_offset=offset))
        prev_tail = segment[-overlap:] if len(segment) > overlap else segment
        offset += len(segment)

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
