"""Markdown-aware chunker: src/core/chunker.py (pure, no DB)."""
import re

import pytest

from config.settings import settings
from src.core import chunker

MAX, OVERLAP, MIN = settings.CHUNK_MAX_SIZE, settings.CHUNK_OVERLAP, settings.CHUNK_MIN_SIZE


def section(title, sentence, repeat):
    return f"## {title}\n" + " ".join(f"{sentence} {i}." for i in range(repeat))


def bodies(chunks, overlap=OVERLAP):
    """Chunk texts without the '...<tail of previous segment>\\n\\n' prefix.

    The prefix is checked exactly, so this also verifies the overlap contract.
    """
    out = []
    for c in chunks:
        if out:
            prefix = "..." + out[-1][-overlap:] + "\n\n"
            assert c.text.startswith(prefix), f"chunk {c.index} lacks the overlap prefix"
            out.append(c.text[len(prefix):])
        else:
            assert not c.text.startswith("...")
            out.append(c.text)
    return out


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip() and not s.startswith("#")]


def test_settings_defaults_are_what_the_chunker_assumes():
    assert (MAX, OVERLAP, MIN) == (1500, 200, 100)


def test_needs_chunking_threshold_is_strictly_greater_than_max():
    assert not chunker.needs_chunking("x" * MAX)
    assert chunker.needs_chunking("x" * (MAX + 1))


@pytest.mark.parametrize("text", ["", "short note", "y" * MAX])
def test_content_up_to_max_size_is_not_chunked(text):
    assert chunker.chunk_text(text) == []


def test_sections_are_split_on_level_two_headers_with_overlap():
    parts = [section(f"Part {n}", f"Line about topic {n}", 40) for n in range(3)]
    text = "\n".join(parts)
    assert len(text) > MAX and all(len(p) <= MAX for p in parts)

    chunks = chunker.chunk_text(text)

    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert bodies(chunks) == [p.strip() for p in parts]  # also checks the overlap prefix


def test_chunk_length_is_bounded_by_max_plus_overlap():
    text = "\n\n".join(f"Paragraph {n}. " + "Words in a paragraph. " * 20 for n in range(40))
    chunks = chunker.chunk_text(text)
    assert len(chunks) > 1
    for body, c in zip(bodies(chunks), chunks):
        assert len(body) <= MAX
        assert len(c.text) <= MAX + OVERLAP + len("...\n\n")


def test_paragraphs_are_packed_up_to_max_size():
    paragraph = "A paragraph of plain text without headers. " * 5  # ~215 chars
    text = "\n\n".join([paragraph.strip()] * 20)
    chunks = chunker.chunk_text(text)
    packed = bodies(chunks)
    # 7 paragraphs would exceed 1500 chars, so 6 + 6 + 6 + 2; the 2-paragraph tail
    # is longer than CHUNK_MIN_SIZE and stays a chunk of its own.
    assert [b.count("A paragraph of plain text") // 5 for b in packed] == [6, 6, 6, 2]


def test_one_long_paragraph_falls_back_to_sentences_russian_included():
    text = " ".join(f"Это предложение номер {i}, оно про погоду и море!" for i in range(80))
    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 2
    packed = bodies(chunks)
    for body in packed:
        assert len(body) <= MAX
        assert body.endswith("!")  # cut on a sentence boundary
    joined = " ".join(packed)
    for i in range(80):
        assert f"номер {i}," in joined


def test_text_that_cannot_be_split_stays_one_oversized_chunk():
    text = "x" * (MAX * 2)
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_small_trailing_section_is_merged_into_the_previous_chunk():
    big = [section(f"Part {n}", f"Line about topic {n}", 40) for n in range(2)]
    tail = "## Note\nshort tail"
    chunks = chunker.chunk_text("\n".join(big + [tail]))
    assert len(chunks) == 2
    assert chunks[-1].text.endswith("\n\n" + tail)


def test_every_sentence_survives_when_no_section_is_too_small():
    parts = [section(f"Part {n}", f"Fact number {n}", 45) for n in range(4)]
    text = "\n".join(parts)
    joined = "\n".join(c.text for c in chunker.chunk_text(text))
    for s in sentences(text):
        assert s in joined


def test_explicit_sizes_override_settings():
    text = "\n\n".join(f"Para {n}. " + "Some words here. " * 6 for n in range(10))
    chunks = chunker.chunk_text(text, max_size=300, overlap=50, min_size=20)
    assert len(chunks) > 3
    assert all(len(body) <= 300 for body in bodies(chunks, overlap=50))


def test_short_leading_section_is_not_lost():
    intro = "## Intro\nShort preface line."
    rest = [section(f"Part {n}", f"Line about topic {n}", 40) for n in range(3)]
    chunks = chunker.chunk_text("\n".join([intro] + rest))
    assert any("Short preface line." in c.text for c in chunks)


def test_char_offset_points_into_the_original_text():
    text = "\n".join(section(f"Part {n}", f"Line about topic {n}", 40) for n in range(3))
    chunks = chunker.chunk_text(text)
    for c, body in zip(chunks, bodies(chunks)):
        assert text[c.char_offset:].startswith(body[:40])


def test_overlap_zero_disables_overlap():
    text = "\n\n".join(f"Para {n}. " + "Some words here. " * 30 for n in range(6))
    chunks = chunker.chunk_text(text, max_size=600, overlap=0)
    assert len(chunks) > 1
    assert not any(c.text.startswith("...") for c in chunks)
