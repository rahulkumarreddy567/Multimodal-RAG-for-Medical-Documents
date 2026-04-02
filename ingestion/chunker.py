"""
Text chunking with sliding window.

Splits extracted text into 512-token chunks with 64-token overlap.
Preserves sentence boundaries to avoid mid-sentence splits.
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Simple sentence boundary pattern
SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+')


@dataclass
class Chunk:
    """A chunk of text ready for embedding."""
    text: str
    chunk_id: int
    page_number: int
    source_file: str
    start_char: int = 0  # character offset in original text
    end_char: int = 0
    metadata: dict = field(default_factory=dict)


def _estimate_tokens(text: str) -> int:
    """Rough token count estimate (words ≈ 0.75 tokens)."""
    return int(len(text.split()) * 1.33)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    source_file: str = "",
    page_number: int = 0,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """
    Split text into chunks using a sliding window with sentence-boundary awareness.

    Args:
        text: The input text to chunk.
        source_file: Source PDF filename (for metadata).
        page_number: Page number the text came from.
        chunk_size: Target chunk size in estimated tokens.
        chunk_overlap: Overlap between consecutive chunks in estimated tokens.

    Returns:
        List of Chunk objects.
    """
    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_sentences = []
    current_token_count = 0
    chunk_id = 0
    char_offset = 0

    for sentence in sentences:
        sentence_tokens = _estimate_tokens(sentence)

        # If adding this sentence exceeds chunk_size, finalize current chunk
        if current_token_count + sentence_tokens > chunk_size and current_sentences:
            chunk_text_str = " ".join(current_sentences)
            chunks.append(
                Chunk(
                    text=chunk_text_str,
                    chunk_id=chunk_id,
                    page_number=page_number,
                    source_file=source_file,
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_text_str),
                    metadata={
                        "num_sentences": len(current_sentences),
                        "est_tokens": current_token_count,
                    },
                )
            )
            chunk_id += 1

            # Calculate overlap: keep last N sentences that fit in overlap window
            overlap_sentences = []
            overlap_tokens = 0
            for s in reversed(current_sentences):
                s_tokens = _estimate_tokens(s)
                if overlap_tokens + s_tokens > chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += s_tokens

            char_offset += len(chunk_text_str) - len(" ".join(overlap_sentences))
            current_sentences = overlap_sentences
            current_token_count = overlap_tokens

        current_sentences.append(sentence)
        current_token_count += sentence_tokens

    # Don't forget the last chunk
    if current_sentences:
        chunk_text_str = " ".join(current_sentences)
        chunks.append(
            Chunk(
                text=chunk_text_str,
                chunk_id=chunk_id,
                page_number=page_number,
                source_file=source_file,
                start_char=char_offset,
                end_char=char_offset + len(chunk_text_str),
                metadata={
                    "num_sentences": len(current_sentences),
                    "est_tokens": current_token_count,
                },
            )
        )

    logger.info(
        f"Chunked text into {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )

    return chunks


def chunk_text_blocks(
    text_blocks: list,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """
    Chunk a list of TextBlock objects (from pdf_extractor).

    Concatenates text from the same page, then chunks.

    Args:
        text_blocks: List of TextBlock objects from pdf_extractor.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of Chunk objects.
    """
    # Group by page
    pages: dict[int, list[str]] = {}
    source_file = ""

    for block in text_blocks:
        page = block.page_number
        if page not in pages:
            pages[page] = []
        pages[page].append(block.text)
        source_file = block.source_file

    # Chunk each page's text
    all_chunks = []
    global_id = 0

    for page_num in sorted(pages.keys()):
        page_text = " ".join(pages[page_num])
        page_chunks = chunk_text(
            text=page_text,
            source_file=source_file,
            page_number=page_num,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        # Re-number chunk IDs globally
        for chunk in page_chunks:
            chunk.chunk_id = global_id
            global_id += 1
        all_chunks.extend(page_chunks)

    logger.info(
        f"Total: {len(all_chunks)} chunks from {len(pages)} pages of {source_file}"
    )

    return all_chunks
