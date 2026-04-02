"""Tests for the ingestion pipeline."""

import pytest
from pathlib import Path


class TestPDFExtractor:
    """Tests for pdf_extractor module."""

    def test_extract_text_invalid_path(self):
        """Should raise FileNotFoundError for nonexistent PDF."""
        from ingestion.pdf_extractor import extract_text

        with pytest.raises(FileNotFoundError):
            extract_text("nonexistent.pdf")

    def test_extraction_result_structure(self):
        """ExtractionResult should have correct fields."""
        from ingestion.pdf_extractor import ExtractionResult, TextBlock

        result = ExtractionResult(source_file="test.pdf")
        assert result.source_file == "test.pdf"
        assert result.text_blocks == []
        assert result.total_pages == 0

    def test_text_block_default_type(self):
        """TextBlock should default to 'text' type."""
        from ingestion.pdf_extractor import TextBlock

        block = TextBlock(text="test", page_number=1, source_file="test.pdf")
        assert block.block_type == "text"


class TestChunker:
    """Tests for chunker module."""

    def test_empty_text(self):
        """Empty text should return no chunks."""
        from ingestion.chunker import chunk_text

        chunks = chunk_text("", source_file="test.pdf")
        assert chunks == []

    def test_short_text_single_chunk(self):
        """Short text should produce a single chunk."""
        from ingestion.chunker import chunk_text

        text = "This is a short medical text about cardiac health."
        chunks = chunk_text(text, source_file="test.pdf", chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0].source_file == "test.pdf"

    def test_long_text_multiple_chunks(self):
        """Long text should produce multiple chunks."""
        from ingestion.chunker import chunk_text

        # Generate text that's definitely longer than one chunk
        sentences = [
            f"Sentence number {i} discusses an important medical finding."
            for i in range(100)
        ]
        text = " ".join(sentences)

        chunks = chunk_text(text, source_file="test.pdf", chunk_size=50)
        assert len(chunks) > 1

    def test_chunk_ids_sequential(self):
        """Chunk IDs should be sequential."""
        from ingestion.chunker import chunk_text

        sentences = [f"Medical finding {i} is very important." for i in range(50)]
        text = " ".join(sentences)

        chunks = chunk_text(text, chunk_size=50)
        ids = [c.chunk_id for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_chunk_metadata(self):
        """Chunks should carry source metadata."""
        from ingestion.chunker import chunk_text

        chunks = chunk_text(
            "A medical study found significant results in the treatment group.",
            source_file="study.pdf",
            page_number=5,
        )
        assert chunks[0].source_file == "study.pdf"
        assert chunks[0].page_number == 5


class TestImageExtractor:
    """Tests for image_extractor module."""

    def test_extract_images_invalid_path(self):
        """Should raise FileNotFoundError for nonexistent PDF."""
        from ingestion.image_extractor import extract_images

        with pytest.raises(FileNotFoundError):
            extract_images("nonexistent.pdf")

    def test_extracted_image_fields(self):
        """ExtractedImage should have correct fields."""
        from ingestion.image_extractor import ExtractedImage

        img = ExtractedImage(
            image_path="test.png",
            page_number=1,
            source_file="test.pdf",
            width=100,
            height=100,
            image_index=0,
        )
        assert img.width == 100
        assert img.page_number == 1
