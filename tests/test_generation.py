"""Tests for the generation pipeline."""

import pytest


class TestCitationFormatter:
    """Tests for citation formatting."""

    def test_format_empty_sources(self):
        """Should handle empty source list."""
        from generation.citation_formatter import format_citations

        result = format_citations([])
        assert "No sources" in result

    def test_format_text_source(self):
        """Should format text sources correctly."""
        from generation.citation_formatter import format_citations

        sources = [
            {"file": "study.pdf", "page": 5, "type": "text", "score": 0.92}
        ]
        result = format_citations(sources)
        assert "study.pdf" in result
        assert "p.5" in result

    def test_format_image_source(self):
        """Should mark image sources."""
        from generation.citation_formatter import format_citations

        sources = [
            {"file": "scan.pdf", "page": 3, "type": "image", "score": 0.88}
        ]
        result = format_citations(sources)
        assert "Image" in result

    def test_extract_cited_sources(self):
        """Should extract only cited source indices."""
        from generation.citation_formatter import extract_cited_sources

        answer = "According to [Source 1, p.5], the treatment was effective. [Source 3] confirms this."
        sources = [
            {"file": "a.pdf", "page": 5, "type": "text", "score": 0.9},
            {"file": "b.pdf", "page": 2, "type": "text", "score": 0.8},
            {"file": "c.pdf", "page": 1, "type": "text", "score": 0.7},
        ]

        cited = extract_cited_sources(answer, sources)
        assert len(cited) == 2  # Source 1 and Source 3

    def test_inline_citations(self):
        """Should append citation block to answer."""
        from generation.citation_formatter import format_inline_citations

        answer = "The study found positive results."
        sources = [
            {"file": "test.pdf", "page": 1, "type": "text", "score": 0.9}
        ]

        result = format_inline_citations(answer, sources)
        assert answer in result
        assert "Sources" in result


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_default_values(self):
        """GenerationResult should have sensible defaults."""
        from generation.llm_chain import GenerationResult

        result = GenerationResult(answer="Test answer")
        assert result.answer == "Test answer"
        assert result.sources == []
        assert result.model == ""
        assert result.tokens_used == 0
