"""Tests for the retrieval pipeline."""

import pytest
import numpy as np


class TestFAISSRetriever:
    """Tests for the FAISS retriever."""

    def test_retrieval_result_fields(self):
        """RetrievalResult should have correct default values."""
        from retrieval.retriever import RetrievalResult

        result = RetrievalResult(
            rank=1,
            score=0.95,
            text="Medical finding",
            page_number=1,
            source_file="test.pdf",
        )
        assert result.rank == 1
        assert result.score == 0.95
        assert result.content_type == "text"


class TestReranker:
    """Tests for the cross-encoder re-ranker."""

    def test_empty_candidates(self):
        """Reranker should handle empty candidate list."""
        from retrieval.reranker import Reranker

        reranker = Reranker.__new__(Reranker)
        reranker._model = None
        result = reranker.rerank("test query", [], top_n=5)
        assert result == []


class TestContextBuilder:
    """Tests for the context builder."""

    def test_empty_results(self):
        """Should return a default message for empty results."""
        from retrieval.context_builder import build_context

        context = build_context([])
        assert "No relevant context" in context

    def test_source_list_deduplication(self):
        """get_source_list should deduplicate by file+page."""
        from retrieval.context_builder import get_source_list
        from retrieval.retriever import RetrievalResult

        results = [
            RetrievalResult(
                rank=1, score=0.9, text="text1",
                page_number=1, source_file="a.pdf"
            ),
            RetrievalResult(
                rank=2, score=0.8, text="text2",
                page_number=1, source_file="a.pdf"  # duplicate
            ),
            RetrievalResult(
                rank=3, score=0.7, text="text3",
                page_number=2, source_file="a.pdf"
            ),
        ]

        sources = get_source_list(results)
        assert len(sources) == 2  # deduped
