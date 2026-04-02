"""
Cross-encoder re-ranker using sentence-transformers.

Re-ranks FAISS top-k candidates to improve precision.
Uses ms-marco-MiniLM-L-6-v2 by default.
"""

import logging
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder re-ranker for passage scoring."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            logger.info(f"Loading re-ranker model: {self.model_name}")
            self._model = CrossEncoder(
                self.model_name,
                max_length=512,
                device=self.device,
            )
            logger.info("Re-ranker model loaded")

    def rerank(
        self,
        query: str,
        candidates: list,
        top_n: int = 5,
    ) -> list:
        """
        Re-rank retrieval candidates using cross-encoder scoring.

        Args:
            query: The original user query.
            candidates: List of RetrievalResult objects from FAISS retriever.
            top_n: Number of top results to return after re-ranking.

        Returns:
            List of RetrievalResult, re-scored and sorted by cross-encoder score.
        """
        self._load_model()

        if not candidates:
            return []

        # Build query-passage pairs
        pairs = [(query, c.text) for c in candidates if c.text]

        if not pairs:
            return candidates[:top_n]

        # Score all pairs
        scores = self._model.predict(pairs)

        # Attach scores and sort
        scored_candidates = []
        for candidate, score in zip(candidates, scores):
            candidate.score = float(score)
            scored_candidates.append(candidate)

        scored_candidates.sort(key=lambda x: x.score, reverse=True)

        # Re-assign ranks
        for i, c in enumerate(scored_candidates[:top_n]):
            c.rank = i + 1

        logger.info(
            f"Re-ranked {len(candidates)} candidates → top {top_n}. "
            f"Best score: {scored_candidates[0].score:.4f}"
        )

        return scored_candidates[:top_n]
