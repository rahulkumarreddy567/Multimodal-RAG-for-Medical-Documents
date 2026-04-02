"""
FAISS top-k retriever.

Searches the vector index for the most similar chunks to a query embedding.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

from ingestion.build_index import load_faiss_index

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with score and metadata."""
    rank: int
    score: float
    text: str
    page_number: int
    source_file: str
    content_type: str = "text"
    image_path: str = ""
    metadata: dict = field(default_factory=dict)


class FAISSRetriever:
    """Top-k retriever using FAISS index."""

    def __init__(
        self,
        index_path: str = "data/index/faiss.index",
        metadata_path: str = "data/index/metadata.json",
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self._index = None
        self._metadata = None

    def _load(self):
        """Load index and metadata if not already loaded."""
        if self._index is None:
            self._index, self._metadata = load_faiss_index(
                self.index_path, self.metadata_path
            )

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 20,
    ) -> list[RetrievalResult]:
        """
        Search the FAISS index for top-k most similar vectors.

        Args:
            query_vector: Query embedding (1D or 2D array).
            k: Number of results to return.

        Returns:
            List of RetrievalResult sorted by score (descending).
        """
        self._load()

        # Ensure 2D
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype(np.float32)

        # Clamp k to index size
        k = min(k, self._index.ntotal)

        # Set nprobe for IVF indices
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = min(10, getattr(self._index, "nlist", 10))

        scores, indices = self._index.search(query_vector, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for missing results
                continue

            meta = self._metadata[idx] if idx < len(self._metadata) else {}
            results.append(
                RetrievalResult(
                    rank=rank + 1,
                    score=float(score),
                    text=meta.get("text", ""),
                    page_number=meta.get("page_number", 0),
                    source_file=meta.get("source_file", ""),
                    content_type=meta.get("content_type", "text"),
                    image_path=meta.get("image_path", ""),
                    metadata=meta,
                )
            )

        logger.info(f"Retrieved {len(results)} results (k={k})")
        return results

    def search_text(
        self,
        query: str,
        embedder,
        k: int = 20,
    ) -> list[RetrievalResult]:
        """
        Convenience: embed a text query and search.

        Args:
            query: Text query string.
            embedder: TextEmbedder or MultiModalEmbedder instance.
            k: Number of results.

        Returns:
            List of RetrievalResult.
        """
        if hasattr(embedder, "embed_texts"):
            vec = embedder.embed_texts([query])
        else:
            vec = embedder.embed([query])

        return self.search(vec[0], k=k)

    @property
    def index_size(self) -> int:
        """Number of vectors in the index."""
        self._load()
        return self._index.ntotal
