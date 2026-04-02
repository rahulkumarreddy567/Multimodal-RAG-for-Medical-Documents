"""
FAISS index builder — creates IVFFlat index and saves with metadata sidecar.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata stored alongside each vector in the index."""
    chunk_id: int
    text: str
    page_number: int
    source_file: str
    content_type: str = "text"  # "text" or "image"
    image_path: str = ""


def build_faiss_index(
    vectors: np.ndarray,
    metadata_list: list[ChunkMetadata],
    index_path: str = "data/index/faiss.index",
    metadata_path: str = "data/index/metadata.json",
    nlist: int = 100,
    use_ivf: bool = True,
) -> faiss.Index:
    """
    Build and save a FAISS index with metadata sidecar.

    Args:
        vectors: Numpy array of shape (N, dim) with normalized embeddings.
        metadata_list: List of ChunkMetadata (one per vector).
        index_path: Path to save the FAISS index.
        metadata_path: Path to save the metadata JSON.
        nlist: Number of IVF clusters.
        use_ivf: Use IVFFlat (True) or Flat index (False, for small datasets).

    Returns:
        The built FAISS index.
    """
    n_vectors, dim = vectors.shape
    vectors = vectors.astype(np.float32)

    logger.info(f"Building FAISS index: {n_vectors} vectors, dim={dim}")

    if use_ivf and n_vectors >= nlist * 10:
        # IVFFlat for larger datasets
        quantizer = faiss.IndexFlatIP(dim)  # Inner product (cosine on normalized vecs)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        # Train the index
        logger.info(f"Training IVFFlat index with nlist={nlist}...")
        index.train(vectors)
        index.add(vectors)

        # Set search-time probe count
        index.nprobe = min(10, nlist)

    else:
        # Flat index for small datasets (< nlist * 10 vectors)
        logger.info("Using Flat index (dataset too small for IVF)")
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

    logger.info(f"Index built: {index.ntotal} vectors indexed")

    # Save index
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    logger.info(f"Index saved to {index_path}")

    # Save metadata sidecar
    metadata_path = Path(metadata_path)
    metadata_dicts = [asdict(m) for m in metadata_list]
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_dicts, f, ensure_ascii=False, indent=2)
    logger.info(f"Metadata saved to {metadata_path} ({len(metadata_dicts)} entries)")

    return index


def load_faiss_index(
    index_path: str = "data/index/faiss.index",
    metadata_path: str = "data/index/metadata.json",
) -> tuple[faiss.Index, list[dict]]:
    """
    Load a saved FAISS index and its metadata.

    Returns:
        Tuple of (FAISS index, list of metadata dicts).
    """
    index = faiss.read_index(str(index_path))
    logger.info(f"Loaded index with {index.ntotal} vectors from {index_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    logger.info(f"Loaded {len(metadata)} metadata entries")

    return index, metadata


def add_to_index(
    index: faiss.Index,
    new_vectors: np.ndarray,
    existing_metadata: list[dict],
    new_metadata: list[ChunkMetadata],
    index_path: str = "data/index/faiss.index",
    metadata_path: str = "data/index/metadata.json",
) -> faiss.Index:
    """
    Add new vectors to an existing index and save.

    Args:
        index: Existing FAISS index.
        new_vectors: New vectors to add.
        existing_metadata: Current metadata list.
        new_metadata: Metadata for new vectors.
        index_path: Path to save updated index.
        metadata_path: Path to save updated metadata.

    Returns:
        Updated FAISS index.
    """
    new_vectors = new_vectors.astype(np.float32)
    index.add(new_vectors)

    # Update metadata
    existing_metadata.extend([asdict(m) for m in new_metadata])

    # Save
    faiss.write_index(index, str(index_path))
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(existing_metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"Added {len(new_vectors)} vectors. Total: {index.ntotal}")

    return index
