"""
Multimodal embedding — BGE-M3 for text, CLIP ViT-L/14 for images.

Produces unified vector representations that live in a shared embedding space
for cross-modal retrieval.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class TextEmbedder:
    """BGE-M3 text embedder using FlagEmbedding."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        batch_size: int = 8,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        """Lazy-load the BGE-M3 model."""
        if self._model is None:
            logger.info(f"Loading text embedding model: {self.model_name}")
            try:
                from FlagEmbedding import BGEM3FlagModel

                self._model = BGEM3FlagModel(
                    self.model_name,
                    use_fp16=(self.device == "cuda"),
                )
                logger.info("BGE-M3 model loaded successfully")
            except ImportError:
                logger.warning(
                    "FlagEmbedding not available, falling back to sentence-transformers"
                )
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    "BAAI/bge-m3", device=self.device
                )

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts into dense vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of shape (N, dim) with normalized embeddings.
        """
        self._load_model()

        if hasattr(self._model, "encode"):
            # FlagEmbedding BGEM3FlagModel
            try:
                result = self._model.encode(
                    texts,
                    batch_size=self.batch_size,
                    max_length=512,
                )
                # BGE-M3 returns dict with 'dense_vecs'
                if isinstance(result, dict):
                    vectors = result["dense_vecs"]
                else:
                    vectors = result
            except Exception:
                # sentence-transformers fallback
                vectors = self._model.encode(
                    texts,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                )
        else:
            raise RuntimeError("Model does not support encode()")

        vectors = np.array(vectors, dtype=np.float32)

        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms

        logger.info(f"Embedded {len(texts)} texts → shape {vectors.shape}")
        return vectors

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        self._load_model()
        test = self.embed(["test"])
        return test.shape[1]


class ImageEmbedder:
    """CLIP ViT-L/14 image embedder."""

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy-load the CLIP model."""
        if self._model is None:
            logger.info(f"Loading CLIP model: {self.model_name}")
            import open_clip

            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device,
            )
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            self._model.eval()
            logger.info("CLIP model loaded successfully")

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        """
        Embed a single image into a vector.

        Args:
            image_path: Path to the image file.

        Returns:
            1D numpy array (embedding vector).
        """
        self._load_model()

        image = Image.open(image_path).convert("RGB")
        image_tensor = self._preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self._model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().flatten().astype(np.float32)

    def embed_images(self, image_paths: list[str | Path]) -> np.ndarray:
        """
        Embed multiple images into vectors.

        Args:
            image_paths: List of image file paths.

        Returns:
            numpy array of shape (N, dim).
        """
        self._load_model()

        vectors = []
        for path in image_paths:
            try:
                vec = self.embed_image(path)
                vectors.append(vec)
            except Exception as e:
                logger.warning(f"Failed to embed image {path}: {e}")

        if not vectors:
            return np.array([], dtype=np.float32)

        result = np.stack(vectors)
        logger.info(f"Embedded {len(vectors)} images → shape {result.shape}")
        return result

    @property
    def dimension(self) -> int:
        """Return the CLIP embedding dimension."""
        self._load_model()
        return self._model.visual.output_dim


class MultiModalEmbedder:
    """
    Unified embedder that handles both text and images.

    Projects CLIP image embeddings into the BGE-M3 text embedding space
    via a learned linear projection (or zero-padding for MVP).
    """

    def __init__(
        self,
        text_model: str = "BAAI/bge-m3",
        image_model: str = "ViT-L-14",
        device: str = "cpu",
        batch_size: int = 8,
    ):
        self.text_embedder = TextEmbedder(
            model_name=text_model, device=device, batch_size=batch_size
        )
        self.image_embedder = ImageEmbedder(
            model_name=image_model, device=device
        )
        self._projection = None

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed text chunks."""
        return self.text_embedder.embed(texts)

    def embed_images(self, image_paths: list[str | Path]) -> np.ndarray:
        """
        Embed images and project to text embedding space.

        For MVP: zero-pads/truncates CLIP vectors to match BGE-M3 dimension.
        For production: use a trained linear projection.
        """
        clip_vecs = self.image_embedder.embed_images(image_paths)

        if clip_vecs.size == 0:
            return clip_vecs

        text_dim = self.text_embedder.dimension
        clip_dim = clip_vecs.shape[1]

        if clip_dim == text_dim:
            return clip_vecs

        # Simple alignment: zero-pad or truncate
        if clip_dim < text_dim:
            padding = np.zeros(
                (clip_vecs.shape[0], text_dim - clip_dim), dtype=np.float32
            )
            aligned = np.concatenate([clip_vecs, padding], axis=1)
        else:
            aligned = clip_vecs[:, :text_dim]

        # Re-normalize
        norms = np.linalg.norm(aligned, axis=1, keepdims=True)
        norms[norms == 0] = 1
        aligned = aligned / norms

        logger.info(
            f"Projected {len(clip_vecs)} image vectors "
            f"from dim={clip_dim} to dim={text_dim}"
        )

        return aligned
