"""
Centralized configuration — all hyperparameters, model names, and paths.
Reads from .env file with sensible defaults.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Force load from .env and OVERRIDE any bad Windows System Ghost Variables
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── Project Paths ──────────────────────────────────────
    project_root: Path = Path(__file__).resolve().parent.parent
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    index_dir: str = "data/index"

    # ── LLM ────────────────────────────────────────────────
    openai_api_key: str = ""
    llm_provider: str = "openai"  # "openai" or "local"
    llm_model: str = "gpt-4o"

    # ── Embedding Models ───────────────────────────────────
    text_embedding_model: str = "BAAI/bge-m3"
    image_embedding_model: str = "openai/clip-vit-large-patch14"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 8

    # ── Chunking ───────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── FAISS ──────────────────────────────────────────────
    faiss_index_path: str = "data/index/faiss.index"
    faiss_metadata_path: str = "data/index/metadata.json"
    faiss_nlist: int = 100

    # ── Retrieval ──────────────────────────────────────────
    top_k_retrieval: int = 20
    rerank_top_n: int = 5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── API ────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    gradio_port: int = 7860

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def raw_data_path(self) -> Path:
        return self.project_root / self.raw_data_dir

    @property
    def processed_data_path(self) -> Path:
        return self.project_root / self.processed_data_dir

    @property
    def index_path(self) -> Path:
        return self.project_root / self.index_dir


# Singleton instance
settings = Settings()
