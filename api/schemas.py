"""
Pydantic request/response models for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field


# ── Request Models ────────────────────────────────────────


class QueryRequest(BaseModel):
    """Request body for /query endpoint."""
    question: str = Field(
        ...,
        description="The medical question to answer",
        min_length=5,
        examples=["What are the risk factors for cardiac arrest?"],
    )
    top_k: int = Field(
        default=20,
        description="Number of FAISS candidates to retrieve",
        ge=1,
        le=100,
    )
    top_n: int = Field(
        default=5,
        description="Number of re-ranked results to use as context",
        ge=1,
        le=20,
    )
    use_reranker: bool = Field(
        default=True,
        description="Whether to apply cross-encoder re-ranking",
    )


class UploadRequest(BaseModel):
    """Metadata for uploaded PDF (file sent as multipart)."""
    description: str = Field(
        default="",
        description="Optional description of the uploaded document",
    )


# ── Response Models ───────────────────────────────────────


class SourceInfo(BaseModel):
    """A single source citation."""
    file: str = Field(description="Source PDF filename")
    page: int = Field(description="Page number")
    type: str = Field(description="Content type: 'text' or 'image'")
    score: float = Field(description="Relevance score")


class QueryResponse(BaseModel):
    """Response body for /query endpoint."""
    answer: str = Field(description="Generated answer with citations")
    sources: list[SourceInfo] = Field(
        default_factory=list,
        description="List of cited sources",
    )
    model: str = Field(default="", description="LLM model used")
    tokens_used: int = Field(default=0, description="Total tokens consumed")


class UploadResponse(BaseModel):
    """Response body for /upload endpoint."""
    filename: str = Field(description="Uploaded filename")
    pages_extracted: int = Field(description="Number of pages processed")
    chunks_created: int = Field(description="Number of text chunks created")
    images_extracted: int = Field(description="Number of images extracted")
    vectors_indexed: int = Field(description="Number of vectors added to index")
    message: str = Field(default="Document processed successfully")


class HealthResponse(BaseModel):
    """Response for /health endpoint."""
    status: str = "healthy"
    index_size: int = Field(description="Number of vectors in FAISS index")
    model: str = Field(description="LLM model name")
