"""
FastAPI application — /query, /upload, and /health endpoints.

Orchestrates the full RAG pipeline: embed → retrieve → rerank → generate.
"""

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.schemas import (
    QueryRequest,
    QueryResponse,
    UploadResponse,
    HealthResponse,
    SourceInfo,
)
from config.settings import settings

logger = logging.getLogger(__name__)

# ── App Setup ─────────────────────────────────────────────

app = FastAPI(
    title="Multimodal Medical RAG API",
    description=(
        "Retrieval-Augmented Generation for medical documents. "
        "Supports text and image queries over PubMed Open Access papers."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-loaded components ────────────────────────────────
# Initialized on first request to avoid slow startup

_components = {}


def _get_components():
    """Lazy-load all pipeline components."""
    if not _components:
        from ingestion.embedder import MultiModalEmbedder
        from retrieval.retriever import FAISSRetriever
        from retrieval.reranker import Reranker
        from generation.llm_chain import MedicalRAGChain

        _components["embedder"] = MultiModalEmbedder(
            text_model=settings.text_embedding_model,
            image_model="ViT-L-14",
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
        )
        _components["retriever"] = FAISSRetriever(
            index_path=settings.faiss_index_path,
            metadata_path=settings.faiss_metadata_path,
        )
        _components["reranker"] = Reranker(
            model_name=settings.reranker_model,
            device=settings.embedding_device,
        )
        _components["chain"] = MedicalRAGChain(
            model_name=settings.llm_model,
            api_key=settings.openai_api_key,
            provider=settings.llm_provider,
        )

        logger.info("All pipeline components loaded")

    return _components


# ── Endpoints ─────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Multimodal Medical RAG API is running!",
        "docs_url": "/docs",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and index status."""
    try:
        components = _get_components()
        return HealthResponse(
            status="healthy",
            index_size=components["retriever"].index_size,
            model=settings.llm_model,
        )
    except Exception as e:
        return HealthResponse(
            status=f"degraded: {str(e)}",
            index_size=0,
            model=settings.llm_model,
        )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Answer a medical question using the RAG pipeline.

    1. Embed the query with BGE-M3
    2. Retrieve top-k from FAISS
    3. Re-rank with cross-encoder (optional)
    4. Generate answer with LLM + citations
    """
    try:
        components = _get_components()

        result = components["chain"].generate_with_retrieval(
            question=request.question,
            retriever=components["retriever"],
            embedder=components["embedder"],
            reranker=components["reranker"] if request.use_reranker else None,
            top_k=request.top_k,
            top_n=request.top_n,
        )

        return QueryResponse(
            answer=result.answer,
            sources=[SourceInfo(**s) for s in result.sources],
            model=result.model,
            tokens_used=result.tokens_used,
        )

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a new PDF document.

    Extracts text + images, chunks, embeds, and adds to FAISS index.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False, dir=str(settings.raw_data_path)
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        # Extract text
        from ingestion.pdf_extractor import extract_full
        extraction = extract_full(tmp_path)

        # Extract images
        from ingestion.image_extractor import extract_images
        images = extract_images(tmp_path)

        # Chunk text
        from ingestion.chunker import chunk_text_blocks
        chunks = chunk_text_blocks(
            extraction.text_blocks,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        # Embed and index
        components = _get_components()
        from ingestion.build_index import ChunkMetadata, add_to_index, load_faiss_index
        import numpy as np

        # Embed text chunks
        texts = [c.text for c in chunks]
        if texts:
            text_vecs = components["embedder"].embed_texts(texts)

            # Create metadata
            metadata_list = [
                ChunkMetadata(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    page_number=c.page_number,
                    source_file=file.filename,
                    content_type="text",
                )
                for c in chunks
            ]

            # Embed images
            image_vecs = np.array([], dtype=np.float32)
            if images:
                img_paths = [img.image_path for img in images]
                image_vecs = components["embedder"].embed_images(img_paths)

                for img in images:
                    metadata_list.append(
                        ChunkMetadata(
                            chunk_id=len(metadata_list),
                            text=f"Image from {img.source_file}, page {img.page_number}",
                            page_number=img.page_number,
                            source_file=file.filename,
                            content_type="image",
                            image_path=img.image_path,
                        )
                    )

            # Combine vectors
            all_vecs = text_vecs
            if image_vecs.size > 0:
                all_vecs = np.vstack([text_vecs, image_vecs])

            # Add to index
            try:
                index, existing_meta = load_faiss_index(
                    settings.faiss_index_path,
                    settings.faiss_metadata_path,
                )
                add_to_index(
                    index, all_vecs, existing_meta, metadata_list,
                    settings.faiss_index_path, settings.faiss_metadata_path,
                )
            except FileNotFoundError:
                from ingestion.build_index import build_faiss_index
                build_faiss_index(
                    all_vecs, metadata_list,
                    settings.faiss_index_path, settings.faiss_metadata_path,
                )

        return UploadResponse(
            filename=file.filename,
            pages_extracted=extraction.total_pages,
            chunks_created=len(chunks),
            images_extracted=len(images),
            vectors_indexed=len(chunks) + len(images),
            message="Document processed and indexed successfully",
        )

    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Entrypoint ────────────────────────────────────────────


def start():
    """Start the API server."""
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )


if __name__ == "__main__":
    start()
