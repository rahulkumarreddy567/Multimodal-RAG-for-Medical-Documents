# 🏥 Multimodal RAG for Medical Documents

A production-grade Retrieval-Augmented Generation (RAG) system that processes medical PDFs and clinical images to answer medical questions with evidence-based, cited responses.

## Architecture

```
PDF docs + Med images
    │
    ▼
┌─────────────────────── INGESTION ───────────────────────┐
│  PyMuPDF → CLIP/ViT → Chunker → BGE-M3 → FAISS index  │
└─────────────────────────────────────────────────────────┘
                          │
                     (indexed)
                          │
┌──────────────── RETRIEVAL + GENERATION ─────────────────┐
│  User query → BGE-M3 → FAISS top-k → Cross-encoder     │
│  re-rank → LLM (LLaMA-3/GPT-4V) → Grounded answer     │
│  with citations + sources                               │
└─────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Tools |
|-------|-------|
| Ingestion | PyMuPDF, Pillow, pdfplumber |
| Embeddings | BGE-M3 (text), CLIP ViT-L/14 (images) |
| Retrieval | FAISS (IVFFlat), sentence-transformers (cross-encoder) |
| Generation | LangChain, GPT-4V / LLaMA-3 |
| Serving | FastAPI, Gradio |
| Evaluation | RAGAS (faithfulness, answer relevancy) |
| Deployment | Docker, HuggingFace Spaces |

## Quick Start

### 1. Setup

```bash
# Clone
git clone https://github.com/rahulkumarreddy567/Multimodal-RAG-for-Medical-Documents.git
cd Multimodal-RAG-for-Medical-Documents

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Configure
cp .env.example .env
# Edit .env with your API keys
```

### 2. Download Data

```bash
python scripts/download_pubmed.py --max 50 --output data/raw
```

### 3. Build Index

```python
from ingestion.pdf_extractor import extract_full
from ingestion.chunker import chunk_text_blocks
from ingestion.embedder import MultiModalEmbedder
from ingestion.build_index import build_faiss_index, ChunkMetadata

# Extract + chunk + embed + index
result = extract_full("data/raw/PMC12345.pdf")
chunks = chunk_text_blocks(result.text_blocks)
embedder = MultiModalEmbedder()
vectors = embedder.embed_texts([c.text for c in chunks])

metadata = [
    ChunkMetadata(chunk_id=c.chunk_id, text=c.text,
                  page_number=c.page_number, source_file=c.source_file)
    for c in chunks
]
build_faiss_index(vectors, metadata)
```

### 4. Run API

```bash
python -m api.main
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 5. Run Gradio UI

```bash
python -m ui.app
# UI at http://localhost:7860
```

### 6. Docker

```bash
docker-compose up --build
```

## Run Verification (Smoke Test)

Use this quick checklist to confirm the project is connected correctly and running without startup errors.

```bash
# 1) Run tests
pytest -q

# 2) Start API
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# 3) In a new terminal, check API + health
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/health

# 4) Start UI (new terminal)
python -m ui.app
# Open http://127.0.0.1:7860

# 5) Query smoke test (new terminal)
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What is this dataset about?\",\"top_k\":3,\"top_n\":2,\"use_reranker\":true}"
```

If you see a healthy status from `/health`, UI loads on port 7860, and `/query` returns JSON with `answer` and `sources`, the end-to-end pipeline is working.

## Evaluation

```bash
# Generate eval dataset (requires indexed documents)
python scripts/seed_eval_dataset.py --num 100

# Run RAGAS evaluation
python -m eval.evaluate
```

## Project Structure

```
multimodal-rag/
├── ingestion/          # PDF/image extraction, chunking, embedding
├── retrieval/          # FAISS search, cross-encoder re-ranking
├── generation/         # LLM chain, citation formatting
├── api/                # FastAPI endpoints
├── ui/                 # Gradio interface
├── eval/               # RAGAS evaluation, benchmarks
├── scripts/            # Data download, eval dataset generation
├── config/             # Centralized settings
├── tests/              # Unit tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Benchmark Results

| Strategy | Faithfulness | Answer Relevancy | Context Precision | Recall@5 |
|----------|-------------|-----------------|-------------------|----------|
| BM25 | — | — | — | — |
| Dense-only (BGE-M3) | — | — | — | — |
| **Multimodal (Ours)** | — | — | — | — |

*Results will be populated after Week 7 evaluation.*

## License

MIT
