# ── Stage 1: Base image with dependencies ────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Application ─────────────────────────────────
FROM base AS app

WORKDIR /app

# Copy application code
COPY config/ config/
COPY ingestion/ ingestion/
COPY retrieval/ retrieval/
COPY generation/ generation/
COPY api/ api/
COPY ui/ ui/
COPY eval/ eval/

# Create data directories
RUN mkdir -p data/raw data/processed data/index

# Copy entrypoint
COPY .env.example .env

# Expose ports
EXPOSE 8000 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
