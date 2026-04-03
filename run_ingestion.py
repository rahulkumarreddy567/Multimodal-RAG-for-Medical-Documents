from ingestion.pdf_extractor import extract_full
from ingestion.chunker import chunk_text_blocks
from ingestion.embedder import MultiModalEmbedder
from ingestion.build_index import build_faiss_index, ChunkMetadata
import logging

logging.basicConfig(level=logging.INFO)

print("Starting ingestion for sample_medical.pdf...")
result = extract_full("data/raw/sample_medical.pdf")
print(f"Extracted {len(result.text_blocks)} text blocks from {result.total_pages} pages.")

chunks = chunk_text_blocks(result.text_blocks)
print(f"Created {len(chunks)} chunks.")

embedder = MultiModalEmbedder()
print("Embedding text chunks...")
vectors = embedder.embed_texts([c.text for c in chunks])
print(f"Created {len(vectors)} vectors of dimension {vectors.shape[1]}.")

metadata = [
    ChunkMetadata(chunk_id=c.chunk_id, text=c.text,
                  page_number=c.page_number, source_file=c.source_file)
    for c in chunks
]

print("Building FAISS index...")
build_faiss_index(vectors, metadata)
print("Index built successfully!")
