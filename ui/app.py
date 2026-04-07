"""
Gradio UI for multimodal medical document QA.
"""

import logging
from pathlib import Path

import gradio as gr

from config.settings import settings

logger = logging.getLogger(__name__)


def create_gradio_app():
    """Build and return the Gradio interface."""
    _pipeline = {}

    def _get_pipeline():
        if not _pipeline:
            from generation.citation_formatter import format_inline_citations
            from generation.llm_chain import MedicalRAGChain
            from ingestion.embedder import MultiModalEmbedder
            from retrieval.reranker import Reranker
            from retrieval.retriever import FAISSRetriever

            _pipeline["embedder"] = MultiModalEmbedder(
                device=settings.embedding_device,
            )
            _pipeline["retriever"] = FAISSRetriever(
                index_path=settings.faiss_index_path,
                metadata_path=settings.faiss_metadata_path,
            )
            _pipeline["reranker"] = Reranker(
                model_name=settings.reranker_model,
            )
            _pipeline["chain"] = MedicalRAGChain(
                model_name=settings.llm_model,
                api_key=settings.openai_api_key,
                provider=settings.llm_provider,
            )
            _pipeline["formatter"] = format_inline_citations
        return _pipeline

    def answer_question(question: str, use_reranker: bool = True):
        """Process a text question through the RAG pipeline."""
        if not question or not question.strip():
            return "Please enter a question.", ""

        try:
            pipeline = _get_pipeline()
            result = pipeline["chain"].generate_with_retrieval(
                question=question,
                retriever=pipeline["retriever"],
                embedder=pipeline["embedder"],
                reranker=pipeline["reranker"] if use_reranker else None,
            )

            formatted = pipeline["formatter"](result.answer, result.sources)
            source_text = "\n".join(
                f"- {s['file']}, p.{s['page']} ({s['type']}) - score: {s['score']:.3f}"
                for s in result.sources
            )
            return formatted, source_text
        except Exception as exc:
            logger.error("Query error: %s", exc, exc_info=True)
            return f"Error: {exc}", ""

    def upload_pdf(file):
        """Upload and index a PDF document."""
        if file is None:
            return "No file uploaded."

        try:
            import numpy as np

            from ingestion.build_index import (
                ChunkMetadata,
                add_to_index,
                build_faiss_index,
                load_faiss_index,
            )
            from ingestion.chunker import chunk_text_blocks
            from ingestion.image_extractor import extract_images
            from ingestion.pdf_extractor import extract_full

            pipeline = _get_pipeline()
            filename = Path(file.name).name
            extraction = extract_full(file.name)
            images = extract_images(file.name)
            chunks = chunk_text_blocks(
                extraction.text_blocks,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )

            texts = [chunk.text for chunk in chunks]
            text_vecs = (
                pipeline["embedder"].embed_texts(texts)
                if texts
                else np.array([], dtype=np.float32)
            )

            metadata_list = [
                ChunkMetadata(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    page_number=chunk.page_number,
                    source_file=filename,
                    content_type="text",
                )
                for chunk in chunks
            ]

            image_vecs = np.array([], dtype=np.float32)
            if images:
                image_vecs = pipeline["embedder"].embed_images(
                    [image.image_path for image in images]
                )
                for image_offset, image in enumerate(images, start=len(metadata_list)):
                    metadata_list.append(
                        ChunkMetadata(
                            chunk_id=image_offset,
                            text=f"Image from {image.source_file}, page {image.page_number}",
                            page_number=image.page_number,
                            source_file=filename,
                            content_type="image",
                            image_path=image.image_path,
                        )
                    )

            all_vecs = text_vecs
            if image_vecs.size > 0:
                all_vecs = (
                    np.vstack([text_vecs, image_vecs])
                    if text_vecs.size > 0
                    else image_vecs
                )

            vectors_indexed = 0
            if all_vecs.size > 0 and metadata_list:
                try:
                    index, existing_meta = load_faiss_index(
                        settings.faiss_index_path,
                        settings.faiss_metadata_path,
                    )
                    add_to_index(
                        index,
                        all_vecs,
                        existing_meta,
                        metadata_list,
                        settings.faiss_index_path,
                        settings.faiss_metadata_path,
                    )
                except FileNotFoundError:
                    build_faiss_index(
                        all_vecs,
                        metadata_list,
                        settings.faiss_index_path,
                        settings.faiss_metadata_path,
                    )
                vectors_indexed = len(metadata_list)

            return (
                f"Processed and indexed: {filename}\n\n"
                f"- Pages: {extraction.total_pages}\n"
                f"- Text chunks: {len(chunks)}\n"
                f"- Images extracted: {len(images)}\n"
                f"- Vectors indexed: {vectors_indexed}\n\n"
                "Document is ready for querying."
            )
        except Exception as exc:
            logger.error("Upload error: %s", exc, exc_info=True)
            return f"Error processing file: {exc}"

    with gr.Blocks(title="Medical RAG - Multimodal Document QA") as demo:
        gr.Markdown(
            """
# Multimodal Medical RAG
### Evidence-based answers from medical documents with citations

Ask questions about medical research papers. Answers are grounded in source
documents with page-level citations.
"""
        )

        with gr.Tabs():
            with gr.TabItem("Ask a Question"):
                with gr.Row():
                    with gr.Column(scale=2):
                        question_input = gr.Textbox(
                            label="Your Medical Question",
                            placeholder="e.g., What are the diagnostic criteria for Type 2 diabetes?",
                            lines=3,
                        )
                        reranker_toggle = gr.Checkbox(
                            label="Use cross-encoder re-ranking",
                            value=True,
                        )
                        submit_btn = gr.Button(
                            "Get Answer",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=3):
                        answer_output = gr.Markdown(label="Answer")
                        sources_output = gr.Textbox(
                            label="Sources",
                            lines=5,
                            interactive=False,
                        )

                submit_btn.click(
                    fn=answer_question,
                    inputs=[question_input, reranker_toggle],
                    outputs=[answer_output, sources_output],
                )

            with gr.TabItem("Upload Document"):
                gr.Markdown("Upload a medical PDF to add it to the knowledge base.")
                file_upload = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                )
                upload_btn = gr.Button("Process and Index", variant="primary")
                upload_result = gr.Markdown()

                upload_btn.click(
                    fn=upload_pdf,
                    inputs=[file_upload],
                    outputs=[upload_result],
                )

        gr.Markdown(
            """
---
*Built with BGE-M3 + CLIP embeddings, FAISS retrieval, cross-encoder
re-ranking, and GPT-4o/LLaMA generation.*
"""
        )

    return demo


if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=settings.gradio_port,
        share=False,
        theme=gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="blue",
        ),
    )
