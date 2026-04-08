import logging
from pathlib import Path

import gradio as gr

from config.settings import settings

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

:root {
    --primary-color: #00d2ff;
    --secondary-color: #3a7bd5;
    --bg-gradient: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    --glass-bg: rgba(255, 255, 255, 0.05);
    --glass-border: rgba(255, 255, 255, 0.1);
}

body {
    font-family: 'Outfit', sans-serif !important;
    background: var(--bg-gradient) !important;
    color: #e0e0e0 !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

.header-container {
    text-align: center;
    padding: 2rem 0;
    background: var(--glass-bg);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--glass-border);
    margin-bottom: 2rem;
    border-radius: 0 0 20px 20px;
}

.header-title {
    font-size: 2.5rem;
    font-weight: 600;
    background: linear-gradient(to right, #00d2ff, #3a7bd5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.header-subtitle {
    font-size: 1.1rem;
    opacity: 0.8;
}

.card {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(15px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 15px !important;
    padding: 1.5rem !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
}

.tab-item {
    border: none !important;
}

.primary-btn {
    background: linear-gradient(to right, #00d2ff, #3a7bd5) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 15px rgba(0, 210, 255, 0.4) !important;
}

.source-item {
    padding: 0.5rem;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 8px;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
    border-left: 3px solid var(--primary-color);
}
"""


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
            source_list = [
                f"<div class='source-item'><b>{s['file']}</b>, p.{s['page']} ({s['type']}) - conf: {s['score']:.3f}</div>"
                for s in result.sources
            ]
            return formatted, "".join(source_list)
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
                f"### ✅ Processed and indexed: {filename}\n\n"
                f"- **Pages**: {extraction.total_pages}\n"
                f"- **Text chunks**: {len(chunks)}\n"
                f"- **Images extracted**: {len(images)}\n"
                f"- **Vectors indexed**: {vectors_indexed}\n\n"
                "Document is ready for querying."
            )
        except Exception as exc:
            logger.error("Upload error: %s", exc, exc_info=True)
            return f"❌ Error processing file: {exc}"

    with gr.Blocks(title="Medical RAG - Multimodal Document QA", css=CUSTOM_CSS) as demo:
        # Header Section
        gr.HTML(
            """
            <div class='header-container'>
                <div class='header-title'>🏥 Medical RAG Assistant</div>
                <div class='header-subtitle'>Evidence-based answers from multimodal research papers</div>
            </div>
            """
        )

        with gr.Tabs():
            with gr.TabItem("🔍 Ask a Question", elem_id="qa-tab"):
                with gr.Row():
                    with gr.Column(scale=2, elem_classes="card"):
                        question_input = gr.Textbox(
                            label="Medical Question",
                            placeholder="e.g., What are current clinical guidelines for immunotherapy in NSCLC?",
                            lines=4,
                        )
                        with gr.Row():
                            reranker_toggle = gr.Checkbox(
                                label="Use Cross-Encoder Re-Ranking",
                                value=True,
                            )
                        submit_btn = gr.Button(
                            "Get Evidence-Based Answer",
                            variant="primary",
                            elem_classes="primary-btn",
                        )

                    with gr.Column(scale=3, elem_classes="card"):
                        gr.Markdown("### 📝 Answer")
                        answer_output = gr.Markdown(label="Answer")
                        gr.Markdown("### 📚 Source Evidence")
                        sources_output = gr.HTML(label="Sources")

                submit_btn.click(
                    fn=answer_question,
                    inputs=[question_input, reranker_toggle],
                    outputs=[answer_output, sources_output],
                )

            with gr.TabItem("📤 Upload Document"):
                with gr.Row():
                    with gr.Column(elem_classes="card"):
                        gr.Markdown("### Add New Research to Knowledge Base")
                        file_upload = gr.File(
                            label="Upload Medical PDF",
                            file_types=[".pdf"],
                        )
                        upload_btn = gr.Button("Extract & Index", variant="primary", elem_classes="primary-btn")
                    
                    with gr.Column(elem_classes="card"):
                        gr.Markdown("### Processing Logs")
                        upload_result = gr.Markdown()

                upload_btn.click(
                    fn=upload_pdf,
                    inputs=[file_upload],
                    outputs=[upload_result],
                )

        gr.HTML(
            """
            <div style='text-align: center; margin-top: 2rem; opacity: 0.6; font-size: 0.8rem;'>
                Built with BGE-M3 + CLIP + FAISS + GPT-4o | Evidence-based Medical Document RAG
            </div>
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=settings.gradio_port,
        share=False,
        theme=gr.themes.Default(), # Base theme overridden by custom CSS
    )

