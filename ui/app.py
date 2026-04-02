"""
Gradio UI — multimodal interface with text/image upload and cited answers.
"""

import logging

import gradio as gr

from config.settings import settings

logger = logging.getLogger(__name__)


def create_gradio_app():
    """Build and return the Gradio interface."""

    # ── Lazy-load pipeline ────────────────────────────────
    _pipeline = {}

    def _get_pipeline():
        if not _pipeline:
            from ingestion.embedder import MultiModalEmbedder
            from retrieval.retriever import FAISSRetriever
            from retrieval.reranker import Reranker
            from generation.llm_chain import MedicalRAGChain
            from generation.citation_formatter import format_inline_citations

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
            )
            _pipeline["formatter"] = format_inline_citations
        return _pipeline

    # ── Query handler ─────────────────────────────────────

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
                f"• {s['file']}, p.{s['page']} ({s['type']}) — score: {s['score']:.3f}"
                for s in result.sources
            )

            return formatted, source_text

        except Exception as e:
            logger.error(f"Query error: {e}", exc_info=True)
            return f"Error: {str(e)}", ""

    # ── Upload handler ────────────────────────────────────

    def upload_pdf(file):
        """Upload and ingest a PDF document."""
        if file is None:
            return "No file uploaded."

        try:
            from ingestion.pdf_extractor import extract_full
            from ingestion.image_extractor import extract_images
            from ingestion.chunker import chunk_text_blocks

            extraction = extract_full(file.name)
            images = extract_images(file.name)
            chunks = chunk_text_blocks(extraction.text_blocks)

            return (
                f"✅ **Processed:** {file.name}\n\n"
                f"- Pages: {extraction.total_pages}\n"
                f"- Text chunks: {len(chunks)}\n"
                f"- Images extracted: {len(images)}\n\n"
                f"Document is ready for querying."
            )
        except Exception as e:
            return f"❌ Error processing file: {str(e)}"

    # ── Build Gradio interface ────────────────────────────

    with gr.Blocks(
        title="Medical RAG — Multimodal Document QA",
        theme=gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="blue",
        ),
    ) as demo:
        gr.Markdown(
            """
            # 🏥 Multimodal Medical RAG
            ### Evidence-based answers from medical documents with citations

            Ask questions about medical research papers. Answers are grounded
            in source documents with page-level citations.
            """
        )

        with gr.Tabs():
            # ── Query Tab ─────────────────────────────────
            with gr.TabItem("🔍 Ask a Question"):
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
                            "🔎 Get Answer",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=3):
                        answer_output = gr.Markdown(label="Answer")
                        sources_output = gr.Textbox(
                            label="📚 Sources",
                            lines=5,
                            interactive=False,
                        )

                submit_btn.click(
                    fn=answer_question,
                    inputs=[question_input, reranker_toggle],
                    outputs=[answer_output, sources_output],
                )

            # ── Upload Tab ────────────────────────────────
            with gr.TabItem("📄 Upload Document"):
                gr.Markdown(
                    "Upload a medical PDF to add it to the knowledge base."
                )
                file_upload = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                )
                upload_btn = gr.Button("📥 Process & Index", variant="primary")
                upload_result = gr.Markdown()

                upload_btn.click(
                    fn=upload_pdf,
                    inputs=[file_upload],
                    outputs=[upload_result],
                )

        gr.Markdown(
            """
            ---
            *Built with BGE-M3 + CLIP embeddings, FAISS retrieval,
            cross-encoder re-ranking, and LLaMA-3/GPT-4V generation.*
            """
        )

    return demo


# ── Entrypoint ────────────────────────────────────────────

if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=settings.gradio_port,
        share=False,
    )
