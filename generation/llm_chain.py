"""
LangChain RAG chain with medical-specific prompt template.

Generates grounded answers with citations from retrieved context.
"""

import logging
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# Medical RAG system prompt — enforces citation and factual grounding
SYSTEM_PROMPT = """You are a medical research assistant specializing in evidence-based answers.
You are given context from medical research papers and clinical documents.

RULES:
1. Answer the question ONLY using the provided context.
2. If the context does not contain enough information, say "I cannot find sufficient evidence in the provided sources to answer this question."
3. ALWAYS cite your sources using the format [Source N] where N matches the source number in the context.
4. When referencing specific findings, include the page number: [Source N, p.X].
5. If the context includes image descriptions, reference them as [Source N, Image].
6. Be precise and clinical in your language.
7. Do NOT hallucinate or add information not present in the context.
8. Structure your answer with clear paragraphs for complex questions.

CONTEXT:
{context}
"""

USER_PROMPT = """Question: {question}

Please provide a grounded, evidence-based answer with citations to the sources above."""


@dataclass
class GenerationResult:
    """Result from the LLM generation."""
    answer: str
    sources: list[dict] = field(default_factory=list)
    model: str = ""
    tokens_used: int = 0


class MedicalRAGChain:
    """LangChain-based RAG chain for medical document QA."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        provider: str = "openai",
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider.lower()

        if self.provider == "local" or (self.provider == "openai" and not api_key):
            try:
                from langchain_community.chat_models import ChatOllama
            except ImportError:
                # Fallback to older import path if required
                from langchain_community.chat_models.ollama import ChatOllama
                
            # Use local free model via Ollama (e.g., Llama-3)
            fallback_model = "llama3" if self.provider == "openai" else model_name
            self._llm = ChatOllama(
                model=fallback_model,
                temperature=temperature,
            )
            self.model_name = fallback_model
            logger.info(f"Using local ChatOllama model: {fallback_model} (Warning: make sure Ollama is installed and running!)")
        else:
            self._llm = ChatOpenAI(
                model=model_name,
                api_key=api_key or None,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        self._prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ])

        logger.info(f"RAG chain initialized with model: {self.model_name}")

    def generate(
        self,
        question: str,
        context: str,
        sources: list[dict] | None = None,
    ) -> GenerationResult:
        """
        Generate a grounded answer from context.

        Args:
            question: The user's question.
            context: Formatted context string from context_builder.
            sources: Optional source metadata list.

        Returns:
            GenerationResult with answer and source info.
        """
        try:
            # Format the prompt
            messages = self._prompt.format_messages(
                context=context,
                question=question,
            )

            # Generate
            response = self._llm.invoke(messages)

            result = GenerationResult(
                answer=response.content,
                sources=sources or [],
                model=self.model_name,
                tokens_used=response.response_metadata.get(
                    "token_usage", {}
                ).get("total_tokens", 0),
            )

            logger.info(
                f"Generated answer ({len(result.answer)} chars, "
                f"{result.tokens_used} tokens)"
            )

            return result

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return GenerationResult(
                answer=f"Error generating answer: {str(e)}",
                sources=sources or [],
                model=self.model_name,
            )

    def generate_with_retrieval(
        self,
        question: str,
        retriever,
        embedder,
        reranker=None,
        top_k: int = 20,
        top_n: int = 5,
    ) -> GenerationResult:
        """
        End-to-end: query → retrieve → rerank → generate.

        Args:
            question: User question.
            retriever: FAISSRetriever instance.
            embedder: TextEmbedder or MultiModalEmbedder.
            reranker: Optional Reranker instance.
            top_k: FAISS retrieval count.
            top_n: Re-ranker output count.

        Returns:
            GenerationResult with answer and sources.
        """
        from retrieval.context_builder import build_context, get_source_list

        # Retrieve
        results = retriever.search_text(question, embedder, k=top_k)

        # Re-rank (if available)
        if reranker and results:
            results = reranker.rerank(question, results, top_n=top_n)

        # Build context
        context = build_context(results)
        sources = get_source_list(results)

        # Generate
        return self.generate(question, context, sources)
