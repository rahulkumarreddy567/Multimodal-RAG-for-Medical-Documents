"""
RAGAS evaluation — faithfulness, answer relevancy, context precision.

Runs the evaluation on a Q&A dataset and produces metrics for the paper.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """A single evaluation sample."""
    question: str
    ground_truth: str
    answer: str = ""
    contexts: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Aggregated evaluation results."""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    num_samples: int = 0
    per_sample: list[dict] = field(default_factory=list)


def load_eval_dataset(path: str = "eval/eval_dataset.json") -> list[EvalSample]:
    """
    Load evaluation Q&A dataset.

    Expected JSON format:
    [
        {
            "question": "...",
            "ground_truth": "...",
            "contexts": ["..."]  // optional, populated during eval
        }
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = [
        EvalSample(
            question=item["question"],
            ground_truth=item["ground_truth"],
            contexts=item.get("contexts", []),
        )
        for item in data
    ]

    logger.info(f"Loaded {len(samples)} evaluation samples from {path}")
    return samples


def run_ragas_eval(
    samples: list[EvalSample],
    rag_chain=None,
    retriever=None,
    embedder=None,
    reranker=None,
) -> EvalResult:
    """
    Run RAGAS evaluation on the provided samples.

    For each sample:
    1. Run the RAG pipeline to get answer + retrieved contexts
    2. Evaluate with RAGAS metrics

    Args:
        samples: List of EvalSample with questions and ground truths.
        rag_chain: MedicalRAGChain instance.
        retriever: FAISSRetriever instance.
        embedder: MultiModalEmbedder instance.
        reranker: Optional Reranker instance.

    Returns:
        EvalResult with aggregated metrics.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from retrieval.context_builder import build_context

    # Generate answers for each sample
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, sample in enumerate(samples):
        logger.info(f"Evaluating sample {i + 1}/{len(samples)}: {sample.question[:50]}...")

        # Retrieve
        results = retriever.search_text(sample.question, embedder, k=20)
        if reranker:
            results = reranker.rerank(sample.question, results, top_n=5)

        # Get contexts
        sample_contexts = [r.text for r in results if r.text]

        # Generate answer
        context_str = build_context(results)
        gen_result = rag_chain.generate(sample.question, context_str)

        questions.append(sample.question)
        answers.append(gen_result.answer)
        contexts.append(sample_contexts)
        ground_truths.append(sample.ground_truth)

    # Build RAGAS dataset
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # Run RAGAS
    logger.info("Running RAGAS evaluation...")
    result = evaluate(
        eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    eval_result = EvalResult(
        faithfulness=float(result.get("faithfulness", 0)),
        answer_relevancy=float(result.get("answer_relevancy", 0)),
        context_precision=float(result.get("context_precision", 0)),
        context_recall=float(result.get("context_recall", 0)),
        num_samples=len(samples),
    )

    logger.info(
        f"RAGAS Results:\n"
        f"  Faithfulness:      {eval_result.faithfulness:.4f}\n"
        f"  Answer Relevancy:  {eval_result.answer_relevancy:.4f}\n"
        f"  Context Precision: {eval_result.context_precision:.4f}\n"
        f"  Context Recall:    {eval_result.context_recall:.4f}"
    )

    return eval_result


def save_eval_results(
    result: EvalResult,
    output_path: str = "eval/eval_results.json",
):
    """Save evaluation results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)

    logger.info(f"Evaluation results saved to {output_path}")
