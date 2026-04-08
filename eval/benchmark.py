"""
Benchmark comparison — BM25 vs dense-only vs multimodal.

Generates comparison tables and charts for the research paper.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

def _metric_from_result(result, metric_name: str) -> float:
    """Safely extract a metric mean from RAGAS EvaluationResult."""
    try:
        df = result.to_pandas()
        if metric_name in df.columns:
            value = float(df[metric_name].dropna().mean())
            return 0.0 if value != value else value  # NaN guard
    except Exception as exc:
        logger.warning("Failed to parse metric %s: %s", metric_name, exc)
    return 0.0


@dataclass
class BenchmarkResult:
    """Results for a single retrieval strategy."""
    strategy_name: str
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    avg_latency_ms: float = 0.0
    recall_at_5: float = 0.0


def run_bm25_baseline(
    eval_samples: list,
    corpus_texts: list[str],
    rag_chain=None,
) -> BenchmarkResult:
    """
    Run BM25 baseline retrieval for comparison.

    Args:
        eval_samples: List of EvalSample objects.
        corpus_texts: List of all text chunks in the corpus.
        rag_chain: MedicalRAGChain for generation.

    Returns:
        BenchmarkResult with BM25 metrics.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank_bm25 not installed. Install with: pip install rank-bm25")
        return BenchmarkResult(strategy_name="BM25")

    if not eval_samples or not corpus_texts or rag_chain is None:
        logger.warning("BM25 benchmark skipped due to missing samples/corpus/chain")
        return BenchmarkResult(strategy_name="BM25")

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
    )
    from retrieval.context_builder import build_context

    # Tokenize corpus
    tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    logger.info(f"BM25 index built with {len(corpus_texts)} documents")

    questions = []
    answers = []
    contexts = []
    ground_truths = []
    latency_ms = []

    for sample in eval_samples:
        q = sample.question.strip()
        query_tokens = q.lower().split()
        start = time.perf_counter()

        scores = bm25.get_scores(query_tokens)
        # Top-5 lexical contexts
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:5]
        top_contexts = [corpus_texts[i] for i in top_indices]

        pseudo_results = []
        for i, ctx in enumerate(top_contexts, start=1):
            pseudo_results.append(
                type(
                    "BM25Result",
                    (),
                    {
                        "content_type": "text",
                        "source_file": "bm25_corpus",
                        "page_number": i,
                        "text": ctx,
                    },
                )()
            )
        context_str = build_context(pseudo_results, include_images=False)
        gen_result = rag_chain.generate(q, context_str)

        elapsed = (time.perf_counter() - start) * 1000
        latency_ms.append(elapsed)

        questions.append(q)
        answers.append(gen_result.answer)
        contexts.append(top_contexts)
        ground_truths.append(sample.ground_truth)

    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    result = evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    return BenchmarkResult(
        strategy_name="BM25",
        faithfulness=_metric_from_result(result, "faithfulness"),
        answer_relevancy=_metric_from_result(result, "answer_relevancy"),
        context_precision=_metric_from_result(result, "context_precision"),
        recall_at_5=0.0,  # lexical baseline uses synthetic corpus mapping
        avg_latency_ms=sum(latency_ms) / max(1, len(latency_ms)),
    )


def run_comparison(
    strategies: list[BenchmarkResult],
    output_path: str = "eval/benchmark_comparison.json",
):
    """
    Save comparison results to JSON for paper figures.

    Args:
        strategies: List of BenchmarkResult for each strategy.
        output_path: Path to save results.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = [asdict(s) for s in strategies]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print comparison table
    header = (
        f"{'Strategy':<20} {'Faithful':>10} {'Relevancy':>10} "
        f"{'Precision':>10} {'Recall@5':>10} {'Latency':>10}"
    )
    logger.info(f"\n{'=' * 70}\n{header}\n{'=' * 70}")

    for s in strategies:
        row = (
            f"{s.strategy_name:<20} {s.faithfulness:>10.4f} "
            f"{s.answer_relevancy:>10.4f} {s.context_precision:>10.4f} "
            f"{s.recall_at_5:>10.4f} {s.avg_latency_ms:>8.0f}ms"
        )
        logger.info(row)

    logger.info(f"{'=' * 70}")
    logger.info(f"Benchmark results saved to {output_path}")


def generate_comparison_chart(
    strategies: list[BenchmarkResult],
    output_path: str = "eval/benchmark_chart.png",
):
    """
    Generate a comparison bar chart for the paper.

    Args:
        strategies: List of BenchmarkResult objects.
        output_path: Path to save the chart image.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        metrics = ["faithfulness", "answer_relevancy", "context_precision", "recall_at_5"]
        labels = ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision", "Recall@5"]

        x = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, strategy in enumerate(strategies):
            values = [getattr(strategy, m) for m in metrics]
            offset = (i - len(strategies) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=strategy.strategy_name)
            ax.bar_label(bars, fmt="%.2f", fontsize=8)

        ax.set_ylabel("Score")
        ax.set_title("Retrieval Strategy Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Comparison chart saved to {output_path}")

    except ImportError:
        logger.warning("matplotlib not installed — skipping chart generation")
