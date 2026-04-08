"""Run zero-API-cost retrieval benchmarks (BM25, dense, multimodal)."""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from config.settings import settings
from eval.evaluate import load_eval_dataset
from ingestion.build_index import load_faiss_index
from ingestion.embedder import MultiModalEmbedder
from retrieval.reranker import Reranker
from retrieval.retriever import FAISSRetriever


logger = logging.getLogger(__name__)


@dataclass
class RetrievalBenchmarkResult:
    strategy_name: str
    precision: float
    recall: float
    accuracy: float
    f1_score: float


def _safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def _compute_metrics(tp: int, fp: int, fn: int, tn: int) -> tuple[float, float, float, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    f1_score = _safe_div(2 * precision * recall, precision + recall)
    return precision, recall, accuracy, f1_score


def _build_bm25(corpus_texts: list[str]):
    try:
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
        model = BM25Okapi(tokenized_corpus)

        def search_fn(query: str, k: int):
            scores = model.get_scores(query.lower().split())
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            return ranked

        return search_fn
    except Exception:
        logger.warning("rank_bm25 unavailable; using token-overlap fallback BM25 approximation")

        corpus_tokens = [set(doc.lower().split()) for doc in corpus_texts]

        def search_fn(query: str, k: int):
            q_tokens = set(query.lower().split())
            scores = [len(q_tokens.intersection(toks)) for toks in corpus_tokens]
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            return ranked

        return search_fn


def _attach_metadata(eval_data_path: Path, samples):
    """Attach source_file/page_number metadata when available."""
    data = json.loads(eval_data_path.read_text(encoding="utf-8"))
    for sample, raw in zip(samples, data):
        sample.metadata = {
            "source_file": raw.get("source_file", ""),
            "page_number": raw.get("page_number", None),
        }


def _evaluate_bm25(samples, metadata: list[dict], top_k: int = 5) -> RetrievalBenchmarkResult:
    corpus_texts = [m.get("text", "") for m in metadata if m.get("content_type") == "text" and m.get("text")]
    corpus_meta = [m for m in metadata if m.get("content_type") == "text" and m.get("text")]
    bm25_search = _build_bm25(corpus_texts)

    tp = fp = fn = tn = 0
    corpus_size = len(corpus_meta)
    for sample in samples:
        expected_file = sample.metadata.get("source_file", "")
        expected_page = sample.metadata.get("page_number", None)
        retrieved_idx = bm25_search(sample.question, top_k)
        hit = 0
        for idx in retrieved_idx:
            m = corpus_meta[idx]
            if m.get("source_file") == expected_file and (
                expected_page is None or m.get("page_number") == expected_page
            ):
                hit = 1
                break
        tp += hit
        fn += 1 - hit
        fp += max(0, top_k - hit)
        tn += max(0, corpus_size - top_k - (1 - hit))

    p, r, a, f1 = _compute_metrics(tp, fp, fn, tn)
    return RetrievalBenchmarkResult("BM25", p, r, a, f1)


def _evaluate_dense(samples, retriever, embedder, reranker=None, top_k: int = 5) -> RetrievalBenchmarkResult:
    tp = fp = fn = tn = 0
    corpus_size = retriever.index_size
    for sample in samples:
        expected_file = sample.metadata.get("source_file", "")
        expected_page = sample.metadata.get("page_number", None)
        results = retriever.search_text(sample.question, embedder, k=max(20, top_k))
        results = reranker.rerank(sample.question, results, top_n=top_k) if reranker else results[:top_k]
        hit = any(
            r.source_file == expected_file and (expected_page is None or r.page_number == expected_page)
            for r in results
        )
        hit_int = 1 if hit else 0
        tp += hit_int
        fn += 1 - hit_int
        fp += max(0, top_k - hit_int)
        tn += max(0, corpus_size - top_k - (1 - hit_int))

    p, r, a, f1 = _compute_metrics(tp, fp, fn, tn)
    name = "Multimodal (Ours)" if reranker else "Dense-only (BGE-M3)"
    return RetrievalBenchmarkResult(name, p, r, a, f1)


def _save_results(results: list[RetrievalBenchmarkResult], output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info("Benchmark results saved to %s", path)


def main():
    parser = argparse.ArgumentParser(description="Run benchmark comparison for Medical RAG")
    parser.add_argument("--dataset", default="eval/eval_dataset.json", help="Evaluation dataset path")
    parser.add_argument("--output", default="eval/benchmark_comparison.json", help="Benchmark JSON output")
    parser.add_argument("--num", type=int, default=30, help="Number of eval samples to use")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval cutoff for metrics")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    eval_data_path = Path(args.dataset)
    if not eval_data_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found at {args.dataset}. "
            "Run scripts/seed_eval_dataset.py first."
        )

    samples = load_eval_dataset(args.dataset)
    if not samples:
        raise RuntimeError("Evaluation dataset is empty.")
    samples = samples[: args.num]
    _attach_metadata(eval_data_path, samples)

    embedder = MultiModalEmbedder(
        text_model=settings.text_embedding_model,
        image_model="ViT-L-14",
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    retriever = FAISSRetriever(
        index_path=settings.faiss_index_path,
        metadata_path=settings.faiss_metadata_path,
    )
    reranker = Reranker(model_name=settings.reranker_model, device=settings.embedding_device)
    _, metadata = load_faiss_index(settings.faiss_index_path, settings.faiss_metadata_path)

    logger.info("Running BM25 baseline (retrieval-only)...")
    bm25 = _evaluate_bm25(samples, metadata, top_k=args.top_k)
    logger.info("Running dense-only baseline (retrieval-only)...")
    dense = _evaluate_dense(samples, retriever, embedder, reranker=None, top_k=args.top_k)
    logger.info("Running multimodal baseline (retrieval-only)...")
    multimodal = _evaluate_dense(samples, retriever, embedder, reranker=reranker, top_k=args.top_k)

    _save_results([bm25, dense, multimodal], args.output)


if __name__ == "__main__":
    main()
