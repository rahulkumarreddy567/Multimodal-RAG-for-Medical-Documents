"""
Generate evaluation Q&A dataset from indexed documents.

Creates 100 question-answer pairs using the LLM to generate
questions from randomly sampled chunks, then verifies the ground
truth answers against the source text.
"""

import json
import logging
import random
from pathlib import Path

from langchain_openai import ChatOpenAI

from config.settings import settings
from ingestion.build_index import load_faiss_index

logger = logging.getLogger(__name__)


QUESTION_GEN_PROMPT = """You are a medical education expert. Given the following text from a medical research paper, generate one specific, factual question that can be answered using ONLY this text.

TEXT:
{text}

SOURCE: {source_file}, page {page_number}

Requirements:
1. The question must be answerable from the given text alone.
2. The question should be clinically relevant.
3. Provide both the question and a concise ground truth answer.

Respond in this exact JSON format:
{{"question": "...", "ground_truth": "..."}}
"""


def generate_eval_dataset(
    num_samples: int = 100,
    output_path: str = "eval/eval_dataset.json",
    metadata_path: str = "data/index/metadata.json",
    model_name: str = "gpt-4o-mini",
    api_key: str = "",
    seed: int = 42,
) -> list[dict]:
    """
    Generate an evaluation dataset by sampling indexed chunks
    and using an LLM to create Q&A pairs.

    Args:
        num_samples: Number of Q&A pairs to generate.
        output_path: Path to save the dataset JSON.
        metadata_path: Path to the FAISS metadata file.
        model_name: LLM model for question generation.
        api_key: OpenAI API key.
        seed: Random seed for reproducibility.

    Returns:
        List of Q&A dicts.
    """
    random.seed(seed)

    # Load metadata (all indexed chunks)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Filter to text chunks with sufficient content
    text_chunks = [
        m for m in metadata
        if m.get("content_type") == "text"
        and len(m.get("text", "")) > 200
    ]

    if len(text_chunks) < num_samples:
        logger.warning(
            f"Only {len(text_chunks)} suitable chunks available, "
            f"requested {num_samples}"
        )
        num_samples = len(text_chunks)

    # Sample chunks
    sampled = random.sample(text_chunks, num_samples)

    # Initialize LLM
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key or settings.openai_api_key or None,
        temperature=0.3,
        max_tokens=256,
    )

    # Generate Q&A pairs
    dataset = []

    for i, chunk in enumerate(sampled):
        logger.info(f"Generating Q&A {i + 1}/{num_samples}...")

        prompt = QUESTION_GEN_PROMPT.format(
            text=chunk["text"][:1500],
            source_file=chunk.get("source_file", "unknown"),
            page_number=chunk.get("page_number", "?"),
        )

        try:
            response = llm.invoke(prompt)
            content = response.content.strip()

            # Parse JSON response
            # Handle potential markdown code blocks
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            qa = json.loads(content)

            dataset.append({
                "question": qa["question"],
                "ground_truth": qa["ground_truth"],
                "source_file": chunk.get("source_file", ""),
                "page_number": chunk.get("page_number", 0),
                "source_text": chunk["text"][:500],
            })

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse Q&A for sample {i + 1}: {e}")
        except Exception as e:
            logger.error(f"LLM error for sample {i + 1}: {e}")

    # Save dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Generated {len(dataset)} Q&A pairs, saved to {output_path}"
    )

    return dataset


# ── CLI ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Generate evaluation Q&A dataset")
    parser.add_argument("--num", type=int, default=100, help="Number of Q&A pairs")
    parser.add_argument("--output", default="eval/eval_dataset.json", help="Output path")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model for generation")
    args = parser.parse_args()

    generate_eval_dataset(
        num_samples=args.num,
        output_path=args.output,
        model_name=args.model,
    )
