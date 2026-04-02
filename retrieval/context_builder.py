"""
Context builder — merges retrieved text chunks and image descriptions
into a unified context string for the LLM.
"""

import logging

logger = logging.getLogger(__name__)


def build_context(
    results: list,
    max_context_tokens: int = 3000,
    include_images: bool = True,
) -> str:
    """
    Build a context string from retrieval results.

    Interleaves text chunks and image descriptions, formatted
    with source citations for the LLM prompt.

    Args:
        results: List of RetrievalResult objects (after re-ranking).
        max_context_tokens: Approximate max tokens for the context.
        include_images: Whether to include image-type results.

    Returns:
        Formatted context string with source attributions.
    """
    if not results:
        return "No relevant context found."

    context_parts = []
    total_chars = 0
    max_chars = max_context_tokens * 4  # rough char-to-token ratio

    for i, result in enumerate(results):
        if not include_images and result.content_type == "image":
            continue

        # Format each source
        if result.content_type == "image":
            source_label = (
                f"[Source {i + 1}: {result.source_file}, "
                f"p.{result.page_number}, Image]"
            )
            content = f"{source_label}\n{result.text}"
        else:
            source_label = (
                f"[Source {i + 1}: {result.source_file}, "
                f"p.{result.page_number}]"
            )
            content = f"{source_label}\n{result.text}"

        # Check token budget
        if total_chars + len(content) > max_chars:
            # Truncate this chunk to fit
            remaining = max_chars - total_chars
            if remaining > 100:
                content = content[:remaining] + "..."
                context_parts.append(content)
            break

        context_parts.append(content)
        total_chars += len(content)

    context = "\n\n---\n\n".join(context_parts)

    logger.info(
        f"Built context from {len(context_parts)} sources "
        f"(~{total_chars // 4} tokens)"
    )

    return context


def get_source_list(results: list) -> list[dict]:
    """
    Extract a clean list of sources from retrieval results.

    Args:
        results: List of RetrievalResult objects.

    Returns:
        List of source dicts with file, page, and type info.
    """
    sources = []
    seen = set()

    for r in results:
        key = (r.source_file, r.page_number)
        if key not in seen:
            sources.append({
                "file": r.source_file,
                "page": r.page_number,
                "type": r.content_type,
                "score": round(r.score, 4),
            })
            seen.add(key)

    return sources
