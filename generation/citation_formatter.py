"""
Citation formatter — transforms raw source metadata into
human-readable citations with page references and figure links.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def format_citations(sources: list[dict]) -> str:
    """
    Format source metadata into a citation block.

    Args:
        sources: List of source dicts with 'file', 'page', 'type', 'score'.

    Returns:
        Formatted citation string.
    """
    if not sources:
        return "No sources cited."

    lines = ["\n**Sources:**"]

    for i, source in enumerate(sources, 1):
        file_name = source.get("file", "Unknown")
        page = source.get("page", "?")
        content_type = source.get("type", "text")
        score = source.get("score", 0)

        # Build citation line
        if content_type == "image":
            citation = f"- **[{i}]** {file_name}, p.{page} (Image) — relevance: {score:.2f}"
        else:
            citation = f"- **[{i}]** {file_name}, p.{page} — relevance: {score:.2f}"

        lines.append(citation)

    return "\n".join(lines)


def format_inline_citations(
    answer: str,
    sources: list[dict],
) -> str:
    """
    Append a formatted citation block to an answer.

    Args:
        answer: The generated answer text (may already contain [Source N] refs).
        sources: Source metadata list.

    Returns:
        Answer with citation footer appended.
    """
    citations = format_citations(sources)
    return f"{answer}\n\n---\n{citations}"


def extract_cited_sources(
    answer: str,
    sources: list[dict],
) -> list[dict]:
    """
    Extract only the sources that are actually cited in the answer.

    Looks for [Source N] or [N] patterns in the answer text.

    Args:
        answer: Generated answer text.
        sources: Full source metadata list.

    Returns:
        Filtered list of only cited sources.
    """
    import re

    # Find all [Source N] or [N] patterns
    pattern = r'\[(?:Source\s+)?(\d+)(?:,\s*p\.\d+)?(?:,\s*Image)?\]'
    matches = re.findall(pattern, answer)

    cited_indices = set(int(m) for m in matches)

    cited = []
    for i, source in enumerate(sources, 1):
        if i in cited_indices:
            cited.append(source)

    logger.info(
        f"Answer cites {len(cited)}/{len(sources)} sources: "
        f"indices {sorted(cited_indices)}"
    )

    return cited
