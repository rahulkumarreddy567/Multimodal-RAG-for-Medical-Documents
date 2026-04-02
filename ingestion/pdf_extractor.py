"""
PDF text and table extraction using PyMuPDF (fitz) with pdfplumber fallback.

Returns structured text blocks with page numbers for citation tracking.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """A block of text extracted from a PDF page."""
    text: str
    page_number: int
    source_file: str
    block_type: str = "text"  # "text" or "table"
    bbox: tuple = ()  # (x0, y0, x1, y1) bounding box


@dataclass
class ExtractionResult:
    """Complete extraction result for a single PDF."""
    source_file: str
    text_blocks: list[TextBlock] = field(default_factory=list)
    total_pages: int = 0
    metadata: dict = field(default_factory=dict)


def extract_text(pdf_path: str | Path) -> ExtractionResult:
    """
    Extract all text from a PDF file using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        ExtractionResult with structured text blocks and metadata.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    result = ExtractionResult(
        source_file=pdf_path.name,
        text_blocks=[],
    )

    try:
        doc = fitz.open(str(pdf_path))
        result.total_pages = len(doc)
        result.metadata = dict(doc.metadata) if doc.metadata else {}

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Extract text blocks with position info
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

            for block in blocks:
                if block.get("type") == 0:  # Text block
                    # Concatenate all spans in all lines
                    text_lines = []
                    for line in block.get("lines", []):
                        spans_text = "".join(
                            span.get("text", "") for span in line.get("spans", [])
                        )
                        if spans_text.strip():
                            text_lines.append(spans_text.strip())

                    full_text = " ".join(text_lines)
                    if full_text.strip() and len(full_text.strip()) > 10:
                        result.text_blocks.append(
                            TextBlock(
                                text=full_text.strip(),
                                page_number=page_num + 1,  # 1-indexed
                                source_file=pdf_path.name,
                                block_type="text",
                                bbox=tuple(block.get("bbox", ())),
                            )
                        )

        doc.close()
        logger.info(
            f"Extracted {len(result.text_blocks)} text blocks from "
            f"{result.total_pages} pages: {pdf_path.name}"
        )

    except Exception as e:
        logger.error(f"PyMuPDF extraction failed for {pdf_path.name}: {e}")
        # Fallback to pdfplumber
        result = _fallback_pdfplumber(pdf_path)

    return result


def extract_tables(pdf_path: str | Path) -> list[TextBlock]:
    """
    Extract tables from a PDF using pdfplumber (better table detection).

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of TextBlock objects containing table text.
    """
    pdf_path = Path(pdf_path)
    table_blocks = []

    try:
        import pdfplumber

        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        # Convert table to readable text
                        rows = []
                        for row in table:
                            cleaned = [
                                str(cell).strip() if cell else ""
                                for cell in row
                            ]
                            rows.append(" | ".join(cleaned))
                        table_text = "\n".join(rows)

                        if table_text.strip():
                            table_blocks.append(
                                TextBlock(
                                    text=table_text,
                                    page_number=page_num + 1,
                                    source_file=pdf_path.name,
                                    block_type="table",
                                )
                            )

        logger.info(f"Extracted {len(table_blocks)} tables from {pdf_path.name}")

    except ImportError:
        logger.warning("pdfplumber not installed — skipping table extraction")
    except Exception as e:
        logger.error(f"Table extraction failed for {pdf_path.name}: {e}")

    return table_blocks


def _fallback_pdfplumber(pdf_path: Path) -> ExtractionResult:
    """Fallback extraction using pdfplumber when PyMuPDF fails."""
    result = ExtractionResult(source_file=pdf_path.name)

    try:
        import pdfplumber

        with pdfplumber.open(str(pdf_path)) as pdf:
            result.total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    result.text_blocks.append(
                        TextBlock(
                            text=text.strip(),
                            page_number=page_num + 1,
                            source_file=pdf_path.name,
                            block_type="text",
                        )
                    )

        logger.info(f"Fallback extraction got {len(result.text_blocks)} blocks")

    except Exception as e:
        logger.error(f"Fallback extraction also failed: {e}")

    return result


def extract_full(pdf_path: str | Path) -> ExtractionResult:
    """
    Full extraction: text blocks + tables merged and sorted by page.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        ExtractionResult with both text and table blocks.
    """
    result = extract_text(pdf_path)
    tables = extract_tables(pdf_path)
    result.text_blocks.extend(tables)

    # Sort by page number
    result.text_blocks.sort(key=lambda b: b.page_number)

    return result
