"""
Image extraction from PDF documents.

Pulls embedded images (X-rays, charts, scans) from PDFs,
filters out tiny icons/artifacts, and saves as PNG with metadata.
"""

import logging
from pathlib import Path
from dataclasses import dataclass

import fitz  # PyMuPDF
from PIL import Image
import io

logger = logging.getLogger(__name__)

# Minimum dimensions to filter out icons and PDF artifacts
MIN_WIDTH = 50
MIN_HEIGHT = 50
MIN_AREA = 5000  # width * height


@dataclass
class ExtractedImage:
    """An image extracted from a PDF."""
    image_path: str
    page_number: int
    source_file: str
    width: int
    height: int
    image_index: int  # index within the page


def extract_images(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    min_width: int = MIN_WIDTH,
    min_height: int = MIN_HEIGHT,
) -> list[ExtractedImage]:
    """
    Extract embedded images from a PDF file.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to save extracted images.
                    Defaults to data/processed/images/<pdf_stem>/
        min_width: Minimum image width to keep (filters icons).
        min_height: Minimum image height to keep.

    Returns:
        List of ExtractedImage objects with saved file paths.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Setup output directory
    if output_dir is None:
        output_dir = Path("data/processed/images") / pdf_path.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted = []
    doc = fitz.open(str(pdf_path))

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image["image"]
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                # Filter small images (icons, bullets, etc.)
                if width < min_width or height < min_height:
                    continue
                if width * height < MIN_AREA:
                    continue

                # Convert to PIL Image and save as PNG
                image = Image.open(io.BytesIO(image_bytes))
                if image.mode != "RGB":
                    image = image.convert("RGB")

                # Filename: <pdf_stem>_p<page>_img<idx>.png
                filename = f"{pdf_path.stem}_p{page_num + 1}_img{img_idx}.png"
                save_path = output_dir / filename
                image.save(save_path, "PNG")

                extracted.append(
                    ExtractedImage(
                        image_path=str(save_path),
                        page_number=page_num + 1,
                        source_file=pdf_path.name,
                        width=width,
                        height=height,
                        image_index=img_idx,
                    )
                )

            except Exception as e:
                logger.warning(
                    f"Failed to extract image {img_idx} from page {page_num + 1} "
                    f"of {pdf_path.name}: {e}"
                )

    doc.close()

    logger.info(
        f"Extracted {len(extracted)} images from {pdf_path.name} "
        f"(filtered by {min_width}x{min_height} minimum)"
    )

    return extracted


def extract_images_batch(
    pdf_dir: str | Path,
    output_base_dir: str | Path | None = None,
) -> dict[str, list[ExtractedImage]]:
    """
    Extract images from all PDFs in a directory.

    Args:
        pdf_dir: Directory containing PDF files.
        output_base_dir: Base directory for extracted images.

    Returns:
        Dict mapping PDF filename → list of ExtractedImage.
    """
    pdf_dir = Path(pdf_dir)
    results = {}

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    logger.info(f"Processing {len(pdf_files)} PDFs from {pdf_dir}")

    for pdf_path in pdf_files:
        out_dir = None
        if output_base_dir:
            out_dir = Path(output_base_dir) / pdf_path.stem

        try:
            images = extract_images(pdf_path, output_dir=out_dir)
            results[pdf_path.name] = images
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            results[pdf_path.name] = []

    total = sum(len(imgs) for imgs in results.values())
    logger.info(f"Total images extracted: {total} from {len(pdf_files)} PDFs")

    return results
