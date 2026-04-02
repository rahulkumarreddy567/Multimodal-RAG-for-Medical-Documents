"""
PubMed Open Access bulk downloader.

Downloads PDFs from PubMed Central Open Access subset.
Targets cardiology/radiology papers for image-rich content.
"""

import os
import logging
import requests
from pathlib import Path
from time import sleep

logger = logging.getLogger(__name__)

# PubMed E-utilities API
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PMC_OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

# Default search query for image-rich medical papers
DEFAULT_QUERY = (
    "(cardiology[MeSH] OR radiology[MeSH] OR diagnostic imaging[MeSH]) "
    "AND open access[filter] AND pdf[filter]"
)


def search_pubmed(
    query: str = DEFAULT_QUERY,
    max_results: int = 500,
    api_key: str = "",
) -> list[str]:
    """
    Search PubMed for article IDs matching the query.

    Args:
        query: PubMed search query string.
        max_results: Maximum number of results.
        api_key: Optional NCBI API key for higher rate limits.

    Returns:
        List of PubMed Central IDs (PMCIDs).
    """
    params = {
        "db": "pmc",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key

    logger.info(f"Searching PubMed: {query[:80]}... (max={max_results})")

    response = requests.get(ESEARCH_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    ids = data.get("esearchresult", {}).get("idlist", [])
    logger.info(f"Found {len(ids)} PMC IDs")

    return ids


def download_pdf(
    pmc_id: str,
    output_dir: str | Path = "data/raw",
    api_key: str = "",
) -> str | None:
    """
    Download a PDF from PubMed Central.

    Args:
        pmc_id: PubMed Central ID (numeric string).
        output_dir: Directory to save the PDF.
        api_key: Optional NCBI API key.

    Returns:
        Path to saved PDF, or None if download failed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"PMC{pmc_id}.pdf"

    # Skip if already downloaded
    if output_path.exists():
        logger.debug(f"Already exists: {output_path}")
        return str(output_path)

    try:
        # Get OA service link
        params = {"id": f"PMC{pmc_id}", "format": "pdf"}
        if api_key:
            params["api_key"] = api_key

        response = requests.get(PMC_OA_URL, params=params, timeout=30)

        if response.status_code == 200:
            # Parse XML response for PDF link
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.text)

            link = root.find(".//link[@format='pdf']")
            if link is not None:
                pdf_url = link.get("href")
                if pdf_url:
                    # Download the actual PDF
                    pdf_response = requests.get(pdf_url, timeout=60)
                    pdf_response.raise_for_status()

                    with open(output_path, "wb") as f:
                        f.write(pdf_response.content)

                    logger.info(f"Downloaded: {output_path} ({len(pdf_response.content)} bytes)")
                    return str(output_path)

        # Fallback: direct PMC PDF URL
        fallback_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/"
        pdf_response = requests.get(fallback_url, timeout=60)

        if pdf_response.status_code == 200 and pdf_response.headers.get(
            "content-type", ""
        ).startswith("application/pdf"):
            with open(output_path, "wb") as f:
                f.write(pdf_response.content)
            logger.info(f"Downloaded (fallback): {output_path}")
            return str(output_path)

    except Exception as e:
        logger.warning(f"Failed to download PMC{pmc_id}: {e}")

    return None


def download_batch(
    query: str = DEFAULT_QUERY,
    max_papers: int = 500,
    output_dir: str | Path = "data/raw",
    api_key: str = "",
    delay: float = 0.5,
) -> list[str]:
    """
    Search and download a batch of PubMed papers.

    Args:
        query: Search query.
        max_papers: Target number of papers.
        output_dir: Download directory.
        api_key: Optional NCBI API key.
        delay: Delay between downloads (be nice to NCBI servers).

    Returns:
        List of paths to successfully downloaded PDFs.
    """
    pmc_ids = search_pubmed(query, max_papers, api_key)

    downloaded = []
    failed = 0

    for i, pmc_id in enumerate(pmc_ids):
        logger.info(f"Downloading {i + 1}/{len(pmc_ids)}: PMC{pmc_id}")

        path = download_pdf(pmc_id, output_dir, api_key)
        if path:
            downloaded.append(path)
        else:
            failed += 1

        # Rate limiting
        sleep(delay)

    logger.info(
        f"\nDownload complete: {len(downloaded)} succeeded, {failed} failed "
        f"out of {len(pmc_ids)} total"
    )

    return downloaded


# ── CLI ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Download PubMed Open Access papers")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Search query")
    parser.add_argument("--max", type=int, default=50, help="Max papers to download")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--api-key", default="", help="NCBI API key")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between downloads")
    args = parser.parse_args()

    results = download_batch(
        query=args.query,
        max_papers=args.max,
        output_dir=args.output,
        api_key=args.api_key,
        delay=args.delay,
    )
    print(f"\nDownloaded {len(results)} papers to {args.output}")
