"""
Fetches paper metadata from the arXiv API and downloads PDFs.
This is both the offline corpus-builder AND the live "search arXiv" tool
the agent calls when a paper isn't in the index yet.
"""
import re
import time
import requests
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import config

ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published: str          # e.g. "2023-06-12"
    pdf_url: str

    def year(self) -> int:
        return int(self.published[:4])


def search_arxiv(query: str, max_results: int = 10) -> List[ArxivPaper]:
    """
    Search arXiv for papers matching `query`. Used both to build the initial
    corpus and as a live agent tool ("find papers on X that aren't indexed yet").
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    resp = requests.get(config.ARXIV_API_URL, params=params, timeout=15)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)

    papers = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        raw_id = entry.find("atom:id", ARXIV_NS).text.strip()
        arxiv_id = raw_id.split("/abs/")[-1]
        title = entry.find("atom:title", ARXIV_NS).text.strip().replace("\n", " ")
        abstract = entry.find("atom:summary", ARXIV_NS).text.strip().replace("\n", " ")
        published = entry.find("atom:published", ARXIV_NS).text[:10]
        authors = [
            a.find("atom:name", ARXIV_NS).text
            for a in entry.findall("atom:author", ARXIV_NS)
        ]
        pdf_url = None
        for link in entry.findall("atom:link", ARXIV_NS):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href")
        if pdf_url is None:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        papers.append(ArxivPaper(
            arxiv_id=arxiv_id, title=title, authors=authors,
            abstract=abstract, published=published, pdf_url=pdf_url,
        ))
    return papers


def download_pdf(paper: ArxivPaper, dest_dir: Path = config.RAW_PDF_DIR) -> Path:
    """Download a paper's PDF if not already present locally."""
    safe_id = re.sub(r"[^\w\.\-]", "_", paper.arxiv_id)
    dest_path = dest_dir / f"{safe_id}.pdf"
    if dest_path.exists():
        return dest_path

    resp = requests.get(paper.pdf_url, timeout=30)
    resp.raise_for_status()
    dest_path.write_bytes(resp.content)
    time.sleep(1)  # be polite to arXiv's servers
    return dest_path


def build_corpus(query: str, max_results: int = 50) -> List[ArxivPaper]:
    """One-shot helper: search + download a batch of papers for the corpus."""
    papers = search_arxiv(query, max_results=max_results)
    for p in papers:
        try:
            download_pdf(p)
        except requests.RequestException as e:
            print(f"  ! failed to download {p.arxiv_id}: {e}")
    return papers


if __name__ == "__main__":
    # Quick manual test: python -m src.arxiv_fetcher
    results = search_arxiv("retrieval augmented generation", max_results=5)
    for r in results:
        print(f"{r.arxiv_id}  ({r.year()})  {r.title}")
