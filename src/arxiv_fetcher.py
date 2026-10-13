import re
import time
import requests
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List

import config

ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published: str
    pdf_url: str

    def year(self) -> int:
        return int(self.published[:4])


def search_arxiv(query: str, max_results: int = 10) -> List[ArxivPaper]:
    """Search arXiv for papers matching a query."""
    params = {
        "search_query": f'all:"{query}"',
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    headers = {"User-Agent": "RAG-Paper-Assistant/1.0 (research paper discovery)"}

    for attempt in range(3):
        try:
            resp = requests.get(
                config.ARXIV_API_URL,
                params=params,
                headers=headers,
                timeout=30,
            )

            if resp.status_code == 429:
                if attempt == 2:
                    resp.raise_for_status()
                retry_after = resp.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else 60 * (attempt + 1)
                except ValueError:
                    delay = 60 * (attempt + 1)
                delay = min(max(delay, 1), 180)
                print(f"arXiv rate limit (429). Waiting {delay:g}s before retry...")
                time.sleep(delay)
                continue

            if resp.status_code in (500, 502, 503, 504):
                if attempt == 2:
                    resp.raise_for_status()
                delay = 5 * (attempt + 1)
                print(f"arXiv server error ({resp.status_code}); waiting {delay}s...")
                time.sleep(delay)
                continue

            resp.raise_for_status()
            break

        except requests.RequestException:
            if attempt == 2:
                raise
            delay = 5 * (attempt + 1)
            print(f"arXiv request failed; waiting {delay}s before retry...")
            time.sleep(delay)
    else:
        raise RuntimeError("arXiv request failed after retries")

    root = ET.fromstring(resp.text)
    papers = []

    for entry in root.findall("atom:entry", ARXIV_NS):
        raw_id = entry.find("atom:id", ARXIV_NS).text.strip()
        arxiv_id = raw_id.split("/abs/")[-1]
        title = " ".join(entry.find("atom:title", ARXIV_NS).text.split())
        abstract = " ".join(entry.find("atom:summary", ARXIV_NS).text.split())
        published = entry.find("atom:published", ARXIV_NS).text[:10]
        authors = [
            a.find("atom:name", ARXIV_NS).text
            for a in entry.findall("atom:author", ARXIV_NS)
        ]

        pdf_url = None
        for link in entry.findall("atom:link", ARXIV_NS):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href")
                break

        if pdf_url is None:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        papers.append(
            ArxivPaper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                published=published,
                pdf_url=pdf_url,
            )
        )

    return papers


def download_pdf(paper: ArxivPaper, dest_dir: Path = None) -> Path:
    """Download a PDF if it isn't already stored locally."""
    if dest_dir is None:
        dest_dir = config.RAW_PDF_DIR

    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^\w.-]", "_", paper.arxiv_id)
    dest_path = dest_dir / f"{safe_id}.pdf"

    if dest_path.exists():
        return dest_path

    response = requests.get(
        paper.pdf_url,
        headers={"User-Agent": "RAG-Paper-Assistant/1.0"},
        timeout=60,
    )
    response.raise_for_status()
    dest_path.write_bytes(response.content)
    time.sleep(1)
    return dest_path


def build_corpus(query: str, max_results: int = 50) -> List[ArxivPaper]:
    """Search for papers and download their PDFs."""
    papers = search_arxiv(query, max_results=max_results)

    for paper in papers:
        try:
            download_pdf(paper)
        except requests.RequestException as exc:
            print(f"  ! failed to download {paper.arxiv_id}: {exc}")

    return papers


if __name__ == "__main__":
    for paper in search_arxiv("retrieval augmented generation", max_results=5):
        print(f"{paper.arxiv_id} ({paper.year()}) {paper.title}")
