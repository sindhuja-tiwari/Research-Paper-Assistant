import random
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


class ArxivRateLimitError(requests.HTTPError):
    """arXiv kept answering 429 after all retries."""


class ArxivUnavailableError(requests.ConnectionError):
    """arXiv kept timing out, refusing connections, or returning 5xx after all retries."""


# arXiv asks clients to wait at least 3s between requests and to send a
# descriptive User-Agent: https://info.arxiv.org/help/api/tou.html
MIN_REQUEST_INTERVAL = 3.0
MAX_ATTEMPTS = 5
MAX_BACKOFF = 300
# (connect, read) timeouts in seconds. arXiv search can be slow under load.
SEARCH_TIMEOUT = (10, 90)
PDF_TIMEOUT = (10, 120)

_last_request_time = 0.0


def _user_agent() -> str:
    ua = "RAG-Paper-Assistant/1.0 (research paper discovery"
    if config.ARXIV_CONTACT_EMAIL:
        ua += f"; mailto:{config.ARXIV_CONTACT_EMAIL}"
    return ua + ")"


def _throttle() -> None:
    """Keep at least MIN_REQUEST_INTERVAL seconds between arXiv requests."""
    global _last_request_time
    wait = MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request_time)
    if wait > 0:
        time.sleep(wait)
    _last_request_time = time.monotonic()


def _retry_after(resp: requests.Response, default: float) -> float:
    value = resp.headers.get("Retry-After")
    try:
        delay = float(value) if value else default
    except ValueError:
        delay = default
    return min(max(delay, MIN_REQUEST_INTERVAL), MAX_BACKOFF)


def _arxiv_get(url: str, timeout, **kwargs) -> requests.Response:
    """GET an arXiv URL politely: throttled, retried, and with clear errors."""
    headers = {"User-Agent": _user_agent()}

    for attempt in range(MAX_ATTEMPTS):
        last_attempt = attempt == MAX_ATTEMPTS - 1
        backoff = min(30 * 2 ** attempt, MAX_BACKOFF) + random.uniform(0, 5)
        _throttle()
        try:
            resp = requests.get(url, headers=headers, timeout=timeout, **kwargs)
        except (requests.Timeout, requests.ConnectionError) as exc:
            reason = "timed out" if isinstance(exc, requests.Timeout) else "could not connect"
            if last_attempt:
                raise ArxivUnavailableError(
                    f"arXiv {reason} on all {MAX_ATTEMPTS} attempts. arXiv may be "
                    f"overloaded or blocking this machine. Wait 15-30 minutes, check "
                    f"https://status.arxiv.org, then try again."
                ) from exc
            delay = min(15 * 2 ** attempt, 120) + random.uniform(0, 5)
            print(f"arXiv request {reason}, attempt {attempt + 1}/{MAX_ATTEMPTS}. "
                  f"Waiting {delay:.0f}s before retry...")
            time.sleep(delay)
            continue

        if resp.status_code == 429:
            if last_attempt:
                raise ArxivRateLimitError(
                    f"arXiv is rate limiting this machine (HTTP 429) after "
                    f"{MAX_ATTEMPTS} attempts. Wait 10-15 minutes without "
                    f"sending any arXiv requests, then try again.",
                    response=resp,
                )
            delay = _retry_after(resp, backoff)
            print(f"arXiv rate limit (429), attempt {attempt + 1}/{MAX_ATTEMPTS}. "
                  f"Waiting {delay:.0f}s before retry...")
            time.sleep(delay)
            continue

        if resp.status_code in (500, 502, 503, 504):
            if last_attempt:
                raise ArxivUnavailableError(
                    f"arXiv returned HTTP {resp.status_code} on all {MAX_ATTEMPTS} "
                    f"attempts. Check https://status.arxiv.org and try again later.",
                    response=resp,
                )
            delay = _retry_after(resp, min(15 * 2 ** attempt, 120))
            print(f"arXiv server error ({resp.status_code}); waiting {delay:.0f}s...")
            time.sleep(delay)
            continue

        resp.raise_for_status()
        return resp

    raise RuntimeError("arXiv request failed after retries")


def safe_filename(arxiv_id: str) -> str:
    """File-safe form of an arXiv id, e.g. 'hep-th/9901001v1' -> 'hep-th_9901001v1'."""
    return re.sub(r"[^\w.-]", "_", arxiv_id)


def search_arxiv(query: str, max_results: int = 10) -> List[ArxivPaper]:
    """Search arXiv for papers matching a query."""
    params = {
        "search_query": f'all:"{query}"',
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    resp = _arxiv_get(config.ARXIV_API_URL, timeout=SEARCH_TIMEOUT, params=params)

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
    dest_path = dest_dir / f"{safe_filename(paper.arxiv_id)}.pdf"

    if dest_path.exists():
        return dest_path

    response = _arxiv_get(paper.pdf_url, timeout=PDF_TIMEOUT)
    dest_path.write_bytes(response.content)
    return dest_path


def build_corpus(query: str, max_results: int = 50) -> List[ArxivPaper]:
    """Search for papers and download their PDFs."""
    papers = search_arxiv(query, max_results=max_results)

    for paper in papers:
        try:
            download_pdf(paper)
        except ArxivRateLimitError:
            raise
        except requests.RequestException as exc:
            print(f"  ! failed to download {paper.arxiv_id}: {exc}")

    return papers


if __name__ == "__main__":
    for paper in search_arxiv("retrieval augmented generation", max_results=5):
        print(f"{paper.arxiv_id} ({paper.year()}) {paper.title}")
