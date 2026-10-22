"""Shared fixtures. Every test runs offline: no arXiv, no OpenAI, and no
model downloads (embeddings are replaced with small random vectors)."""
import sys
from pathlib import Path

import fitz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pdf_parser import parse_pdf  # noqa: E402


@pytest.fixture
def sample_pdf(tmp_path):
    """A tiny paper-like PDF: title, three headed sections, and a results table."""
    doc = fitz.open()
    page = doc.new_page()
    lines = [
        ("A Tiny Test Paper", 18),
        ("Abstract", 13),
        ("We study retrieval for question answering over scientific papers.", 10),
        ("1 Introduction", 13),
        ("Language models hallucinate. Grounding answers in retrieved passages helps.", 10),
        ("2 Results", 13),
        ("Model      Accuracy     F1", 10),
        ("Ours       94.1         90.2", 10),
        ("Baseline   88.0         85.3", 10),
        ("Our retriever beats the baseline on every benchmark we tried.", 10),
    ]
    y = 60
    for text, size in lines:
        page.insert_text((50, y), text, fontsize=size)
        y += 26
    path = tmp_path / "paper.pdf"
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def parsed_paper(sample_pdf):
    return parse_pdf(sample_pdf, arxiv_id="2401.00001v1", title="A Tiny Test Paper",
                     authors=["A. Author"], year=2024, abstract="We study retrieval.")
