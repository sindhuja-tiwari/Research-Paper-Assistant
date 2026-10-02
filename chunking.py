"""
Three chunking strategies, so you can empirically compare them (see eval.py).
This comparison table is the single best artifact for your README/interview:
it's concrete proof you understand *why* chunking strategy matters, not just
that you picked one.
"""
from dataclasses import dataclass
from typing import List

import config
from src.pdf_parser import ParsedPaper


@dataclass
class Chunk:
    chunk_id: str
    text: str
    arxiv_id: str
    title: str
    year: int
    section: str
    strategy: str          # "naive" | "section" | "section_table"
    is_table: bool = False


def naive_chunks(paper: ParsedPaper, strategy_name: str = "naive") -> List[Chunk]:
    """Baseline: fixed-size token windows with overlap, ignoring document structure.
    Included deliberately as the weak baseline in the eval comparison."""
    words = paper.full_text.split()
    size, overlap = config.NAIVE_CHUNK_SIZE, config.NAIVE_CHUNK_OVERLAP
    chunks = []
    i = 0
    idx = 0
    while i < len(words):
        window = words[i : i + size]
        if not window:
            break
        text = " ".join(window)
        chunks.append(Chunk(
            chunk_id=f"{paper.arxiv_id}_naive_{idx}",
            text=text, arxiv_id=paper.arxiv_id, title=paper.title,
            year=paper.year, section="unknown", strategy=strategy_name,
        ))
        idx += 1
        i += size - overlap
    return chunks


def section_aware_chunks(paper: ParsedPaper) -> List[Chunk]:
    """Chunk by logical section (Abstract, Method, Results, ...). Long sections
    get sub-split on sentence boundaries so no chunk blows past the size limit,
    but we never cross a section boundary mid-chunk -- that's the whole point."""
    chunks = []
    idx = 0
    for sec in paper.sections:
        words = sec.text.split()
        if len(words) <= config.MAX_SECTION_CHUNK_SIZE:
            chunks.append(Chunk(
                chunk_id=f"{paper.arxiv_id}_sec_{idx}",
                text=f"[{sec.heading}] {sec.text}",
                arxiv_id=paper.arxiv_id, title=paper.title, year=paper.year,
                section=sec.heading, strategy="section",
            ))
            idx += 1
        else:
            # sub-split long sections, still tagged with the parent heading
            step = config.MAX_SECTION_CHUNK_SIZE
            for i in range(0, len(words), step):
                sub_text = " ".join(words[i : i + step])
                chunks.append(Chunk(
                    chunk_id=f"{paper.arxiv_id}_sec_{idx}",
                    text=f"[{sec.heading}] {sub_text}",
                    arxiv_id=paper.arxiv_id, title=paper.title, year=paper.year,
                    section=sec.heading, strategy="section",
                ))
                idx += 1
    return chunks


def section_and_table_aware_chunks(paper: ParsedPaper) -> List[Chunk]:
    """Section-aware chunking PLUS tables kept as their own dedicated chunks
    (never flattened into surrounding prose). This is what makes numeric
    questions like "what accuracy did Table 2 report?" actually answerable --
    flattening a table into a paragraph destroys its structure."""
    chunks = section_aware_chunks(paper)
    # Re-tag strategy and add table chunks
    for c in chunks:
        c.strategy = "section_table"
        c.chunk_id = c.chunk_id.replace("_sec_", "_sectbl_")

    idx = len(chunks)
    for table in paper.tables:
        chunks.append(Chunk(
            chunk_id=f"{paper.arxiv_id}_table_{idx}",
            text=f"[TABLE — {table.caption}]\n{table.raw_text}",
            arxiv_id=paper.arxiv_id, title=paper.title, year=paper.year,
            section=table.section, strategy="section_table", is_table=True,
        ))
        idx += 1
    return chunks


STRATEGIES = {
    "naive": naive_chunks,
    "section": section_aware_chunks,
    "section_table": section_and_table_aware_chunks,
}


def chunk_paper(paper: ParsedPaper, strategy: str) -> List[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(STRATEGIES)}")
    return STRATEGIES[strategy](paper)
