"""
Parses a paper PDF into structured sections + tables using PyMuPDF.

This is a heuristic parser, not a full layout model like GROBID. The README
explains that trade-off explicitly: GROBID would give cleaner section/table
boundaries but requires running a separate Java service. This heuristic
approach (font-size-based heading detection + text-grid table detection)
gets ~80% of the value with zero extra infrastructure, which is the right
call for a single-person project. Swapping in GROBID later is a one-file change.
"""
import re
import json
import fitz  # PyMuPDF
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict

import config

# Common academic paper section headers, used to anchor heading detection
# even when font-size heuristics are ambiguous (varies a lot by template).
KNOWN_HEADINGS = [
    "abstract", "introduction", "related work", "background",
    "method", "methods", "methodology", "approach",
    "experiments", "experimental setup", "results", "evaluation",
    "discussion", "limitations", "conclusion", "conclusions",
    "references", "acknowledgments", "appendix",
]


@dataclass
class Table:
    section: str
    caption: str
    raw_text: str          # the grid-like text block as extracted


@dataclass
class Section:
    heading: str
    text: str
    order: int


@dataclass
class ParsedPaper:
    arxiv_id: str
    title: str
    authors: List[str]
    year: int
    abstract: str
    sections: List[Section] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    full_text: str = ""     # fallback for naive chunking baseline

    def to_json(self, path: Path):
        d = asdict(self)
        path.write_text(json.dumps(d, indent=2))

    @staticmethod
    def from_json(path: Path) -> "ParsedPaper":
        d = json.loads(path.read_text())
        d["sections"] = [Section(**s) for s in d["sections"]]
        d["tables"] = [Table(**t) for t in d["tables"]]
        return ParsedPaper(**d)


def _is_heading_line(line: str, font_size: float, body_font_size: float) -> bool:
    """A line is probably a heading if it's short, bigger than body text,
    and either matches a known heading word or looks like '3. Method' / 'III. RESULTS'."""
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return False
    lower = stripped.lower().strip(" .0123456789ivx")
    if any(h in lower for h in KNOWN_HEADINGS):
        return True
    if font_size > body_font_size + 1.0 and len(stripped.split()) <= 6:
        return True
    if re.match(r"^\d+(\.\d+)?\.?\s+[A-Z]", stripped) and len(stripped.split()) <= 8:
        return True
    return False


def _looks_like_table_row(line: str) -> bool:
    """Crude signal: multiple runs of whitespace (columns) or many numeric tokens."""
    if re.search(r"\s{3,}", line):
        return True
    tokens = line.split()
    if len(tokens) >= 3:
        numeric = sum(1 for t in tokens if re.match(r"^-?\d+(\.\d+)?%?$", t))
        if numeric / len(tokens) > 0.4:
            return True
    return False


def parse_pdf(pdf_path: Path, arxiv_id: str, title: str, authors: List[str],
              year: int, abstract: str) -> ParsedPaper:
    doc = fitz.open(pdf_path)

    all_spans = []  # (text, font_size) across the whole doc, in reading order
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                line_text = "".join(s["text"] for s in line["spans"]).strip()
                if not line_text:
                    continue
                avg_size = sum(s["size"] for s in line["spans"]) / len(line["spans"])
                all_spans.append((line_text, avg_size))

    if not all_spans:
        return ParsedPaper(arxiv_id, title, authors, year, abstract, full_text="")

    body_font_size = sorted(s[1] for s in all_spans)[len(all_spans) // 2]  # median ~= body text

    sections: List[Section] = []
    tables: List[Table] = []
    current_heading = "Preamble"
    current_lines: List[str] = []
    current_table_lines: List[str] = []
    order = 0

    def flush_section():
        nonlocal current_lines, order
        text = " ".join(current_lines).strip()
        if text:
            sections.append(Section(heading=current_heading, text=text, order=order))
            order += 1
        current_lines = []

    def flush_table():
        nonlocal current_table_lines
        if len(current_table_lines) >= 2:  # require at least 2 rows to call it a table
            tables.append(Table(
                section=current_heading,
                caption=f"Table in section: {current_heading}",
                raw_text="\n".join(current_table_lines),
            ))
        current_table_lines = []

    in_table = False
    for line_text, font_size in all_spans:
        if _is_heading_line(line_text, font_size, body_font_size):
            flush_table()
            flush_section()
            current_heading = line_text.strip()
            in_table = False
            continue

        if _looks_like_table_row(line_text):
            in_table = True
            current_table_lines.append(line_text)
            continue
        elif in_table:
            flush_table()
            in_table = False

        current_lines.append(line_text)

    flush_table()
    flush_section()

    full_text = " ".join(s.text for s in sections)
    doc.close()

    return ParsedPaper(
        arxiv_id=arxiv_id, title=title, authors=authors, year=year,
        abstract=abstract, sections=sections, tables=tables, full_text=full_text,
    )


def parse_and_cache(pdf_path: Path, arxiv_id: str, title: str, authors: List[str],
                     year: int, abstract: str, force: bool = False) -> ParsedPaper:
    """Parse a PDF and cache the structured result as JSON so re-runs are instant."""
    out_path = config.PROCESSED_DIR / f"{arxiv_id.replace('/', '_')}.json"
    if out_path.exists() and not force:
        return ParsedPaper.from_json(out_path)

    parsed = parse_pdf(pdf_path, arxiv_id, title, authors, year, abstract)
    parsed.to_json(out_path)
    return parsed
