"""
Tools the agent can call. Each is a plain Python function plus an OpenAI
function-calling schema. Keeping schema next to implementation avoids drift
between the two.
"""
import requests
from typing import List, Dict, Any

import config
from src.arxiv_fetcher import search_arxiv


# ---------- Tool 1: live arXiv search ----------
def tool_search_arxiv(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Used when the user asks about a paper/topic not yet in the local index."""
    papers = search_arxiv(query, max_results=max_results)
    return [
        {
            "arxiv_id": p.arxiv_id, "title": p.title, "year": p.year(),
            "abstract": p.abstract[:400], "authors": p.authors[:3],
        }
        for p in papers
    ]


# ---------- Tool 2: citation graph (Semantic Scholar) ----------
def tool_citation_graph(arxiv_id: str, direction: str = "citations") -> List[Dict[str, Any]]:
    """
    direction = "citations" -> papers that cite this one
    direction = "references" -> papers this one cites (good "read this first" list)
    """
    paper_id = f"arXiv:{arxiv_id}"
    field = "citations" if direction == "citations" else "references"
    url = f"{config.SEMANTIC_SCHOLAR_API}/paper/{paper_id}"
    params = {"fields": f"{field}.title,{field}.year,{field}.externalIds"}

    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code != 200:
        return [{"error": f"Semantic Scholar lookup failed ({resp.status_code})"}]

    data = resp.json().get(field, [])
    out = []
    for item in data[:10]:
        out.append({
            "title": item.get("title"),
            "year": item.get("year"),
            "arxiv_id": (item.get("externalIds") or {}).get("ArXiv"),
        })
    return out


# ---------- Tool 3: paper summarizer ----------
def tool_summarize_paper(arxiv_id: str) -> Dict[str, Any]:
    """Returns the abstract plus section headings so the agent can give a fast
    TL;DR without doing a full retrieval + generation pass."""
    from src.pdf_parser import ParsedPaper  # local import avoids circular import at module load
    path = config.PROCESSED_DIR / f"{arxiv_id.replace('/', '_')}.json"
    if not path.exists():
        return {"error": "Paper not indexed locally. Use search_arxiv first, then ingest it."}
    paper = ParsedPaper.from_json(path)
    return {
        "title": paper.title,
        "abstract": paper.abstract,
        "sections": [s.heading for s in paper.sections],
    }


# ---------- OpenAI function-calling schemas ----------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": "Search arXiv for papers on a topic. Use this when the user asks "
                            "about a paper or topic that might not be in the local index yet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, e.g. 'retrieval augmented generation'"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "citation_graph",
            "description": "Get papers that cite, or are cited by, a given arXiv paper. "
                            "Use 'references' to find what to read BEFORE a paper, "
                            "'citations' to find newer work that built on it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"},
                    "direction": {"type": "string", "enum": ["citations", "references"]},
                },
                "required": ["arxiv_id", "direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_paper",
            "description": "Get a fast TL;DR (abstract + section list) for a paper already "
                            "in the local index, without running full retrieval.",
            "parameters": {
                "type": "object",
                "properties": {"arxiv_id": {"type": "string"}},
                "required": ["arxiv_id"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "search_arxiv": tool_search_arxiv,
    "citation_graph": tool_citation_graph,
    "summarize_paper": tool_summarize_paper,
}
