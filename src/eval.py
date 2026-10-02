"""
Evaluates retrieval quality (recall@k) across chunking strategies and with/
without re-ranking. This produces the comparison table that should go
straight into your README -- it's the single most convincing artifact in
this whole project.

Eval set format (eval/questions.json):
[
  {
    "question": "What dataset was used to evaluate the model?",
    "arxiv_id": "2005.11401",
    "relevant_sections": ["Experiments", "Experimental Setup"]
  },
  ...
]
A retrieved chunk counts as "relevant" if it's from the right paper AND
(when relevant_sections is given) from one of the right sections. This is a
simple but defensible proxy -- exact chunk-level ground truth would require
hand-labeling every chunk, which doesn't scale for a solo project.
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import config
from src.retrieval import RetrievalPipeline


def load_eval_set(path: Path = config.EVAL_DIR / "questions.json") -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"No eval set at {path}. Create one -- see the docstring in src/eval.py "
            "for the expected format, or run scripts/build_eval_template.py."
        )
    return json.loads(path.read_text())


def _is_relevant(chunk, question: Dict[str, Any]) -> bool:
    if chunk.arxiv_id != question["arxiv_id"]:
        return False
    rel_sections = question.get("relevant_sections")
    if not rel_sections:
        return True  # paper-level match is enough if no section specified
    return any(rs.lower() in chunk.section.lower() for rs in rel_sections)


def recall_at_k(pipeline: RetrievalPipeline, questions: List[Dict[str, Any]],
                 k: int) -> float:
    hits = 0
    for q in questions:
        retrieved = pipeline.retrieve(q["question"])["results"][:k]
        if any(_is_relevant(r["chunk"], q) for r in retrieved):
            hits += 1
    return round(hits / len(questions), 3)


def run_full_comparison(strategies: List[str] = ("naive", "section", "section_table"),
                         k_values: List[int] = (1, 3, 5),
                         questions_path: Path = None) -> Dict[str, Any]:
    """
    Runs recall@k for every strategy, both with and without re-ranking.
    Returns a results dict ready to render as a markdown table.
    """
    questions = load_eval_set(questions_path) if questions_path else load_eval_set()
    results = {}

    for strategy in strategies:
        for use_reranker in (False, True):
            label = f"{strategy} {'+ rerank' if use_reranker else '(no rerank)'}"
            try:
                pipeline = RetrievalPipeline(strategy=strategy, use_reranker=use_reranker)
            except RuntimeError as e:
                print(f"  skipping {label}: {e}")
                continue

            row = {}
            t0 = time.time()
            for k in k_values:
                row[f"recall@{k}"] = recall_at_k(pipeline, questions, k)
            row["avg_query_time_ms"] = round(
                (time.time() - t0) / len(questions) / len(k_values) * 1000, 1
            )
            results[label] = row
            print(f"  {label}: {row}")

    return results


def print_markdown_table(results: Dict[str, Any]):
    if not results:
        print("No results to show.")
        return
    cols = list(next(iter(results.values())).keys())
    print("| Strategy | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 1))
    for label, row in results.items():
        print(f"| {label} | " + " | ".join(str(row[c]) for c in cols) + " |")


if __name__ == "__main__":
    results = run_full_comparison()
    print_markdown_table(results)
