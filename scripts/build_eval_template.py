"""
Scaffolds eval/questions.json with one placeholder question per indexed
paper, so you just have to fill in a real question + relevant_sections
for each rather than writing the file by hand from scratch.

You still need to write real questions yourself -- the eval is only as
good as the judgment you put into labeling it, and that labeling work is
itself a legitimate thing to describe in an interview.

Usage:
    python -m scripts.build_eval_template
"""
import json

import config
from src.pdf_parser import ParsedPaper


def main():
    template = []
    for path in sorted(config.PROCESSED_DIR.glob("*.json")):
        paper = ParsedPaper.from_json(path)
        template.append({
            "question": f"TODO: write a real question about '{paper.title}'",
            "arxiv_id": paper.arxiv_id,
            "relevant_sections": [],  # e.g. ["Experiments", "Results"] -- leave empty for paper-level match
        })

    out_path = config.EVAL_DIR / "questions_template.json"
    out_path.write_text(json.dumps(template, indent=2))
    print(f"Wrote {len(template)} placeholder questions to {out_path}")
    print("Edit these, rename to questions.json, then run scripts/run_eval.py")


if __name__ == "__main__":
    main()
