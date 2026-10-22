"""
End-to-end ingestion: search arXiv -> download PDFs -> parse -> chunk (all
3 strategies) -> embed (with caching) -> build + save FAISS indexes.

Usage:
    python -m scripts.ingest --query "retrieval augmented generation" --max_results 40
    python -m scripts.ingest --ids_file eval/corpus_ids.txt   # fixed corpus used by eval/questions.json
"""
import argparse
import sys
import time

import config
from src.arxiv_fetcher import (
    ArxivRateLimitError, ArxivUnavailableError, build_corpus, safe_filename,
)
from src.pdf_parser import parse_and_cache
from src.chunking import chunk_paper, STRATEGIES
from src.embeddings import embed_texts, EmbeddingCache
from src.vector_store import VectorStore


def main(query: str = None, max_results: int = 40, arxiv_ids=None):
    if arxiv_ids:
        print(f"Fetching {len(arxiv_ids)} papers from arXiv by id...")
    else:
        print(f"Searching arXiv for '{query}' (max {max_results} papers)...")
    papers = build_corpus(query, max_results=max_results, arxiv_ids=arxiv_ids)
    print(f"Downloaded {len(papers)} papers.\n")

    print("Parsing PDFs into sections/tables...")
    parsed_papers = []
    for p in papers:
        pdf_path = config.RAW_PDF_DIR / f"{safe_filename(p.arxiv_id)}.pdf"
        if not pdf_path.exists():
            continue
        try:
            parsed = parse_and_cache(
                pdf_path, arxiv_id=p.arxiv_id, title=p.title,
                authors=p.authors, year=p.year(), abstract=p.abstract,
            )
            parsed_papers.append(parsed)
        except Exception as e:
            print(f"  ! failed to parse {p.arxiv_id}: {e}")
    print(f"Parsed {len(parsed_papers)} papers.\n")
    if not parsed_papers:
        print("No papers could be parsed, so there is nothing to index. "
              "Try a broader --query.", file=sys.stderr)
        sys.exit(1)

    cache = EmbeddingCache()
    for strategy in STRATEGIES:
        print(f"--- Building index for strategy: {strategy} ---")
        all_chunks = []
        for paper in parsed_papers:
            all_chunks.extend(chunk_paper(paper, strategy))
        print(f"  {len(all_chunks)} chunks")

        texts = [c.text for c in all_chunks]
        t0 = time.time()
        vectors, stats = embed_texts(texts, cache=cache)
        print(f"  embedding stats: {stats}  (wall time {time.time()-t0:.1f}s)")

        store = VectorStore(strategy)
        store.build(all_chunks, vectors)
        store.save()
        print(f"  saved index to {store._index_path}\n")

    cache.save()
    print("Done. Indexes ready for all strategies:", list(STRATEGIES.keys()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query", type=str,
                        help="arXiv search query defining your corpus, e.g. 'efficient transformers'")
    source.add_argument("--ids_file", type=str,
                        help="text file with one arXiv id per line (# comments allowed)")
    parser.add_argument("--max_results", type=int, default=40)
    args = parser.parse_args()

    ids = None
    if args.ids_file:
        with open(args.ids_file) as f:
            ids = [line.split("#")[0].strip() for line in f]
        ids = [i for i in ids if i]
    try:
        main(args.query, args.max_results, arxiv_ids=ids)
    except (ArxivRateLimitError, ArxivUnavailableError) as e:
        print(f"\nError: {e}\nPDFs already downloaded are kept in {config.RAW_PDF_DIR}, "
              f"so the next run only fetches what is missing.", file=sys.stderr)
        sys.exit(1)
