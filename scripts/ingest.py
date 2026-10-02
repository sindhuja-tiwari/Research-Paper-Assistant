"""
End-to-end ingestion: search arXiv -> download PDFs -> parse -> chunk (all
3 strategies) -> embed (with caching) -> build + save FAISS indexes.

Usage:
    python -m scripts.ingest --query "retrieval augmented generation" --max_results 40
"""
import argparse
import time

import config
from src.arxiv_fetcher import build_corpus
from src.pdf_parser import parse_and_cache
from src.chunking import chunk_paper, STRATEGIES
from src.embeddings import embed_texts, EmbeddingCache
from src.vector_store import VectorStore


def main(query: str, max_results: int):
    print(f"Searching arXiv for '{query}' (max {max_results} papers)...")
    papers = build_corpus(query, max_results=max_results)
    print(f"Downloaded {len(papers)} papers.\n")

    print("Parsing PDFs into sections/tables...")
    parsed_papers = []
    for p in papers:
        pdf_path = config.RAW_PDF_DIR / f"{p.arxiv_id.replace('/', '_')}.pdf"
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
    parser.add_argument("--query", type=str, required=True,
                         help="arXiv search query defining your corpus, e.g. 'efficient transformers'")
    parser.add_argument("--max_results", type=int, default=40)
    args = parser.parse_args()
    main(args.query, args.max_results)
