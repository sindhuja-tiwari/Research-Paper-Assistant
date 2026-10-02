"""
Two-stage retrieval, stage 2: re-rank the top-K candidates from vector search
using a cross-encoder, which scores (query, chunk) pairs jointly instead of
comparing independent embeddings. Cross-encoders are far more accurate but
too slow to run over an entire corpus -- hence the two-stage pattern:
fast-but-approximate (FAISS) narrows the field, slow-but-accurate (cross-encoder)
picks the final answer set. We measure the accuracy/latency trade-off in eval.py.
"""
import time
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

import config

_reranker = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(config.RERANKER_MODEL)
    return _reranker


def rerank(query: str, candidates: List[Dict[str, Any]],
           top_k: int = config.TOP_K_RERANK) -> List[Dict[str, Any]]:
    """
    candidates: list of {"chunk": Chunk, "score": float} from vector_store.search()
    Returns the top_k re-ordered by cross-encoder relevance, with timing attached
    so you can report the latency this stage adds.
    """
    if not candidates:
        return []

    model = get_reranker()
    pairs = [(query, c["chunk"].text) for c in candidates]

    t0 = time.time()
    scores = model.predict(pairs)
    elapsed_ms = (time.time() - t0) * 1000

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
    return reranked, elapsed_ms
