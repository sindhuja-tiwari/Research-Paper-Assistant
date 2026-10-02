"""
Orchestrates the full retrieval pipeline: embed query -> vector search ->
(optional) re-rank. Tracks latency per stage, which feeds directly into the
latency/cost report that's one of this project's key differentiators.
"""
import time
from typing import List, Dict, Any, Optional

import numpy as np

import config
from src.embeddings import get_model, EmbeddingCache
from src.vector_store import VectorStore
from src.rerank import rerank as rerank_fn


class RetrievalPipeline:
    def __init__(self, strategy: str, use_reranker: bool = True):
        self.strategy = strategy
        self.use_reranker = use_reranker
        self.store = VectorStore(strategy)
        if not self.store.load():
            raise RuntimeError(
                f"No index found for strategy '{strategy}'. Run scripts/ingest.py first."
            )
        self.cache = EmbeddingCache()

    def retrieve(self, query: str, year_min: Optional[int] = None,
                 sections: Optional[List[str]] = None) -> Dict[str, Any]:
        timings = {}

        t0 = time.time()
        q_vec = get_model().encode([query], normalize_embeddings=True)[0].astype("float32")
        timings["embed_query_ms"] = (time.time() - t0) * 1000

        t0 = time.time()
        candidates = self.store.search(
            q_vec, top_k=config.TOP_K_RETRIEVE, year_min=year_min, sections=sections
        )
        timings["vector_search_ms"] = (time.time() - t0) * 1000

        if self.use_reranker and candidates:
            final, rerank_ms = rerank_fn(query, candidates, top_k=config.TOP_K_RERANK)
            timings["rerank_ms"] = rerank_ms
        else:
            final = candidates[: config.TOP_K_RERANK]
            timings["rerank_ms"] = 0.0

        timings["total_retrieval_ms"] = sum(timings.values())

        return {"query": query, "results": final, "timings": timings}
