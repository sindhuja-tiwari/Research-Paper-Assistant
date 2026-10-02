"""
FAISS-backed vector store with metadata filtering support (filter by year,
section, strategy before/after the similarity search). One index is built
per chunking strategy so you can A/B them independently.
"""
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any

import faiss
import numpy as np

import config
from src.chunking import Chunk


class VectorStore:
    def __init__(self, strategy: str):
        self.strategy = strategy
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Chunk] = []

    @property
    def _index_path(self) -> Path:
        return config.INDEX_DIR / f"{self.strategy}.faiss"

    @property
    def _meta_path(self) -> Path:
        return config.INDEX_DIR / f"{self.strategy}_meta.pkl"

    def build(self, chunks: List[Chunk], embeddings: np.ndarray):
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine sim
        self.index.add(embeddings)
        self.chunks = chunks

    def save(self):
        faiss.write_index(self.index, str(self._index_path))
        with open(self._meta_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self) -> bool:
        if not self._index_path.exists():
            return False
        self.index = faiss.read_index(str(self._index_path))
        with open(self._meta_path, "rb") as f:
            self.chunks = pickle.load(f)
        return True

    def search(self, query_vector: np.ndarray, top_k: int = config.TOP_K_RETRIEVE,
               year_min: Optional[int] = None,
               sections: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Vector similarity search with optional metadata filters applied
        post-hoc on an over-fetched candidate pool. (For a corpus this size,
        over-fetch-then-filter is simpler and fast enough; a production system
        with millions of chunks would push filters into the index itself,
        e.g. via FAISS's IDSelector or a filtered vector DB like Qdrant.)
        """
        if self.index is None:
            raise RuntimeError("Index not built/loaded.")

        fetch_k = min(top_k * 5, self.index.ntotal)  # over-fetch to survive filtering
        scores, idxs = self.index.search(query_vector.reshape(1, -1), fetch_k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            if year_min is not None and chunk.year < year_min:
                continue
            if sections is not None and chunk.section.lower() not in [s.lower() for s in sections]:
                continue
            results.append({"chunk": chunk, "score": float(score)})
            if len(results) >= top_k:
                break
        return results
