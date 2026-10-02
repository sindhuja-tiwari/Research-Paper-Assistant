"""
Embeds chunks using a local sentence-transformers model, with a disk cache
keyed by content hash. The caching is the real engineering point here:
re-indexing a corpus after a small change (e.g. one new paper, or a fixed
parsing bug) should NOT recompute embeddings for chunks whose text hasn't
changed. We measure this explicitly in eval.py.
"""
import hashlib
import pickle
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

import config

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Simple disk-backed cache: content_hash -> embedding vector."""

    def __init__(self, cache_path: Path = config.CACHE_DIR / "embedding_cache.pkl"):
        self.cache_path = cache_path
        self._cache: Dict[str, np.ndarray] = {}
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                self._cache = pickle.load(f)

    def save(self):
        with open(self.cache_path, "wb") as f:
            pickle.dump(self._cache, f)

    def get(self, text: str):
        return self._cache.get(_hash_text(text))

    def set(self, text: str, vector: np.ndarray):
        self._cache[_hash_text(text)] = vector


def embed_texts(texts: List[str], cache: EmbeddingCache = None,
                 verbose: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Embed a list of texts, using the cache wherever possible.
    Returns (embeddings_array, stats_dict) where stats shows how much the
    cache saved you -- this is your "before/after caching" evidence for the README.
    """
    own_cache = cache is None
    if own_cache:
        cache = EmbeddingCache()

    model = get_model()
    vectors = [None] * len(texts)
    to_compute_idx = []
    to_compute_texts = []

    for i, t in enumerate(texts):
        cached = cache.get(t)
        if cached is not None:
            vectors[i] = cached
        else:
            to_compute_idx.append(i)
            to_compute_texts.append(t)

    t0 = time.time()
    if to_compute_texts:
        new_vectors = model.encode(to_compute_texts, show_progress_bar=verbose,
                                    convert_to_numpy=True, normalize_embeddings=True)
        for idx, vec in zip(to_compute_idx, new_vectors):
            vectors[idx] = vec
            cache.set(texts[idx], vec)
    elapsed = time.time() - t0

    if own_cache:
        cache.save()

    stats = {
        "total_chunks": len(texts),
        "cache_hits": len(texts) - len(to_compute_texts),
        "newly_computed": len(to_compute_texts),
        "compute_time_sec": round(elapsed, 3),
    }
    if verbose:
        print(f"  embeddings: {stats['cache_hits']} cached, "
              f"{stats['newly_computed']} newly computed in {stats['compute_time_sec']}s")

    return np.vstack(vectors).astype("float32"), stats
