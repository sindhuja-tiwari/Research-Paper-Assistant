import numpy as np

from src import eval as ev
from src.chunking import Chunk
from src.vector_store import VectorStore


def _chunk(i, arxiv_id="2401.00001v1", year=2024, section="Results"):
    return Chunk(chunk_id=f"c{i}", text=f"text {i}", arxiv_id=arxiv_id, title="T",
                 year=year, section=section, strategy="section")


def _unit(v):
    v = np.asarray(v, dtype="float32")
    return v / np.linalg.norm(v)


def test_vector_store_ranks_by_cosine_and_filters_by_year(tmp_path, monkeypatch):
    monkeypatch.setattr("config.INDEX_DIR", tmp_path)
    chunks = [_chunk(0, year=2019), _chunk(1, year=2023), _chunk(2, year=2024)]
    vecs = np.vstack([_unit([1, 0, 0]), _unit([0.9, 0.1, 0]), _unit([0, 1, 0])])

    store = VectorStore("test")
    store.build(chunks, vecs)
    store.save()

    loaded = VectorStore("test")
    assert loaded.load()
    hits = loaded.search(_unit([1, 0, 0]), top_k=3)
    assert [h["chunk"].chunk_id for h in hits] == ["c0", "c1", "c2"]

    recent = loaded.search(_unit([1, 0, 0]), top_k=3, year_min=2023)
    assert [h["chunk"].chunk_id for h in recent] == ["c1", "c2"]


def test_relevance_ignores_arxiv_version_suffix():
    q = {"arxiv_id": "2401.00001", "relevant_sections": []}
    assert ev._is_relevant(_chunk(0, arxiv_id="2401.00001v3"), q)
    assert not ev._is_relevant(_chunk(0, arxiv_id="2401.99999v1"), q)


def test_relevance_checks_sections_when_given():
    q = {"arxiv_id": "2401.00001", "relevant_sections": ["Experiments"]}
    assert ev._is_relevant(_chunk(0, section="4 Experiments"), q)
    assert not ev._is_relevant(_chunk(0, section="Introduction"), q)


class _FakePipeline:
    """Returns a fixed ranking: the right paper only at position 2."""
    def retrieve(self, query):
        ranked = [_chunk(0, arxiv_id="9999.99999"), _chunk(1), _chunk(2, arxiv_id="9999.99999")]
        return {"results": [{"chunk": c} for c in ranked]}


def test_recall_at_k():
    questions = [{"question": "q", "arxiv_id": "2401.00001", "relevant_sections": []}]
    assert ev.recall_at_k(_FakePipeline(), questions, k=1) == 0.0
    assert ev.recall_at_k(_FakePipeline(), questions, k=2) == 1.0


def test_load_eval_set_skips_todo_placeholders(tmp_path):
    path = tmp_path / "q.json"
    path.write_text('[{"question": "TODO: x", "arxiv_id": "1"},'
                    ' {"question": "Real?", "arxiv_id": "2"}]')
    assert [q["arxiv_id"] for q in ev.load_eval_set(path)] == ["2"]


def test_shipped_eval_set_matches_corpus():
    import config
    ids = {line.split("#")[0].strip()
           for line in (config.EVAL_DIR / "corpus_ids.txt").read_text().splitlines()}
    ids.discard("")
    questions = ev.load_eval_set()
    assert questions
    assert {q["arxiv_id"] for q in questions} <= ids
