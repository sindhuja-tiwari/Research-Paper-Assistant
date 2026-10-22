import json
from types import SimpleNamespace

from src import agent as agent_mod
from src.chunking import Chunk
from src.generation import _check_groundedness


def _chunk(text):
    return Chunk(chunk_id="c", text=text, arxiv_id="2401.00001", title="T", year=2024,
                 section="Results", strategy="section")


def test_groundedness_accepts_supported_claim_and_flags_unsupported():
    citation_map = {
        "C1": _chunk("The proposed retriever achieved strong accuracy on biomedical "
                     "question answering benchmarks."),
        "C2": _chunk("Unrelated text about tokenizer vocabulary sizes."),
    }
    answer = ("The proposed retriever achieved strong accuracy on biomedical benchmarks [C1]. "
              "It also reduced hospital readmission rates dramatically [C2].")
    score, flagged = _check_groundedness(answer, citation_map)
    assert score == 0.5
    assert len(flagged) == 1 and "[C2]" in flagged[0]


def test_groundedness_with_no_citations_is_vacuously_grounded():
    assert _check_groundedness("No citations here.", {}) == (1.0, [])


# ---------- agent loop with a scripted fake OpenAI client ----------

def _tool_call(call_id, name, arguments):
    return SimpleNamespace(id=call_id, type="function",
                           function=SimpleNamespace(name=name, arguments=arguments))


def _response(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls, role="assistant")
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                           usage=SimpleNamespace(total_tokens=10))


class _ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _make_agent(monkeypatch, responses):
    monkeypatch.setattr("config.OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(agent_mod, "RetrievalPipeline", lambda **kw: object())
    monkeypatch.setattr(agent_mod, "OpenAI", lambda **kw: _ScriptedClient(responses))
    return agent_mod.ResearchAgent()


def test_agent_runs_tool_then_answers(monkeypatch):
    monkeypatch.setitem(agent_mod.TOOL_DISPATCH, "search_arxiv",
                        lambda query, max_results=5: [{"title": f"Paper about {query}"}])
    agent = _make_agent(monkeypatch, [
        _response(tool_calls=[_tool_call("1", "search_arxiv", '{"query": "rag"}')]),
        _response(content="Here is what I found."),
    ])
    trace = agent.run("find rag papers")

    assert trace.final_answer == "Here is what I found."
    assert [s.tool_name for s in trace.steps] == ["search_arxiv"]
    assert "Paper about rag" in trace.steps[0].result_summary
    assert trace.total_tokens == 20


def test_agent_survives_failing_tool_and_bad_arguments(monkeypatch):
    def broken(**kwargs):
        raise ConnectionError("arXiv is down")

    monkeypatch.setitem(agent_mod.TOOL_DISPATCH, "search_arxiv", broken)
    client_responses = [
        _response(tool_calls=[
            _tool_call("1", "search_arxiv", '{"query": "rag"}'),
            _tool_call("2", "citation_graph", "{not json"),
            _tool_call("3", "no_such_tool", "{}"),
        ]),
        _response(content="arXiv is unavailable, sorry."),
    ]
    agent = _make_agent(monkeypatch, client_responses)
    trace = agent.run("find rag papers")

    assert trace.final_answer == "arXiv is unavailable, sorry."
    errors = [json.loads(s.result_summary)["error"] for s in trace.steps]
    assert "arXiv is down" in errors[0]
    assert "not valid JSON" in errors[1]
    assert "Unknown tool" in errors[2]


def test_agent_requires_api_key(monkeypatch):
    import pytest
    monkeypatch.setattr("config.OPENAI_API_KEY", None)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        agent_mod.ResearchAgent()
