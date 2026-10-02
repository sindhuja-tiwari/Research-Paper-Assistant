"""
The agent layer: a tool-calling loop where the LLM decides whether to
(a) search the local paper index (RAG retrieval), (b) search live arXiv,
(c) pull the citation graph, or (d) just summarize a paper -- then combines
results across possibly-multiple tool calls into a final answer.

The full trace (which tools were called, in what order, with what args) is
logged and returned, because that trace is your best demo artifact: showing
a multi-step "search -> realize it needs more -> call another tool -> answer"
sequence is far more convincing live than describing it.
"""
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from openai import OpenAI

import config
from src.tools import TOOL_SCHEMAS, TOOL_DISPATCH
from src.retrieval import RetrievalPipeline
from src.generation import generate_answer, GenerationResult

AGENT_SYSTEM_PROMPT = """You are a research assistant that helps students find and understand
academic papers. You have access to these tools:

- search_local_papers: search the ALREADY-INDEXED paper corpus for an answer (use this FIRST
  for questions about specific content, methods, results, or comparisons between indexed papers).
- search_arxiv: search live arXiv for papers not yet indexed.
- citation_graph: find what a paper cites, or what cites it.
- summarize_paper: get a fast abstract + outline for an indexed paper.

Use multiple tools in sequence when a question needs it (e.g. a comparison question may need
search_local_papers twice, once per paper). Always ground factual claims using
search_local_papers results with their citation keys -- don't answer from general knowledge
if the local index can answer it.
"""


@dataclass
class AgentStep:
    tool_name: str
    arguments: Dict[str, Any]
    result_summary: str
    latency_ms: float


@dataclass
class AgentTrace:
    query: str
    steps: List[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    groundedness_score: Optional[float] = None


class ResearchAgent:
    def __init__(self, strategy: str = "section_table", use_reranker: bool = True):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.retrieval = RetrievalPipeline(strategy=strategy, use_reranker=use_reranker)

        # search_local_papers is defined here (not tools.py) because it needs
        # the live RetrievalPipeline instance, not just a stateless function.
        self._local_schema = {
            "type": "function",
            "function": {
                "name": "search_local_papers",
                "description": "Search the locally indexed paper corpus and get a "
                                "citation-grounded answer. Use this for any question "
                                "about content, methods, results, or comparisons.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "year_min": {"type": "integer", "description": "Optional: only papers from this year onward"},
                    },
                    "required": ["question"],
                },
            },
        }
        self.all_schemas = [self._local_schema] + TOOL_SCHEMAS
        self._last_generation: Optional[GenerationResult] = None

    def _dispatch(self, name: str, args: Dict[str, Any]) -> str:
        if name == "search_local_papers":
            retrieved = self.retrieval.retrieve(args["question"], year_min=args.get("year_min"))
            gen = generate_answer(args["question"], retrieved["results"])
            self._last_generation = gen
            return gen.answer
        else:
            fn = TOOL_DISPATCH[name]
            result = fn(**args)
            return json.dumps(result)

    def run(self, query: str, max_steps: int = 5) -> AgentTrace:
        trace = AgentTrace(query=query)
        t_start = time.time()

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        for _ in range(max_steps):
            resp = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                tools=self.all_schemas,
                tool_choice="auto",
                temperature=0.1,
            )
            msg = resp.choices[0].message
            trace.total_tokens += resp.usage.total_tokens

            if not msg.tool_calls:
                trace.final_answer = msg.content
                break

            messages.append(msg)
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                t0 = time.time()
                result_str = self._dispatch(name, args)
                elapsed_ms = (time.time() - t0) * 1000

                trace.steps.append(AgentStep(
                    tool_name=name, arguments=args,
                    result_summary=result_str[:300],
                    latency_ms=elapsed_ms,
                ))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })
        else:
            trace.final_answer = "(hit max_steps without a final answer -- consider raising max_steps)"

        trace.total_latency_ms = (time.time() - t_start) * 1000
        if self._last_generation is not None:
            trace.groundedness_score = self._last_generation.groundedness_score

        return trace


def print_trace(trace: AgentTrace):
    print(f"\n=== Query: {trace.query} ===")
    for i, step in enumerate(trace.steps, start=1):
        print(f"  Step {i}: {step.tool_name}({step.arguments})  [{step.latency_ms:.0f}ms]")
        print(f"           -> {step.result_summary[:150]}...")
    print(f"\nFinal answer:\n{trace.final_answer}")
    print(f"\n(total: {trace.total_latency_ms:.0f}ms, {trace.total_tokens} tokens, "
          f"groundedness: {trace.groundedness_score})")
