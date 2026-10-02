"""
Generates an answer grounded in retrieved chunks, with inline citations like
[Vaswani et al., 2017 - Method], and then PROGRAMMATICALLY checks that the
answer's claims are actually supported by the cited chunks. This groundedness
check is a real production-RAG concern (hallucination/attribution) and is the
kind of thing that separates "I called an LLM" from "I built a system."
"""
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Any

from openai import OpenAI

import config

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


SYSTEM_PROMPT = """You are a research assistant that answers questions about academic papers.
You will be given a question and a set of retrieved excerpts, each tagged with a citation key
like [C1], [C2], etc.

Rules:
1. Answer ONLY using the provided excerpts. If they don't contain the answer, say so plainly.
2. Every factual claim in your answer MUST end with the citation key(s) it came from, e.g. "... achieved 94% accuracy [C2]."
3. Be concise and precise. Do not pad the answer with generic commentary.
"""


def _build_context_block(results: List[Dict[str, Any]]) -> str:
    lines = []
    for i, r in enumerate(results, start=1):
        c = r["chunk"]
        lines.append(f"[C{i}] (Paper: {c.title}, {c.year}, Section: {c.section})\n{c.text}\n")
    return "\n".join(lines)


@dataclass
class GenerationResult:
    answer: str
    citation_map: Dict[str, Any]        # "C1" -> chunk
    groundedness_score: float           # fraction of cited claims verified
    flagged_claims: List[str]           # claims citing a key whose text doesn't actually support them
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


def generate_answer(query: str, results: List[Dict[str, Any]]) -> GenerationResult:
    citation_map = {f"C{i}": r["chunk"] for i, r in enumerate(results, start=1)}
    context_block = _build_context_block(results)

    user_prompt = f"Question: {query}\n\nRetrieved excerpts:\n{context_block}"

    client = get_client()
    t0 = time.time()
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    latency_ms = (time.time() - t0) * 1000

    answer = resp.choices[0].message.content
    usage = resp.usage

    groundedness_score, flagged = _check_groundedness(answer, citation_map)

    return GenerationResult(
        answer=answer,
        citation_map=citation_map,
        groundedness_score=groundedness_score,
        flagged_claims=flagged,
        latency_ms=latency_ms,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )


def _check_groundedness(answer: str, citation_map: Dict[str, Any]) -> (float, List[str]):
    """
    Lightweight automated check: split the answer into sentences, find each
    sentence's cited key(s), and verify there's reasonable lexical overlap
    between the sentence and the cited chunk's text. This won't catch subtle
    misattribution, but it DOES catch the common failure mode of citing a
    chunk that shares no content with the claim -- and it's cheap to run on
    every answer, unlike an LLM-as-judge call.
    """
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    flagged = []
    checked = 0

    for sent in sentences:
        keys = re.findall(r"\[(C\d+)\]", sent)
        if not keys:
            continue
        checked += 1
        sent_words = set(w.lower() for w in re.findall(r"\w+", sent) if len(w) > 4)
        supported = False
        for key in keys:
            chunk = citation_map.get(key)
            if chunk is None:
                continue
            chunk_words = set(w.lower() for w in re.findall(r"\w+", chunk.text) if len(w) > 4)
            overlap = len(sent_words & chunk_words)
            if overlap >= 3:  # crude but effective threshold
                supported = True
                break
        if not supported:
            flagged.append(sent.strip())

    score = 1.0 if checked == 0 else (checked - len(flagged)) / checked
    return round(score, 3), flagged
