# Research Paper Assistant — A RAG System with Real Engineering Depth

Students waste hours hunting for relevant papers and then more hours decoding them.
This is a retrieval-augmented agent that answers questions about a focused corpus
of papers, compares papers against each other, and can pull in new papers live
from arXiv — with every answer grounded in and cited back to the source text.

This is **not** "built a chatbot with LangChain." It's built to demonstrate the
engineering decisions that separate a toy RAG demo from something closer to a
production system:

- Three chunking strategies, empirically compared (not just picked)
- Two-stage retrieval: vector search → cross-encoder re-ranking
- Embedding caching with measured before/after savings
- Citation-grounded generation with an automated groundedness check
- A tool-calling agent (local RAG search, live arXiv search, citation graph, summarizer)
- Per-stage latency and token-cost instrumentation on every query

## Architecture

```
arXiv API ──▶ PDF download ──▶ PyMuPDF parsing (sections + tables)
                                        │
                                        ▼
                    ┌───────────────────────────────────┐
                    │   3 chunking strategies (compare)  │
                    │   naive │ section-aware │ +tables  │
                    └───────────────────────────────────┘
                                        │
                      sentence-transformers embeddings
                         (disk-cached by content hash)
                                        │
                                        ▼
                               FAISS vector index
                                        │
                         vector search (top-20) ──▶ cross-encoder re-rank (top-5)
                                        │
                                        ▼
                       citation-grounded generation (OpenAI)
                          + automated groundedness check
                                        │
                                        ▼
                    Agent loop (tool-calling): wraps the above as
                    "search_local_papers" alongside search_arxiv,
                    citation_graph, and summarize_paper tools
```

## Why these specific engineering choices

**Why compare chunking strategies instead of picking one?**
Naive fixed-window chunking is the default in most tutorials, but it routinely
splits a method description across chunk boundaries or merges unrelated
paragraphs. Section-aware chunking fixes that but still flattens tables into
prose, which destroys numeric answers like "what accuracy did Table 2 report?"
The point isn't that section+table chunking "wins" — it's being able to show
*why*, with numbers (`scripts/run_eval.py`).

**Why a heuristic PDF parser instead of GROBID?**
GROBID (the tool real research-tooling teams use) gives much cleaner section
boundaries, but it runs as a separate Java service — real infra overhead for a
solo project. The heuristic parser here (font-size-based heading detection +
grid-pattern table detection) gets most of the value with zero extra
infrastructure. Swapping in GROBID later is a one-file change in
`src/pdf_parser.py` — worth mentioning as a known limitation, not hiding it.

**Why two-stage retrieval instead of just vector search?**
Cross-encoders score a (query, chunk) pair jointly and are meaningfully more
accurate than comparing independent embeddings — but they're too slow to run
over an entire corpus. So vector search narrows thousands of chunks to ~20
fast, then the cross-encoder picks the best 5 from that shortlist. `eval.py`
reports the accuracy gain and the latency cost of this extra stage, so you can
argue the trade-off with real numbers instead of just asserting it's better.

**Why cache embeddings?**
Re-indexing after fixing one parsing bug or adding one new paper shouldn't
recompute embeddings for every unchanged chunk. The cache is keyed by content
hash; `scripts/ingest.py` prints hit/miss stats on every run.

**Why a groundedness check?**
LLMs confidently cite sources that don't actually support their claim. The
check in `src/generation.py` isn't a full hallucination detector — it's a
cheap, explainable lexical-overlap check that catches the common failure mode
(citing a chunk sharing no real content with the sentence). Its limits are
documented in the code, not hidden.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
```

Embeddings and re-ranking run locally via sentence-transformers — free, no API
key needed. Only generation and the agent's tool-calling loop call OpenAI.

## Usage

**1. Build your corpus** (pick a focused topic — don't try to index all of arXiv):
```bash
python -m scripts.ingest --query "retrieval augmented generation" --max_results 40
```
This downloads papers, parses them, builds all three chunking strategies, and
saves a FAISS index per strategy.

**2. Build and fill in an eval set:**
```bash
python -m scripts.build_eval_template
# edit eval/questions_template.json with real questions, save as eval/questions.json
```

**3. Run the comparison:**
```bash
python -m scripts.run_eval
```
Outputs a markdown table of recall@1/3/5 for every strategy × reranker
combination — paste this straight into your own write-up.

**4. Ask the agent questions** (this is the demo):
```bash
python -m scripts.ask
> What dataset did [paper] use for evaluation?
> Compare how [paper A] and [paper B] approach long-context retrieval
> Find recent papers on efficient attention mechanisms
> What papers should I read before understanding [paper]?
```
Each run prints the full tool-call trace — which tool was called, with what
arguments, and the latency of each step — before the final answer.

## Known limitations (worth stating up front, not discovering in an interview)

- The PDF parser is heuristic; it will mis-detect headings in unusual paper
  templates. GROBID would fix this at the cost of extra infrastructure.
- The groundedness check is lexical-overlap based, not semantic — it catches
  gross misattribution, not subtle ones.
- The cross-paper comparison eval metric is a simplification (paper-level,
  optionally section-level match) rather than exact chunk-level ground truth,
  which would require hand-labeling every chunk.
- Metadata filtering is post-hoc (over-fetch then filter), which is fine at
  this corpus size but wouldn't scale to millions of chunks without pushing
  filters into the index itself.

## Suggested resume bullets

> Built a research-paper RAG agent with section- and table-aware chunking,
> two-stage retrieval (FAISS + cross-encoder re-ranking), and citation-grounded
> generation with automated groundedness checks; empirically compared 3
> chunking strategies, improving recall@5 by X% over naive fixed-window
> chunking. Added a tool-calling agent layer (live arXiv search, citation-graph
> traversal, paper summarization) with full latency/cost instrumentation per
> pipeline stage.

Fill in the X% from your own `scripts/run_eval.py` run — don't guess a number,
measure it. An interviewer asking "how did you get that number" and you having
a real answer is the entire point of this project.
