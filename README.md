# Research Paper Assistant: Citation-Grounded RAG over arXiv

Finding relevant papers and then decoding them takes students hours. This retrieval-augmented
agent answers questions about a focused corpus of papers, compares papers against each other, and
can pull in new papers from arXiv on demand. Every answer is grounded in, and cited back to, the
source text.

## Features

- Three chunking strategies (naive fixed-window, section-aware, section-aware plus tables),
  evaluated head to head
- Two-stage retrieval: vector search (top 20), then cross-encoder re-ranking (top 5)
- Embedding cache keyed by content hash, with hit/miss statistics printed on every ingest
- Citation-grounded generation with an automated groundedness check
- Tool-calling agent: local paper search, live arXiv search, citation graph, paper summarizer
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
                    citation_graph and summarize_paper tools
```

## Design decisions

**Chunking.** Fixed-window chunking splits method descriptions across chunk boundaries and merges
unrelated paragraphs. Section-aware chunking fixes that but flattens tables into prose, which
breaks numeric questions such as "what accuracy did Table 2 report?". All three strategies are
built and compared on the same questions (`scripts/run_eval.py`) rather than picking one up front.

**Heuristic PDF parser instead of GROBID.** GROBID gives cleaner section boundaries but runs as a
separate Java service. The parser here uses font-size-based heading detection and grid-pattern
table detection, which needs no extra infrastructure. Swapping in GROBID is a one-file change in
`src/pdf_parser.py`.

**Two-stage retrieval.** A cross-encoder scores a (query, chunk) pair jointly and is more accurate
than comparing independent embeddings, but it is too slow to run over a whole corpus. Vector search
narrows the corpus to about 20 candidates, then the cross-encoder picks the best 5. `eval.py`
reports both the accuracy gain and the latency cost of the extra stage.

**Embedding cache.** Re-indexing after one parsing fix or one new paper should not recompute every
unchanged chunk. The cache is keyed by content hash, and `scripts/ingest.py` prints hit/miss
statistics on every run.

**Groundedness check.** Language models sometimes cite sources that do not support their claim.
The check in `src/generation.py` is a lexical-overlap test that catches the common failure of
citing a chunk that shares no real content with the sentence. It does not detect subtle
misattribution (see limitations).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env     # then set OPENAI_API_KEY in .env
```

Embeddings and re-ranking run locally with sentence-transformers and need no API key. Only
generation and the agent's tool-calling loop call OpenAI.

## Usage

**1. Build a corpus** (pick a focused topic rather than indexing all of arXiv):

```bash
python -m scripts.ingest --query "retrieval augmented generation" --max_results 40
```

Downloads the papers, parses them, builds all three chunking strategies and saves one FAISS index
per strategy.

**2. Create an evaluation set:**

```bash
python -m scripts.build_eval_template
# edit eval/questions_template.json with real questions, save as eval/questions.json
```

**3. Run the comparison:**

```bash
python -m scripts.run_eval
```

Prints a markdown table of recall@1/3/5 for every chunking strategy and re-ranker combination.

**4. Ask the agent questions:**

```bash
python -m scripts.ask
> What dataset did [paper] use for evaluation?
> Compare how [paper A] and [paper B] approach long-context retrieval
> Find recent papers on efficient attention mechanisms
> What papers should I read before understanding [paper]?
```

Each run prints the full tool-call trace (which tool, which arguments, latency of each step)
before the final answer.

## Project layout

```
Research-Paper-Assistant/
├── src/        config, pdf_parser, chunking, embeddings, vector_store, retrieval,
│               rerank, generation, agent, tools, arxiv_fetcher, eval
├── scripts/    ingest, build_eval_template, run_eval, ask
├── eval/       evaluation questions
├── requirements.txt
└── .env.example
```

## Known limitations

- The PDF parser is heuristic and will mis-detect headings in unusual paper templates. GROBID
  would fix this at the cost of extra infrastructure.
- The groundedness check is lexical rather than semantic: it catches gross misattribution, not
  subtle misattribution.
- The cross-paper comparison metric is a simplification (paper-level, optionally section-level
  match) rather than exact chunk-level ground truth, which would require hand-labelling every chunk.
- Metadata filtering is applied after retrieval (over-fetch, then filter). That is fine at this
  corpus size but would not scale to millions of chunks without pushing filters into the index.
