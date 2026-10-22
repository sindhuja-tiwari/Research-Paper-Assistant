# Research Paper Assistant: Citation-Grounded RAG over arXiv

[![tests](https://github.com/sindhuja-tiwari/Research-Paper-Assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/sindhuja-tiwari/Research-Paper-Assistant/actions/workflows/tests.yml)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)

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

## Quickstart (macOS or Linux, Python 3.9+)

Works with the Python 3.9 that ships with macOS. No Docker or GPU needed.

```bash
git clone https://github.com/sindhuja-tiwari/Research-Paper-Assistant.git
cd Research-Paper-Assistant

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env     # then open .env and set OPENAI_API_KEY
```

Embeddings and re-ranking run locally with sentence-transformers and need no API key. The first
run downloads two small models (about 100 MB) from Hugging Face. Only generation and the agent's
tool-calling loop call OpenAI (`gpt-4o-mini` by default, set in `config.py`).

## Usage

**1. Build a corpus.** Either rebuild the fixed 8-paper corpus the eval set is written for:

```bash
python -m scripts.ingest --ids_file eval/corpus_ids.txt
```

or index any focused topic (pick a topic rather than indexing all of arXiv):

```bash
python -m scripts.ingest --query "retrieval augmented generation" --max_results 40
```

This downloads the papers, parses them, builds all three chunking strategies and saves one FAISS
index per strategy. arXiv asks clients to wait 3 seconds between requests, so 40 papers take a few
minutes. Downloaded PDFs are kept, so a re-run only fetches what is missing.

**2. Run the retrieval comparison:**

```bash
python -m scripts.run_eval
```

This prints a markdown table of recall@1/3/5 for every chunking strategy, with and without the
re-ranker. It uses `eval/questions.json`, 16 hand-written questions over the fixed corpus. It needs
no OpenAI key. To label your own corpus instead, run `python -m scripts.build_eval_template`, fill
in `eval/questions_template.json` and save it as `eval/questions.json`.

**3. Ask the agent questions:**

```bash
python -m scripts.ask
> How does RAG-Sequence differ from RAG-Token?
> Compare how ColBERT and DPR score passages
> Find recent papers on efficient attention mechanisms
> What papers should I read before understanding Self-RAG?
```

Or ask a single question with `python -m scripts.ask --query "..."`. Each run prints the full
tool-call trace (which tool, which arguments, latency of each step), then the final answer, the
token count and the groundedness score. If a tool fails (arXiv down, Semantic Scholar rate limit),
the agent sees the error and carries on rather than crashing.

**4. Run the tests** (offline, about 10 seconds, no API key or model download):

```bash
pip install -r requirements-dev.txt
pytest
```

## Results

Recall@k on the fixed 8-paper corpus (`eval/questions.json`, 16 questions). A question counts as
a hit when a chunk from the correct paper appears in the top k.

| Strategy | recall@1 | recall@3 | recall@5 | avg query time |
|---|---|---|---|---|
| naive (no rerank) | 0.812 | 0.875 | 0.875 | 189 ms |
| naive + rerank | **0.938** | **1.0** | **1.0** | 8.9 s |
| section (no rerank) | 0.688 | 0.812 | 0.938 | 30 ms |
| section + rerank | 0.875 | **1.0** | **1.0** | 8.0 s |
| section_table (no rerank) | 0.688 | 0.812 | 0.938 | 26 ms |
| section_table + rerank | 0.875 | **1.0** | **1.0** | 19.9 s |

Corpus: 381 naive chunks, 614 section chunks, and 625 section+table chunks (11 tables). Measured
on a MacBook Air CPU with Python 3.9.

**What this shows**

- **Re-ranking is the biggest win.** It lifts recall@1 by 12 to 19 points and brings recall@3 to
  1.0 for every strategy. The cost is latency: scoring 20 candidates with a cross-encoder on a
  laptop CPU takes seconds per query, against tens of milliseconds for vector search alone. The
  timings include loading the model and vary from run to run on a laptop, so treat them as an
  order of magnitude. A GPU, fewer candidates (`TOP_K_RETRIEVE`) or a smaller cross-encoder
  would all cut it.
- **Naive chunking does better than expected without re-ranking.** It scores 0.812 recall@1,
  against 0.688 for section chunks. This metric only asks for the right *paper*. Overlapping
  300-word windows repeat key terms across neighbouring chunks, which helps a paper-level match.
  Section-aware chunking is meant to keep a method or result in one piece for *generation*, and a
  paper-level recall metric can't see that. Section-level labels would test it properly.
- **Table chunks don't move paper-level recall.** These questions are about methods, not numbers,
  and only 11 of 625 chunks are tables. Table-specific questions such as "what accuracy did
  Table 2 report?" are the natural next addition to the eval set.
- **The embedding cache works as intended.** On ingest, the section+table index reused 614 cached
  embeddings and computed only the 11 new table chunks. That took 1.7 s, against 62 s for the
  section index.

## Troubleshooting

- **`OMP: Error #15 ... libomp already initialized` on macOS.** FAISS and PyTorch each bundle
  OpenMP. Run `export KMP_DUPLICATE_LIB_OK=TRUE` before the commands above.
- **`NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+`.** This is harmless with macOS's
  system Python and can be ignored.
- **arXiv rate limit (HTTP 429) or timeouts.** The fetcher already backs off and retries. If it
  still gives up, wait 10 to 15 minutes and re-run. Optionally set `ARXIV_CONTACT_EMAIL` in `.env`.
- **`citation_graph` says Semantic Scholar is unreachable.** The public API shares one rate limit
  across all keyless users. A free key set as `SEMANTIC_SCHOLAR_API_KEY` in `.env` fixes that.

## Project layout

```
Research-Paper-Assistant/
├── config.py             models, chunk sizes, top-k, API endpoints
├── src/                  pdf_parser, chunking, embeddings, vector_store, retrieval,
│                         rerank, generation, agent, tools, arxiv_fetcher, eval
├── scripts/              ingest, build_eval_template, run_eval, ask
├── eval/                 corpus_ids.txt (fixed corpus), questions.json (eval set)
├── tests/                offline unit tests (pytest)
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```

## Known limitations

- The PDF parser is heuristic and will mis-detect headings in unusual paper templates. GROBID
  would fix this at the cost of extra infrastructure.
- The groundedness check is lexical rather than semantic: it catches gross misattribution, not
  subtle misattribution.
- The cross-paper comparison metric is a simplification (paper-level, optionally section-level
  match) rather than exact chunk-level ground truth, which would require hand-labelling every chunk.
- The eval set is small (8 papers, 16 questions) and labelled at paper level, so a few questions
  swing recall by 6 points each. It is a sanity check for comparing configurations, not a
  benchmark.
- Metadata filtering is applied after retrieval (over-fetch, then filter). That is fine at this
  corpus size but would not scale to millions of chunks without pushing filters into the index.
