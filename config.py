import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"       
CACHE_DIR = DATA_DIR / "cache"               
INDEX_DIR = DATA_DIR / "index"               
EVAL_DIR = ROOT_DIR / "eval"

for d in [RAW_PDF_DIR, PROCESSED_DIR, CACHE_DIR, INDEX_DIR, EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

LLM_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

NAIVE_CHUNK_SIZE = 300        # tokens (approx, word-based)
NAIVE_CHUNK_OVERLAP = 50
MAX_SECTION_CHUNK_SIZE = 400  # a section longer than this gets sub-split

# --- Retrieval ---
TOP_K_RETRIEVE = 20    # first-stage vector retrieval
TOP_K_RERANK = 5       # final number of chunks passed to the LLM after re-ranking

# --- arXiv ---
ARXIV_API_URL = "http://export.arxiv.org/api/query"

# --- Semantic Scholar (citation graph tool) ---
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
