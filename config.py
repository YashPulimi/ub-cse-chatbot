"""
config.py — Central configuration for UB CSE Chatbot
=====================================================
All settings live here. Override via environment variables or .env file.
Never hardcode secrets or paths anywhere else in the codebase.

Usage:
    from config import cfg
    print(cfg.qdrant.collection)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (silently ignored if missing)
load_dotenv(Path(__file__).parent / ".env")


# ── helpers ───────────────────────────────────────────────────────────────────

def _env(key: str, default: str) -> str:
    return os.getenv(key, default)

def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))

def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))

def _env_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes")


# ── sub-configs ───────────────────────────────────────────────────────────────

@dataclass
class PathsConfig:
    base:        Path = field(default_factory=lambda: Path(__file__).parent)
    data:        Path = field(init=False)
    raw:         Path = field(init=False)
    pdfs:        Path = field(init=False)
    processed:   Path = field(init=False)
    indexes:     Path = field(init=False)
    eval_dir:    Path = field(init=False)
    logs:        Path = field(init=False)
    graph:       Path = field(init=False)   # NetworkX fallback pickle
    bm25:        Path = field(init=False)
    embed_cache: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data        = self.base / "data"
        self.raw         = self.data / "raw"
        self.pdfs        = self.raw  / "pdfs"
        self.processed   = self.data / "processed"
        self.indexes     = self.data / "indexes"
        self.eval_dir    = self.data / "eval"
        self.logs        = self.base / "logs"
        self.graph       = self.indexes / "kg.gpickle"
        self.bm25        = self.indexes / "bm25.pkl"
        self.embed_cache = self.indexes / "embed_cache.pkl"
        # Create all dirs on import — every module can write immediately
        for p in [self.raw, self.pdfs, self.processed,
                  self.indexes, self.eval_dir, self.logs]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class CrawlerConfig:
    max_pages:    int = field(default_factory=lambda: _env_int("CRAWLER_MAX_PAGES",   600))
    max_depth:    int = field(default_factory=lambda: _env_int("CRAWLER_MAX_DEPTH",   6))
    concurrency:  int = field(default_factory=lambda: _env_int("CRAWLER_CONCURRENCY", 10))
    pdf_max_kb:   int = field(default_factory=lambda: _env_int("PDF_MAX_KB",          5000))
    # Block appears on ≥ this many pages before being treated as boilerplate
    bp_threshold: int = field(default_factory=lambda: _env_int("BP_THRESHOLD",        12))


@dataclass
class ChunkerConfig:
    # Token-approximate sizes  (1 token ≈ 0.75 English words)
    chunk_size:      int = field(default_factory=lambda: _env_int("CHUNK_SIZE",      512))
    chunk_overlap:   int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP",   64))
    min_chunk_words: int = field(default_factory=lambda: _env_int("MIN_CHUNK_WORDS", 30))
    # Sentences to consider around each semantic boundary
    semantic_buffer: int = field(default_factory=lambda: _env_int("SEMANTIC_BUFFER", 1))


@dataclass
class EmbeddingConfig:
    # BAAI/bge-small-en-v1.5  → 384-dim, fast on CPU  (default)
    # BAAI/bge-base-en-v1.5   → 768-dim, better quality
    # nomic-ai/nomic-embed-text-v1.5 → 768-dim, best, needs GPU
    model:      str = field(default_factory=lambda: _env("EMBED_MODEL",       "BAAI/bge-small-en-v1.5"))
    batch_size: int = field(default_factory=lambda: _env_int("EMBED_BATCH_SIZE", 64))
    dim:        int = field(default_factory=lambda: _env_int("EMBED_DIM",        384))
    device:     str = field(default_factory=lambda: _env("EMBED_DEVICE",      "cpu"))
    # Set EMBED_DEVICE=cuda if GPU available; "auto" triggers detection in embeddings.py


@dataclass
class QdrantConfig:
    # Leave QDRANT_HOST empty → local on-disk mode, no Docker needed for dev
    host:       str  = field(default_factory=lambda: _env("QDRANT_HOST",       ""))
    port:       int  = field(default_factory=lambda: _env_int("QDRANT_PORT",   6333))
    local_path: str  = field(default_factory=lambda: _env("QDRANT_LOCAL_PATH", "data/indexes/qdrant_db"))
    collection: str  = field(default_factory=lambda: _env("QDRANT_COLLECTION", "ub_cse"))
    use_grpc:   bool = field(default_factory=lambda: _env_bool("QDRANT_USE_GRPC", False))


@dataclass
class Neo4jConfig:
    uri:      str = field(default_factory=lambda: _env("NEO4J_URI",      "bolt://localhost:7687"))
    user:     str = field(default_factory=lambda: _env("NEO4J_USER",     "neo4j"))
    password: str = field(default_factory=lambda: _env("NEO4J_PASSWORD", "ubcse2025"))
    database: str = field(default_factory=lambda: _env("NEO4J_DATABASE", "neo4j"))


@dataclass
class BM25Config:
    k1:    float = field(default_factory=lambda: _env_float("BM25_K1",  1.5))
    b:     float = field(default_factory=lambda: _env_float("BM25_B",   0.75))
    top_k: int   = field(default_factory=lambda: _env_int("BM25_TOP_K", 20))


@dataclass
class RetrievalConfig:
    top_k:           int   = field(default_factory=lambda: _env_int("RETRIEVAL_TOP_K",    20))
    rrf_k:           int   = field(default_factory=lambda: _env_int("RRF_K",              60))
    dense_weight:    float = field(default_factory=lambda: _env_float("DENSE_WEIGHT",     0.6))
    bm25_weight:     float = field(default_factory=lambda: _env_float("BM25_WEIGHT",      0.4))
    graph_hops:      int   = field(default_factory=lambda: _env_int("GRAPH_HOPS",         2))
    graph_max_nodes: int   = field(default_factory=lambda: _env_int("GRAPH_MAX_NODES",    40))


@dataclass
class RerankerConfig:
    # cross-encoder/ms-marco-MiniLM-L-6-v2 → fast, good quality  (default)
    # BAAI/bge-reranker-base                → slightly better, slower
    model:        str  = field(default_factory=lambda: _env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    top_k:        int  = field(default_factory=lambda: _env_int("RERANK_TOP_K", 5))
    use_executor: bool = field(default_factory=lambda: _env_bool("RERANKER_USE_EXECUTOR", True))


@dataclass
class LLMConfig:
    # Pull model with:  ollama pull qwen2.5:3b
    # Options: qwen2.5:3b (fast/default) | qwen2.5:7b (quality) | llama3.2:3b
    model:           str   = field(default_factory=lambda: _env("LLM_MODEL",           "qwen2.5:3b"))
    ollama_base_url: str   = field(default_factory=lambda: _env("OLLAMA_BASE_URL",     "http://localhost:11434"))
    max_new_tokens:  int   = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS",  512))
    temperature:     float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.1))
    stream:          bool  = field(default_factory=lambda: _env_bool("LLM_STREAM",     True))
    context_window:  int   = field(default_factory=lambda: _env_int("LLM_CONTEXT_WINDOW", 4096))


@dataclass
class GuardrailConfig:
    # "nemo"       → NeMo Guardrails (best, slower to start)
    # "classifier" → lightweight local model  (default)
    # "rules"      → regex rules only (fastest, least precise)
    mode:            str   = field(default_factory=lambda: _env("GUARDRAIL_MODE",       "classifier"))
    threshold:       float = field(default_factory=lambda: _env_float("GUARDRAIL_THRESHOLD", 0.75))
    nemo_config_dir: str   = field(default_factory=lambda: _env("NEMO_CONFIG_DIR",      "guardrails/"))


@dataclass
class MemoryConfig:
    max_turns:       int  = field(default_factory=lambda: _env_int("MEMORY_MAX_TURNS",     8))
    summary_every:   int  = field(default_factory=lambda: _env_int("MEMORY_SUMMARY_EVERY", 6))
    ask_personalize: bool = field(default_factory=lambda: _env_bool("MEMORY_ASK_PERSONALIZE", True))


@dataclass
class CacheConfig:
    query_ttl:       int = field(default_factory=lambda: _env_int("CACHE_QUERY_TTL",    300))
    result_max_size: int = field(default_factory=lambda: _env_int("CACHE_RESULT_SIZE",  256))
    embed_max_size:  int = field(default_factory=lambda: _env_int("CACHE_EMBED_SIZE",  4096))


@dataclass
class EvalConfig:
    dataset_path: str  = field(default_factory=lambda: _env("EVAL_DATASET", "data/eval/eval_dataset.json"))
    output_path:  str  = field(default_factory=lambda: _env("EVAL_OUTPUT",  "data/eval/eval_results.json"))
    k_values:     list = field(default_factory=lambda: [1, 3, 5, 10])


# ── root config ───────────────────────────────────────────────────────────────

@dataclass
class Config:
    paths:     PathsConfig     = field(default_factory=PathsConfig)
    crawler:   CrawlerConfig   = field(default_factory=CrawlerConfig)
    chunker:   ChunkerConfig   = field(default_factory=ChunkerConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    qdrant:    QdrantConfig    = field(default_factory=QdrantConfig)
    neo4j:     Neo4jConfig     = field(default_factory=Neo4jConfig)
    bm25:      BM25Config      = field(default_factory=BM25Config)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker:  RerankerConfig  = field(default_factory=RerankerConfig)
    llm:       LLMConfig       = field(default_factory=LLMConfig)
    guardrail: GuardrailConfig = field(default_factory=GuardrailConfig)
    memory:    MemoryConfig    = field(default_factory=MemoryConfig)
    cache:     CacheConfig     = field(default_factory=CacheConfig)
    eval:      EvalConfig      = field(default_factory=EvalConfig)


# ── singleton ─────────────────────────────────────────────────────────────────
# Import everywhere:   from config import cfg
cfg = Config()
