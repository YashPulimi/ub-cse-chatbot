# UB CSE Chatbot — RAG + GraphRAG Pipeline

**CSE 635 Final Project** | University at Buffalo | Spring 2026

A production-grade chatbot for the UB Computer Science & Engineering department.
Combines dense retrieval (Qdrant), sparse retrieval (BM25), and a knowledge graph
(Neo4j/NetworkX) with a cross-encoder reranker and a local LLM (Ollama/Qwen2.5).

---

## Project Structure

```
ub_cse_chatbot/
│
├── data/                          ← auto-created on first run
│   ├── raw/
│   │   ├── crawl_<timestamp>.jsonl   crawled pages
│   │   ├── summary.json
│   │   └── pdfs/                     downloaded PDFs
│   ├── processed/
│   │   ├── processed_docs.jsonl      LlamaIndex Documents
│   │   ├── ingestion_summary.json
│   │   ├── chunks.jsonl              TextNode chunks
│   │   └── chunks_summary.json
│   ├── indexes/
│   │   ├── qdrant_db/                Qdrant local vector store
│   │   ├── bm25.pkl                  BM25 index
│   │   ├── embed_cache.pkl           embedding cache
│   │   ├── graph_data.json           raw extracted entities
│   │   └── kg.gpickle                NetworkX graph (if no Neo4j)
│   └── eval/
│       ├── eval_results.json
│       └── eval_summary.json
│
├── logs/                          ← per-module log files
│
├── Pipeline (run in order)
├── crawler.py                     S3  — async Crawl4AI BFS crawler
├── ingestion_pipeline.py          S4  — crawler JSONL → LlamaIndex Documents
├── chunker.py                     S5  — Documents → TextNode chunks
├── embeddings.py                  S6  — BGE-small embedding model wrapper
├── vector_store.py                S7  — Qdrant dense index
├── sparse_retriever.py            S8  — BM25 keyword index
├── graph_store.py                 S9  — Neo4j / NetworkX knowledge graph
├── graph_retriever.py             S10 — graph query layer
├── retriever.py                   S11 — hybrid RRF fusion (dense+BM25+graph)
├── reranker.py                    S12 — cross-encoder reranker
├── guardrails.py                  S13 — 3-pass guardrail classifier
├── memory.py                      S14 — dialogue memory + coreference
├── generator.py                   S15 — Ollama streaming LLM
│
├── App
├── server.py                      FastAPI backend (replaces Chainlit)
├── index.html                     UB-branded single-file frontend
│
├── Evaluation
├── evaluator.py                   RAGAS + IR metrics + latency + guardrails
│
├── Config
├── config.py                      central config (all env-overridable)
├── utils.py                       shared helpers
├── .env                           secrets + overrides (copy from env.example)
├── env.example                    template
└── requirements.txt
```

---

## Quick Start

### 1. Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11 or 3.12 | [python.org](https://python.org) |
| Ollama | latest | [ollama.com/download](https://ollama.com/download) |
| Neo4j | 5.x (optional) | Docker command below |

> **Python 3.14 is NOT supported** — `torch` has no wheels for it yet.

---

### 2. Environment setup

```bash
# Clone / enter project directory
cd ub_cse_chatbot

# Create virtual environment with Python 3.11
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Verify Python version
python --version                   # must say 3.11.x or 3.12.x

# Install dependencies (two-step due to aiofiles conflict)
pip install chainlit>=1.2.0 sentence-transformers>=3.0.0 \
    transformers>=4.40.0 torch>=2.2.0 huggingface-hub>=0.23.0 \
    qdrant-client>=1.12.0 rank-bm25>=0.2.2 neo4j>=5.20.0 networkx>=3.2.0 \
    llama-index-core>=0.12.0 llama-index-readers-file>=0.4.0 \
    llama-index-embeddings-huggingface>=0.5.0 \
    llama-index-vector-stores-qdrant>=0.5.0 \
    llama-index-retrievers-bm25>=0.5.0 \
    llama-index-graph-stores-neo4j>=0.5.0 \
    llama-index-llms-ollama>=0.5.0 \
    fastapi "uvicorn[standard]" \
    httpx>=0.27.0 aiohttp>=3.10.0 nest-asyncio>=1.6.0 \
    python-dotenv>=1.0.0 PyYAML>=6.0.1 \
    pypdf>=4.2.0 pdfplumber>=0.11.0 \
    beautifulsoup4>=4.12.0 lxml>=5.2.0 \
    ollama>=0.3.0 ragas>=0.2.0 datasets>=2.20.0 \
    numpy>=1.26.0 pandas>=2.2.0 tqdm>=4.66.0 \
    tenacity>=8.3.0 loguru>=0.7.0

pip install crawl4ai --no-deps
pip install "aiofiles>=24.1.0"
crawl4ai-setup
```

---

### 3. Start Ollama and pull the model

```bash
# In a SEPARATE terminal — keep it running
ollama serve

# Back in your project terminal
ollama pull qwen2.5:3b
```

---

### 4. (Optional) Start Neo4j with Docker

```bash
docker run -d \
  --name neo4j-ubcse \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/ubcse2025 \
  neo4j:5
```

> Skip this if you don't have Docker — `graph_store.py` falls back to NetworkX automatically.

---

### 5. Configure

```bash
cp env.example .env
# Edit .env if needed — all defaults work out of the box
```

---

### 6. Run the pipeline (one time)

Run these in order. Each step produces files consumed by the next.

```bash
# Step 1 — Crawl the UB CSE website (~10-20 min)
python crawler.py

# Step 2 — Convert crawled pages to LlamaIndex Documents (~30 sec)
python ingestion_pipeline.py

# Step 3 — Chunk documents into TextNodes (~2-5 min)
python chunker.py

# Step 4 — Build Qdrant vector index (~5-10 min)
python vector_store.py

# Step 5 — Build BM25 keyword index (~10 sec)
python sparse_retriever.py

# Step 6 — Build knowledge graph (~30 sec)
python graph_store.py
```

**Expected output after all steps:**
```
data/raw/crawl_<timestamp>.jsonl     ~400-600 pages
data/processed/chunks.jsonl          ~3000-4000 chunks
data/indexes/qdrant_db/              vector store
data/indexes/bm25.pkl                BM25 index
data/indexes/kg.gpickle              graph (or Neo4j if Docker running)
```

---

### 7. Launch the app

```bash
python server.py
```

Open **http://localhost:8000** in your browser.

---

## Rebuild flags (if you need to re-run a step)

```bash
python vector_store.py --reset       # drop and rebuild Qdrant collection
python sparse_retriever.py --rebuild # rebuild BM25 from chunks.jsonl
python graph_store.py --rebuild      # clear and rebuild knowledge graph
```

---

## Test individual components

```bash
# Dense search
python vector_store.py --search "who teaches NLP"

# BM25 search
python sparse_retriever.py --search "CSE 574 prerequisites"

# Graph query
python graph_store.py --query "CSE 574"
python graph_store.py --query "Rohini Srihari"
python graph_store.py --stats

# Full retrieval pipeline
python retriever.py --query "who teaches computer vision" --verbose

# Reranker
python reranker.py --query "MS admission requirements"

# Guardrails (no indexes needed)
python guardrails.py --batch
python guardrails.py --query "Where is the best pizza in Buffalo?"

# Generator (requires Ollama running + indexes built)
python generator.py --query "who teaches CSE 574"
```

---

## Evaluation

```bash
# Retrieval metrics only (no Ollama needed)
python evaluator.py --retrieval-only

# Guardrail robustness only (no indexes needed)
python evaluator.py --guardrails-only

# Full evaluation (Ollama must be running)
python evaluator.py

# Results saved to:
#   data/eval/eval_results.json
#   data/eval/eval_summary.json
```

---

## Configuration

All settings live in `config.py` and can be overridden via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `qwen2.5:3b` | Ollama model name |
| `LLM_MAX_TOKENS` | `512` | Max tokens to generate |
| `LLM_CONTEXT_WINDOW` | `4096` | Context window size |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `RETRIEVAL_TOP_K` | `20` | Candidates before reranking |
| `RERANK_TOP_K` | `5` | Final chunks sent to LLM |
| `CRAWLER_MAX_PAGES` | `600` | Max pages to crawl |
| `QDRANT_COLLECTION` | `ub_cse` | Qdrant collection name |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection |
| `GUARDRAIL_MODE` | `classifier` | `rules` / `classifier` / `nemo` |

---

## Architecture

```
User query
    │
    ▼
┌─────────────┐
│  Guardrails │  3-pass: rules (<1ms) → NLI classifier (~30ms) → LLM policy
└──────┬──────┘
       │ allowed
       ▼
┌──────────────────────────────────────────────┐
│              Hybrid Retriever                │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │  Dense   │  │   BM25   │  │   Graph   │  │  concurrent
│  │ (Qdrant) │  │ (rank-   │  │ (Neo4j /  │  │  via asyncio
│  │ bge-small│  │  bm25)   │  │ NetworkX) │  │
│  └──────────┘  └──────────┘  └───────────┘  │
│              RRF Fusion (k=60)               │
└──────────────────┬───────────────────────────┘
                   │ top-20 candidates
                   ▼
         ┌──────────────────┐
         │  Cross-Encoder   │  ms-marco-MiniLM-L-6-v2
         │    Reranker      │  top-20 → top-5
         └────────┬─────────┘
                  │ top-5 chunks + graph evidence
                  ▼
         ┌──────────────────┐
         │    Generator     │  Qwen2.5:3b via Ollama
         │  (grounded RAG)  │  streaming SSE
         └────────┬─────────┘
                  │ streamed tokens
                  ▼
         ┌──────────────────┐
         │   FastAPI +      │  UB-branded UI
         │   index.html     │  chunk viewer, latency panel
         └──────────────────┘
```

---

## Bonus features implemented

- ✅ **Cross-encoder reranker** (lexical BM25 + cross-encoder CE)
- ✅ **UI backend panel** showing chunk text, CE/RRF scores, latency breakdown
- ✅ **Knowledge Graph** with 2-hop faculty suggestion from course queries
- ✅ **Personalized memory** — bot asks to personalize, remembers program/interests
- ✅ **Coreference resolution** — "what are his office hours?" resolves correctly
