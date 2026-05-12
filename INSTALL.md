# Install Guide — UB CSE Chatbot

## Requirements
- Python 3.11 or higher
- 8 GB RAM minimum (16 GB recommended for 7B model)
- ~10 GB disk space (models + indexes + crawl data)

---

## Step 1 — Python environment

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Upgrade pip first
pip install --upgrade pip
```

---

## Step 2 — Install all Python dependencies

```bash
pip install -r requirements.txt
```

This installs everything in one shot:
- Crawl4AI (async web crawler)
- LlamaIndex core + all integrations (Qdrant, Neo4j, BM25, Ollama, HuggingFace)
- Qdrant client
- sentence-transformers (embeddings + cross-encoder reranker)
- rank-bm25
- Neo4j driver
- pypdf + pdfplumber (PDF extraction)
- RAGAS (evaluation)
- Chainlit (UI)

---

## Step 3 — Set up Crawl4AI (downloads Playwright browsers)

```bash
crawl4ai-setup
# This downloads Chromium (~150 MB) used for dynamic page rendering
```

---

## Step 4 — Copy and edit your .env

```bash
cp .env.example .env
# Edit .env if you want to change models, ports, or paths
# Defaults work out of the box for local development
```

---

## Step 5 — Install and start Ollama (local LLM)

```bash
# Install Ollama from https://ollama.com
# Then pull the model:
ollama pull qwen2.5:3b          # fast, low RAM (~2 GB)
# OR for better quality:
ollama pull qwen2.5:7b          # needs ~5 GB RAM

# Ollama runs automatically in the background after install
# Verify:
ollama list
```

---

## Step 6 — Start Neo4j (graph store)

```bash
# Option A: Docker (recommended)
docker compose up -d neo4j
# Neo4j browser: http://localhost:7474  (user: neo4j / pass: ubcse2025)

# Option B: Neo4j Desktop
# Download from https://neo4j.com/download/
# Create a database with password: ubcse2025
```

---

## Step 7 — Qdrant (vector store — no setup needed for local mode)

By default the chatbot uses **Qdrant local on-disk mode** — no Docker, no server needed.
The database is stored at `data/indexes/qdrant_db/`.

If you want a Qdrant server instead:
```bash
docker compose up -d qdrant
# Then set in .env:  QDRANT_HOST=localhost
```

---

## Step 8 — Verify everything works

```bash
python -c "
from config import cfg
print('config OK:', cfg.llm.model)

from llama_index.core import Document
print('llama_index OK')

from qdrant_client import QdrantClient
print('qdrant_client OK')

from sentence_transformers import CrossEncoder
print('sentence_transformers OK')
"
```

---

## Run the full pipeline

```bash
# 1. Crawl UB CSE pages + PDFs
python crawler.py

# 2. Convert to LlamaIndex Documents
python ingestion_pipeline.py

# 3. Chunk documents
python chunker.py

# 4. Build embeddings + Qdrant index
python vector_store.py

# 5. Build BM25 index
python sparse_retriever.py

# 6. Build Neo4j knowledge graph
python graph_store.py

# 7. Launch chatbot UI
chainlit run app.py

# 8. Run evaluation
python evaluator.py
```

---

## Troubleshooting

**`crawl4ai-setup` fails**
```bash
playwright install chromium
```

**Neo4j connection refused**
```bash
docker compose logs neo4j    # check if container started
# Or change NEO4J_URI in .env to match your setup
```

**Out of memory during embedding**
```bash
# In .env, reduce batch size:
EMBED_BATCH_SIZE=16
# Or switch to smaller model:
EMBED_MODEL=BAAI/bge-small-en-v1.5
EMBED_DIM=384
```

**Ollama model not found**
```bash
ollama pull qwen2.5:3b
ollama list    # verify it appears
```

**`torch` install is slow / large**
```bash
# CPU-only torch (smaller, faster to install):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
