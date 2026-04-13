# UB CSE Chatbot 🎓

A RAG-based conversational assistant for the University at Buffalo Department of Computer Science and Engineering. Built for CSE 635 — NLP and Text Mining.

## Features

- **Hybrid Retrieval** — BM25 + dense vector search with Reciprocal Rank Fusion
- **Cross-Encoder Reranking** — ms-marco-MiniLM-L-6-v2 for precise result ranking
- **Knowledge Graph** — Neo4j graph of professors, courses, labs, and programs
- **GraphRAG** — structured KG facts + semantic retrieval combined
- **Conversation Memory** — sliding window of last 5 turns
- **Guardrails** — blocks off-topic and harmful queries
- **Local LLM** — runs fully offline via Ollama (no API key needed)
- **Chainlit UI** — clean conversational browser interface

## Architecture

```
User Query
    ↓
Guardrail (keyword classifier)
    ↓
Neo4j KG Lookup (structured facts)     ChromaDB Dense Search
    ↓                                          ↓
                    BM25 Search
                        ↓
                  RRF Fusion (top 20)
                        ↓
              Cross-Encoder Reranking (top 5)
                        ↓
              Ollama LLM + Conversation Memory
                        ↓
                  Chainlit Response
```

## Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed
- [Neo4j](https://neo4j.com) installed (`brew install neo4j`)

### Install

```bash
git clone https://github.com/YOUR_USERNAME/ub-cse-chatbot.git
cd ub-cse-chatbot
pip install -r requirements.txt
crawl4ai-setup
```

### Pull models

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

### Start Neo4j

```bash
brew services start neo4j
# Open http://localhost:7474 and set password to: password123
```

### Run pipeline

```bash
# 1. Crawl data
python crawler.py

# 2. Extract faculty emails
python fetch_faculty.py

# 3. Build knowledge graph
python graph.py

# 4. Index into ChromaDB
python indexer.py

# 5. Launch chatbot
chainlit run app.py
```

Open `http://localhost:8000` in your browser.

## Project Structure

```
├── crawler.py        # Web crawler (department site + catalog + faculty)
├── fetch_faculty.py  # Faculty email extractor (mailto DOM parsing)
├── graph.py          # Neo4j knowledge graph builder
├── indexer.py        # ChromaDB vector indexer
├── app.py            # Chatbot (retrieval + generation + UI)
├── chainlit.md       # Chat welcome screen
└── requirements.txt  # Python dependencies
```

## Stack

| Component | Tool |
|---|---|
| Web crawling | Crawl4AI |
| PDF parsing | PyMuPDF |
| Embeddings | nomic-embed-text (Ollama) |
| Vector store | ChromaDB |
| Keyword search | BM25 (rank-bm25) |
| Reranking | sentence-transformers cross-encoder |
| Knowledge graph | Neo4j |
| LLM | Qwen2.5:3b (Ollama) |
| UI | Chainlit |

## Course

CSE 635 — NLP and Text Mining, Spring 2026  
University at Buffalo
