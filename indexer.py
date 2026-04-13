"""
indexer.py — Chunk + Embed + Store
====================================
Reads:  data/raw/crawl_*.jsonl   (web pages)
        data/raw/pdfs/*.pdf      (downloaded PDFs)
Writes: data/chroma/             (ChromaDB vector store)

Pipeline:
  1. Load web pages from JSONL
  2. Extract text from PDFs (pymupdf)
  3. Chunk with LangChain RecursiveCharacterTextSplitter
  4. Embed with Ollama nomic-embed-text (local, free)
  5. Store in ChromaDB with metadata

HOW TO RUN:
  python indexer.py              # uses latest crawl
  python indexer.py --force      # re-index from scratch
  python indexer.py --no-pdf     # skip PDFs

INTERVIEW NOTE — Why character-based chunking?
  Embedding models have token limits (~2048 tokens).
  Characters are predictable — 1200 chars ≈ 300 tokens, safely within limits.
  Word-based chunking can produce variable-length chunks that exceed
  model context windows on dense technical text.

INTERVIEW NOTE — Why nomic-embed-text?
  Specifically trained for retrieval (MTEB benchmark).
  Outperforms all-MiniLM, approaches OpenAI ada-002 quality.
  Runs locally on M4 Mac via Metal GPU. Zero cost, zero latency variance.
"""

import argparse
import hashlib
import json
import logging
import subprocess
from pathlib import Path

import chromadb
import fitz  # pymupdf
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("indexer")

# ── Config ────────────────────────────────────────────────────────────────────

RAW_DIR    = Path("data/raw")
PDF_DIR    = Path("data/raw/pdfs")
CHROMA_DIR = Path("data/chroma")
COLLECTION = "ub_cse"

CHUNK_SIZE    = 1200   # characters — safe within nomic-embed-text context window
CHUNK_OVERLAP = 120    # 10% overlap to preserve context at boundaries
MIN_CHUNK_LEN = 100    # skip very short chunks

EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE  = 50       # chunks per ChromaDB upsert

CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# ── PDF extraction ────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from PDF using pymupdf."""
    try:
        doc  = fitz.open(str(pdf_path))
        text = "\n".join(page.get_text("text") for page in doc)
        doc.close()
        return text.strip()
    except Exception as e:
        log.error(f"PDF failed {pdf_path.name}: {e}")
        return ""


def load_pdf_documents(pdf_dir: Path) -> list[dict]:
    """Load all PDFs as document dicts."""
    docs      = []
    pdf_files = list(pdf_dir.glob("*.pdf"))
    log.info(f"Found {len(pdf_files)} PDFs")

    for pdf_path in pdf_files:
        text = extract_pdf_text(pdf_path)
        if not text or len(text) < MIN_CHUNK_LEN:
            continue

        name  = pdf_path.stem.lower()
        ptype = "pdf_document"
        if "handbook" in name:
            ptype = "graduate_handbook"
        elif any(x in name for x in ["syllabus", "cse_5", "cse_6", "cse-5", "cse-6"]):
            ptype = "syllabus"
        elif "checklist" in name:
            ptype = "degree_requirements"
        elif "_cv" in name or "-cv" in name:
            ptype = "faculty_cv"

        docs.append({
            "content":   text,
            "title":     pdf_path.stem.replace("-", " ").replace("_", " ").title(),
            "url":       f"https://engineering.buffalo.edu/{pdf_path.name}",
            "page_type": ptype,
            "source":    "pdf",
            "filename":  pdf_path.name,
        })
        log.info(f"  ✓ {pdf_path.name} ({len(text):,} chars) [{ptype}]")

    log.info(f"Loaded {len(docs)} PDFs with content")
    return docs


# ── Web page loader ───────────────────────────────────────────────────────────

def load_web_pages(crawl_file: Path) -> list[dict]:
    """Load crawled web pages from JSONL."""
    docs = []
    with open(crawl_file) as f:
        for line in f:
            page = json.loads(line.strip())
            if not page.get("content") or page.get("word_count", 0) < 30:
                continue
            docs.append({
                "content":   page["content"],
                "title":     page.get("title", ""),
                "url":       page["url"],
                "page_type": page.get("page_type", "general"),
                "source":    page.get("source", "web"),
                "filename":  "",
            })
    log.info(f"Loaded {len(docs)} web pages")
    return docs


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_documents(docs: list[dict]) -> list[dict]:
    """
    Split documents into chunks.

    INTERVIEW NOTE — RecursiveCharacterTextSplitter splits in order:
      1. Double newline (paragraph)
      2. Single newline (line break)
      3. Period + space (sentence)
      4. Space (word)
      Never cuts mid-word. Preserves natural text boundaries.
      This is the production standard for RAG chunking.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size      = CHUNK_SIZE,
        chunk_overlap   = CHUNK_OVERLAP,
        length_function = len,
        separators      = ["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        # Prepend title to each chunk — improves retrieval
        # "Who teaches CSE 574?" matches better when chunk starts with
        # "Introduction to Machine Learning\n\nInstructor: Sargur Srihari"
        text = f"{doc['title']}\n\n{doc['content']}" if doc["title"] else doc["content"]

        for i, split in enumerate(splitter.split_text(text)):
            if len(split) < MIN_CHUNK_LEN:
                continue
            chunks.append({
                "text":      split,
                "url":       doc["url"],
                "title":     doc["title"],
                "page_type": doc["page_type"],
                "source":    doc["source"],
                "filename":  doc["filename"],
                "chunk_idx": i,
            })

    # Log distribution
    by_source = {}
    by_type   = {}
    for c in chunks:
        by_source[c["source"]]    = by_source.get(c["source"], 0) + 1
        by_type[c["page_type"]]   = by_type.get(c["page_type"], 0) + 1

    log.info(f"Created {len(chunks)} chunks from {len(docs)} documents")
    log.info(f"By source: {by_source}")
    log.info(f"By type:   {by_type}")
    return chunks


# ── Indexing ──────────────────────────────────────────────────────────────────

def make_id(chunk: dict, idx: int) -> str:
    key = f"{chunk['url']}_{chunk['chunk_idx']}_{idx}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def index_chunks(chunks: list[dict], force: bool = False) -> None:
    """Embed all chunks and store in ChromaDB."""
    log.info(f"Pulling {EMBED_MODEL}...")
    subprocess.run(["ollama", "pull", EMBED_MODEL], capture_output=True)

    embedder = OllamaEmbeddings(model=EMBED_MODEL)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    if force:
        try:
            client.delete_collection(COLLECTION)
            log.info("Deleted existing collection")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name     = COLLECTION,
        metadata = {"hnsw:space": "cosine"},
    )

    log.info(f"Existing chunks: {collection.count()}")

    ids       = [make_id(c, i) for i, c in enumerate(chunks)]
    texts     = [c["text"] for c in chunks]
    metadatas = [{
        "url":       c["url"],
        "title":     c["title"][:200] if c["title"] else "",
        "page_type": c["page_type"],
        "source":    c["source"],
        "filename":  c["filename"],
    } for c in chunks]

    total = len(chunks)
    log.info(f"Embedding {total} chunks...")

    for i in range(0, total, BATCH_SIZE):
        end         = min(i + BATCH_SIZE, total)
        log.info(f"  Batch {i//BATCH_SIZE + 1}/{(total+BATCH_SIZE-1)//BATCH_SIZE} "
                 f"({end}/{total})")
        try:
            vectors = embedder.embed_documents(texts[i:end])
            collection.upsert(
                ids        = ids[i:end],
                embeddings = vectors,
                documents  = texts[i:end],
                metadatas  = metadatas[i:end],
            )
        except Exception as e:
            log.error(f"Batch failed: {e} — skipping")
            continue

    log.info(f"ChromaDB total: {collection.count()} chunks")


# ── Smoke test ────────────────────────────────────────────────────────────────

def smoke_test(queries: list[str]) -> None:
    """Quick retrieval test to verify index quality."""
    embedder   = OllamaEmbeddings(model=EMBED_MODEL)
    client     = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(COLLECTION)

    print(f"\n{'='*60}")
    print("SMOKE TEST")
    print('='*60)

    for query in queries:
        print(f"\nQ: {query}")
        vec     = embedder.embed_query(query)
        results = collection.query(
            query_embeddings = [vec],
            n_results        = 3,
            include          = ["documents", "metadatas", "distances"],
        )
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ), 1):
            sim = round(1 - dist, 3)
            print(f"  [{i}] sim={sim} type={meta['page_type']} src={meta['source']}")
            print(f"       {doc[:150]}...")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crawl",  type=Path, help="Crawl JSONL path")
    parser.add_argument("--force",  action="store_true", help="Re-index from scratch")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDFs")
    args = parser.parse_args()

    # Find latest crawl
    crawl_file = args.crawl
    if not crawl_file:
        candidates = sorted(RAW_DIR.glob("crawl_*.jsonl"), reverse=True)
        if not candidates:
            log.error("No crawl file found. Run crawler.py first.")
            return
        crawl_file = candidates[0]
        log.info(f"Using: {crawl_file}")

    # Load
    web_docs = load_web_pages(crawl_file)
    pdf_docs = [] if args.no_pdf else load_pdf_documents(PDF_DIR)
    all_docs = web_docs + pdf_docs
    log.info(f"Total: {len(all_docs)} docs ({len(web_docs)} web + {len(pdf_docs)} PDF)")

    # Chunk
    chunks = chunk_documents(all_docs)

    # Index
    index_chunks(chunks, force=args.force)

    # Smoke test
    smoke_test([
        "Who teaches CSE 635?",
        "What are the MS program breadth requirements?",
        "What is the application deadline for international students?",
        "What courses are required for the AI focus area?",
    ])

    log.info("\n✅ Indexing complete! Next: python app.py")


if __name__ == "__main__":
    main()
