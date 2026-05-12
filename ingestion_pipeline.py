"""
ingestion_pipeline.py — Crawl Output → LlamaIndex Documents
============================================================
Reads the crawler's JSONL output and converts every page + every
downloaded PDF into standard LlamaIndex Document objects with full
metadata.  This is the standardization boundary: everything downstream
(chunker, vector store, graph builder) works with LlamaIndex Documents
and never reads raw crawler JSONL again.

Responsibilities:
  1. Load latest crawl_*.jsonl from data/raw/
  2. Convert each Page record → LlamaIndex Document  (HTML pages)
  3. Extract text from downloaded PDFs → LlamaIndex Document  (PDFs)
  4. Normalize metadata for every document
  5. Deduplicate by content_hash across pages + PDFs
  6. Save processed Documents as processed_docs.jsonl for downstream use

Why LlamaIndex Documents?
  - Standard interface used by chunker, Qdrant vector store, BM25, graph builder
  - Carries metadata dict that flows through to every chunk
  - SimpleDirectoryReader / PDFReader handle PDF text extraction
  - No custom loaders needed for standard page types

HOW TO RUN:
  python ingestion_pipeline.py
  # Output → data/processed/processed_docs.jsonl
  # Next   → python chunker.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterator

from llama_index.core import Document
from llama_index.core.readers import SimpleDirectoryReader

from config import cfg
from utils import get_logger, content_hash, save_jsonl, load_jsonl

log = get_logger(__name__)

# ── Metadata keys we preserve on every Document ───────────────────────────────
# These flow through chunking → Qdrant payload → retrieval → UI citations.
# Keep this list stable — adding keys later requires reindexing.

METADATA_KEYS = [
    "url",
    "title",
    "page_type",
    "source",                  # department | catalog | faculty_website | hub | pdf
    "last_modified",
    "crawled_at",
    "content_hash",
    "word_count",
    "depth",
    "heading_path",            # list[str] — ordered ## headings, for chunker
    "courses_mentioned",       # list[str] — normalized CSE codes, for GraphRAG
    "faculty_names",           # list[str] — for GraphRAG
    "pdf_paths",               # list[str] — local PDF file paths linked from page
    # Structured sub-fields (flattened for Qdrant payload compatibility)
    "faculty_email",
    "faculty_rank",
    "faculty_office",
    "faculty_phone",
    "faculty_research_areas",  # list[str]
    "faculty_personal_websites",
    "course_codes",            # list[str] — codes extracted from structured.courses
]


# ── Page → Document ───────────────────────────────────────────────────────────

def _flatten_structured(structured: dict) -> dict:
    """
    Flatten the Page.structured dict into top-level metadata keys
    so Qdrant can filter on them directly (no nested dicts in payload).
    """
    out: dict = {}

    # Faculty profile fields
    if "email" in structured:
        out["faculty_email"] = structured["email"]
    if "rank" in structured:
        out["faculty_rank"] = structured["rank"]
    if "office" in structured:
        out["faculty_office"] = structured["office"]
    if "phone" in structured:
        out["faculty_phone"] = structured["phone"]
    if "research_areas" in structured:
        out["faculty_research_areas"] = structured["research_areas"]
    if "personal_websites" in structured:
        out["faculty_personal_websites"] = structured["personal_websites"]

    # Course page fields — store just the codes for quick filtering
    if "courses" in structured:
        out["course_codes"] = [c["code"] for c in structured["courses"] if "code" in c]

    return out


def page_to_document(record: dict) -> Document | None:
    """
    Convert a single crawler JSONL record (Page.asdict()) into a LlamaIndex Document.
    Returns None if the record has no usable text.
    """
    text = (record.get("content") or "").strip()
    if not text or len(text.split()) < cfg.chunker.min_chunk_words:
        return None

    structured = record.get("structured") or {}

    metadata: dict = {
        "url":              record.get("url", ""),
        "title":            record.get("title", ""),
        "page_type":        record.get("page_type", "general"),
        "source":           record.get("source", "department"),
        "last_modified":    record.get("last_modified", ""),
        "crawled_at":       record.get("crawled_at", ""),
        "content_hash":     record.get("content_hash", content_hash(text)),
        "word_count":       record.get("word_count", len(text.split())),
        "depth":            record.get("depth", 0),
        # list fields — stored as JSON strings for Qdrant payload compatibility
        "heading_path":     json.dumps(record.get("heading_path") or []),
        "courses_mentioned":json.dumps(record.get("courses_mentioned") or []),
        "faculty_names":    json.dumps(record.get("faculty_names") or []),
        "pdf_paths":        json.dumps(record.get("pdf_paths") or []),
        **_flatten_structured(structured),
    }

    # doc_id = content_hash so chunker can deduplicate across re-runs
    return Document(
        text=text,
        metadata=metadata,
        id_=metadata["content_hash"],
    )


# ── PDF → Document ────────────────────────────────────────────────────────────

def _pdf_to_document(pdf_path: Path, source_url: str = "") -> Document | None:
    """
    Extract text from a single PDF using LlamaIndex SimpleDirectoryReader.
    Falls back to empty string on failure (corrupt / image-only PDFs).
    """
    try:
        reader = SimpleDirectoryReader(input_files=[str(pdf_path)])
        docs   = reader.load_data()
        if not docs:
            return None

        # Concatenate all pages into one document (most PDFs are short)
        full_text = "\n\n".join(d.text for d in docs if d.text.strip())
        if not full_text or len(full_text.split()) < cfg.chunker.min_chunk_words:
            return None

        ch = content_hash(full_text)
        metadata: dict = {
            "url":               source_url or f"file://{pdf_path}",
            "title":             pdf_path.stem.replace("_", " ").replace("-", " "),
            "page_type":         _infer_pdf_type(pdf_path.name),
            "source":            "pdf",
            "last_modified":     datetime.utcfromtimestamp(
                                     pdf_path.stat().st_mtime
                                 ).isoformat() if pdf_path.exists() else "",
            "crawled_at":        datetime.utcnow().isoformat(),
            "content_hash":      ch,
            "word_count":        len(full_text.split()),
            "depth":             0,
            "heading_path":      json.dumps([]),
            "courses_mentioned": json.dumps([]),
            "faculty_names":     json.dumps([]),
            "pdf_paths":         json.dumps([str(pdf_path)]),
        }
        return Document(text=full_text, metadata=metadata, id_=ch)

    except Exception as e:
        log.warning("PDF extraction failed %s: %s", pdf_path.name, e)
        return None


def _infer_pdf_type(filename: str) -> str:
    """Best-effort page_type label for a PDF based on its filename."""
    name = filename.lower()
    if any(k in name for k in ("syllabus", "syllabi")):
        return "syllabus_pdf"
    if any(k in name for k in ("handbook",)):
        return "handbook_pdf"
    if any(k in name for k in ("checklist", "worksheet", "degree", "program")):
        return "degree_pdf"
    if any(k in name for k in ("advising", "guideline", "guide")):
        return "advising_pdf"
    if any(k in name for k in ("_cv", "-cv", "vitae")):
        return "cv_pdf"
    if any(k in name for k in ("schedule",)):
        return "schedule_pdf"
    return "pdf"


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _latest_crawl_file() -> Path | None:
    """Return the most recently written crawl_*.jsonl file."""
    files = sorted(cfg.paths.raw.glob("crawl_*.jsonl"), reverse=True)
    return files[0] if files else None


def _iter_page_records(jsonl_path: Path) -> Iterator[dict]:
    """Yield raw dicts from a JSONL file, skipping malformed lines."""
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("Skipping malformed JSONL line %d: %s", i, e)


def _collect_pdf_paths(records: list[dict]) -> dict[Path, str]:
    """
    Return {local_pdf_path → source_page_url} for all PDFs referenced
    in crawled pages that actually exist on disk.
    """
    result: dict[Path, str] = {}
    for rec in records:
        source_url = rec.get("url", "")
        for path_str in (rec.get("pdf_paths") or []):
            p = Path(path_str)
            if p.exists() and p not in result:
                result[p] = source_url
    return result


class IngestionPipeline:
    """
    Converts crawler output into LlamaIndex Documents.
    Deduplicates by content_hash — safe to re-run after incremental crawls.
    """

    def __init__(self) -> None:
        self.documents:     list[Document] = []
        self._seen_hashes:  set[str]       = set()

    def _add(self, doc: Document | None) -> bool:
        if doc is None:
            return False
        ch = doc.metadata.get("content_hash", doc.id_)
        if ch in self._seen_hashes:
            return False
        self._seen_hashes.add(ch)
        self.documents.append(doc)
        return True

    def load_pages(self, jsonl_path: Path) -> int:
        """Load HTML page records from crawler JSONL. Returns count added."""
        added = 0
        total = 0
        for record in _iter_page_records(jsonl_path):
            total += 1
            doc = page_to_document(record)
            if self._add(doc):
                added += 1
        log.info("Pages: %d loaded, %d accepted (deduped %d)", total, added, total - added)
        return added

    def load_pdfs(self, pdf_url_map: dict[Path, str]) -> int:
        """Extract text from PDF files and add as Documents. Returns count added."""
        added = 0
        for pdf_path, source_url in pdf_url_map.items():
            doc = _pdf_to_document(pdf_path, source_url)
            if self._add(doc):
                added += 1
                log.debug("PDF ✓ %s (%d words)", pdf_path.name, doc.metadata.get("word_count", 0))
        log.info("PDFs: %d files processed, %d accepted", len(pdf_url_map), added)
        return added

    def run(self) -> list[Document]:
        """Full pipeline: find latest crawl → load pages → load PDFs → return Documents."""
        jsonl_path = _latest_crawl_file()
        if jsonl_path is None:
            log.error("No crawl_*.jsonl found in %s — run crawler.py first", cfg.paths.raw)
            return []

        log.info("Loading crawl file: %s", jsonl_path.name)

        # Load all raw records first (needed to find PDF paths)
        raw_records = list(_iter_page_records(jsonl_path))

        # HTML pages
        pages_added = 0
        for record in raw_records:
            doc = page_to_document(record)
            if self._add(doc):
                pages_added += 1
        log.info("Pages accepted: %d", pages_added)

        # PDFs — discover from pdf_paths fields across all pages
        pdf_map = _collect_pdf_paths(raw_records)
        if pdf_map:
            self.load_pdfs(pdf_map)
        else:
            # Also scan the pdfs/ directory directly as fallback
            all_pdfs = list(cfg.paths.pdfs.glob("*.pdf"))
            if all_pdfs:
                log.info("No pdf_paths in JSONL — scanning %s directly (%d files)",
                         cfg.paths.pdfs, len(all_pdfs))
                self.load_pdfs({p: "" for p in all_pdfs})

        log.info(
            "Ingestion complete: %d total documents  (%d page types)",
            len(self.documents),
            len({d.metadata.get("page_type") for d in self.documents}),
        )
        return self.documents

    def save(self) -> Path:
        """Save Documents as JSONL for chunker. Returns output path."""
        out = cfg.paths.processed / "processed_docs.jsonl"
        records = [
            {
                "id":       doc.id_,
                "text":     doc.text,
                "metadata": doc.metadata,
            }
            for doc in self.documents
        ]
        save_jsonl(records, out)

        # Summary
        by_type: dict[str, int] = {}
        for doc in self.documents:
            pt = doc.metadata.get("page_type", "unknown")
            by_type[pt] = by_type.get(pt, 0) + 1

        summary = {
            "total_documents": len(self.documents),
            "total_words":     sum(doc.metadata.get("word_count", 0) for doc in self.documents),
            "by_page_type":    by_type,
            "created_at":      datetime.utcnow().isoformat(),
        }
        (cfg.paths.processed / "ingestion_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        log.info("Saved %d documents → %s", len(self.documents), out)
        log.info("By type: %s", by_type)
        return out

    @staticmethod
    def load_documents(path: Path | None = None) -> list[Document]:
        """
        Re-load Documents from processed_docs.jsonl.
        Called by chunker.py and graph_builder.py instead of re-running ingestion.
        """
        p = path or (cfg.paths.processed / "processed_docs.jsonl")
        if not p.exists():
            log.error("processed_docs.jsonl not found — run ingestion_pipeline.py first")
            return []
        records = load_jsonl(p)
        docs = []
        for r in records:
            try:
                docs.append(Document(text=r["text"], metadata=r["metadata"], id_=r["id"]))
            except Exception as e:
                log.warning("Skipping malformed record: %s", e)
        log.info("Loaded %d documents from %s", len(docs), p.name)
        return docs


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    pipeline = IngestionPipeline()
    pipeline.run()
    out = pipeline.save()
    print(f"\n✅  Documents → {out}")
    print(f"    Next      → python chunker.py")


if __name__ == "__main__":
    main()
