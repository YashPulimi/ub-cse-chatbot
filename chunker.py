"""
chunker.py — Structure-Aware Semantic Chunking for UB CSE Chatbot
=================================================================
Converts LlamaIndex Documents from ingestion_pipeline.py into
well-formed, metadata-rich chunks (TextNodes) ready for indexing.

Chunking strategy (applied in priority order per page_type):

  1. faculty_profile / faculty_profile_pdf
       → ONE chunk per faculty member, never split
         Reason: splitting faculty profiles breaks the atomic unit
         a retriever needs. "What is Dr. X's email?" needs the
         whole profile in one chunk, not half a bio and half a list
         of papers.

  2. courses / catalog_course / catalog_program
       → ONE chunk per course record (from structured.courses)
         + one header chunk for the page intro
         Reason: "What are the prerequisites for CSE 574?" needs
         exactly the CSE 574 block, not a 2000-word course list.

  3. degree_requirements / catalog_program / admissions / handbook_pdf
       → Section-based: split on ## headings (from heading_path)
         + SentenceSplitter fallback for long sections
         Reason: degree requirement pages have clear H2/H3 sections
         (Admission Requirements, Core Courses, Electives, etc).
         Splitting by section keeps requirement rules intact.

  4. research / general / graduate_programs / undergraduate_programs
       → SemanticSplitterNodeParser using bge-small embeddings
         Reason: research pages are free-form prose — semantic
         boundaries are more meaningful than fixed token windows.

  5. syllabus_pdf / advising_pdf / schedule_pdf / cv_pdf / pdf
       → SentenceSplitter with overlap
         Reason: PDFs lack reliable heading structure; sentence-level
         splitting with overlap is the safe fallback.

  6. Everything else
       → SentenceSplitter with overlap (safe fallback)

WHY SEMANTIC CHUNKING BEATS FIXED-SIZE:
  - Prerequisites: fixed splitting can put "CSE 250 or permission
    of instructor" in a different chunk from the course header —
    semantic splitting keeps the sentence together.
  - Degree requirements: a 3-credit rule for a core course might
    span two fixed chunks; section-based chunking keeps the rule
    with its heading context.
  - Faculty lookup: splitting a faculty bio mid-sentence loses the
    named entity that anchors retrieval.
  - Research/faculty recommendations: semantic boundaries align
    with topic transitions, so "Prof. X works on NLP..." stays
    with the next sentence that names their lab.
  - Follow-up questions: chunk metadata carries the heading_path,
    so memory.py can resolve "that professor" → chunk from same
    heading subtree.

HOW TO RUN:
  python chunker.py
  # Input  → data/processed/processed_docs.jsonl
  # Output → data/processed/chunks.jsonl
  # Next   → python vector_store.py
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
)
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import cfg
from ingestion_pipeline import IngestionPipeline
from utils import get_logger, extract_course_codes, save_jsonl, word_count

log = get_logger(__name__)

# ── Page-type routing ─────────────────────────────────────────────────────────

# Page types that get one-chunk-per-entity treatment
FACULTY_TYPES = {"faculty_profile", "faculty_profile_pdf", "faculty_website"}

COURSE_TYPES  = {"courses", "catalog_course", "undergraduate_courses",
                 "graduate_courses"}

# Page types that are section-split on ## headings first
SECTION_TYPES = {
    "degree_requirements", "catalog_program", "admissions",
    "handbook_pdf", "degree_pdf", "advising_pdf",
    "graduate_programs", "undergraduate_programs", "hub_curriculum",
}

# Page types that use semantic splitter
SEMANTIC_TYPES = {"research", "general", "registrar", "catalog_department"}

# PDF types that get sentence splitting
PDF_TYPES = {"syllabus_pdf", "schedule_pdf", "cv_pdf", "pdf"}


# ── Splitter cache (singletons — expensive to init) ───────────────────────────

_sentence_splitter: SentenceSplitter | None = None
_semantic_splitter: SemanticSplitterNodeParser | None = None


def _get_sentence_splitter() -> SentenceSplitter:
    global _sentence_splitter
    if _sentence_splitter is None:
        _sentence_splitter = SentenceSplitter(
            chunk_size=cfg.chunker.chunk_size,
            chunk_overlap=cfg.chunker.chunk_overlap,
            paragraph_separator="\n\n",
        )
    return _sentence_splitter


def _get_semantic_splitter() -> SemanticSplitterNodeParser:
    """
    SemanticSplitter uses bge-small embeddings to find topic boundaries.
    Falls back to SentenceSplitter if model fails to load.
    """
    global _semantic_splitter
    if _semantic_splitter is None:
        try:
            embed = HuggingFaceEmbedding(
                model_name=cfg.embedding.model,
                device=cfg.embedding.device,
            )
            _semantic_splitter = SemanticSplitterNodeParser(
                embed_model=embed,
                buffer_size=cfg.chunker.semantic_buffer,
                breakpoint_percentile_threshold=95,
            )
            log.info("SemanticSplitter loaded: %s", cfg.embedding.model)
        except Exception as e:
            log.warning("SemanticSplitter failed to load (%s) — using SentenceSplitter", e)
            _semantic_splitter = None  # type: ignore[assignment]
    return _semantic_splitter  # type: ignore[return-value]


# ── Metadata helpers ──────────────────────────────────────────────────────────

def _base_metadata(doc: Document, extra: dict | None = None) -> dict:
    """
    Copy all document-level metadata into the chunk.
    Every chunk carries the full source context so the UI can cite it.
    """
    meta = {k: v for k, v in doc.metadata.items()}
    if extra:
        meta.update(extra)
    return meta


def _make_node(
    text: str,
    doc: Document,
    chunk_index: int = 0,
    heading: str = "",
    extra_meta: dict | None = None,
) -> TextNode | None:
    """Create a TextNode with full metadata. Returns None if text is too short."""
    text = text.strip()
    if word_count(text) < cfg.chunker.min_chunk_words:
        return None

    meta = _base_metadata(doc, extra_meta)
    meta["chunk_index"]   = chunk_index
    meta["heading"]       = heading
    meta["chunk_words"]   = word_count(text)
    # Freshen course codes for this specific chunk text
    meta["courses_in_chunk"] = json.dumps(extract_course_codes(text))

    return TextNode(
        text=text,
        metadata=meta,
        id_=str(uuid.uuid4()),
    )



# ── Minimal-metadata document factory ────────────────────────────────────────

# Fields that LlamaIndex stores in the chunk token budget alongside text.
# All other metadata is re-attached AFTER splitting via _make_node().
_SPLITTER_META_KEYS = {"url", "title", "page_type"}


def _slim_doc(doc: Document, text: str | None = None) -> Document:
    """
    Return a Document with only the fields LlamaIndex needs for splitting.
    LlamaIndex serialises ALL metadata into the chunk and counts it against
    the chunk_size budget — a 1 KB metadata dict leaves only ~300 tokens
    for actual text in a 512-token window.  We pass a slim doc to the splitter
    and re-attach full metadata in _make_node().
    """
    slim_meta = {k: doc.metadata.get(k, "") for k in _SPLITTER_META_KEYS}
    return Document(text=text or doc.text, metadata=slim_meta)

# ── Strategy 1: one chunk per faculty ────────────────────────────────────────

def _chunk_faculty(doc: Document) -> list[TextNode]:
    """
    Faculty profiles are never split — the entire document is one chunk.
    This ensures retrieval always returns the complete profile including
    email, office, rank, research areas, and bio in a single context window.
    """
    node = _make_node(
        text=doc.text,
        doc=doc,
        chunk_index=0,
        heading=doc.metadata.get("title", ""),
        extra_meta={"chunk_strategy": "faculty_atomic"},
    )
    return [node] if node else []


# ── Strategy 2: one chunk per course ─────────────────────────────────────────

def _chunk_courses(doc: Document) -> list[TextNode]:
    """
    Extract individual course records from structured metadata first.
    Each course (code + title + description + prereqs) becomes its own chunk.
    Remaining text (page intro, notes) gets sentence-split as a page chunk.
    """
    nodes: list[TextNode] = []

    # Try structured course records first
    course_codes_str = doc.metadata.get("course_codes", "")
    structured_courses = _recover_courses_from_metadata(doc)

    if structured_courses:
        for i, course in enumerate(structured_courses):
            code  = course.get("code", "")
            title = course.get("title", "")
            desc  = course.get("description", "")
            prereqs = course.get("prereqs", [])
            credits = course.get("credits", "3")

            # Build a self-contained course chunk
            text = f"{code} — {title}\nCredits: {credits}\n"
            if prereqs:
                text += f"Prerequisites: {', '.join(prereqs)}\n"
            if desc:
                text += f"\n{desc}"

            node = _make_node(
                text=text,
                doc=doc,
                chunk_index=i,
                heading=f"{code}: {title}",
                extra_meta={
                    "chunk_strategy": "course_atomic",
                    "course_code": code,
                    "course_title": title,
                    "course_credits": credits,
                    "course_prereqs": json.dumps(prereqs),
                },
            )
            if node:
                nodes.append(node)

    # Fallback: no structured courses → sentence-split the full text
    if not nodes:
        nodes = _chunk_sentence(doc)

    return nodes


def _recover_courses_from_metadata(doc: Document) -> list[dict]:
    """
    Recover course records from the metadata that ingestion_pipeline stored.
    The ingestion pipeline stores course_codes as a JSON list of codes.
    We reconstruct minimal course dicts from the page text using those codes.
    """
    # Try to parse from raw structured JSON if available
    raw = doc.metadata.get("_structured_courses")
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass

    # Fall back to extracting from text using course code positions
    text = doc.text
    codes = extract_course_codes(text)
    if not codes:
        return []

    courses = []
    _COURSE_BLOCK = re.compile(
        r"\b(CSE\s*\d{3}[A-Za-z]?)\b[:\s\-–—LEC]+([A-Z][A-Za-z ,&:()/]{4,80}?)"
        r"(?:\s*Credits?:\s*(\d))?",
        re.MULTILINE,
    )
    _PREREQ = re.compile(r"pre-?requisites?[:\s]+([^\n]{5,200})", re.I)
    _CODE   = re.compile(r"\b(CSE\s*\d{3}[A-Za-z]?)\b", re.I)

    seen: set[str] = set()
    for m in _COURSE_BLOCK.finditer(text):
        code = "CSE " + re.sub(r"\s+", "", m.group(1))[3:].upper()
        if code in seen:
            continue
        seen.add(code)
        start = m.end()
        nxt   = _CODE.search(text, start)
        desc  = text[start: nxt.start() if nxt else start + 500].strip()
        desc  = re.sub(r"\s+", " ", desc)[:400]
        prereqs: list[str] = []
        pm = _PREREQ.search(desc)
        if pm:
            prereqs = [
                "CSE " + re.sub(r"\s+", "", x)[3:].upper()
                for x in _CODE.findall(pm.group(1))
            ]
        courses.append({
            "code":        code,
            "title":       m.group(2).strip().rstrip(",;:."),
            "credits":     m.group(3) or "3",
            "description": desc,
            "prereqs":     prereqs,
        })
    return courses


# ── Strategy 3: section-based on ## headings ──────────────────────────────────

def _chunk_sections(doc: Document) -> list[TextNode]:
    """
    Split on ## headings from the structured text.
    Each H2 section becomes its own chunk. Long sections are further
    split by SentenceSplitter to stay within the token budget.
    This keeps degree rules, admission requirements, and policies intact.
    """
    text = doc.text
    # Split on the ## markers produced by _structured_text in crawler
    sections = re.split(r"\n##\s+", text)

    nodes: list[TextNode] = []
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue

        # First line after split is the heading (for sections 1+)
        lines    = section.split("\n", 1)
        heading  = lines[0].strip() if i > 0 else doc.metadata.get("title", "")
        body     = lines[1].strip() if len(lines) > 1 else section

        # Prepend heading back into body so context is self-contained
        full_text = f"## {heading}\n\n{body}" if heading else body

        # If the section is short enough, keep it whole
        if word_count(full_text) <= cfg.chunker.chunk_size:
            node = _make_node(
                text=full_text,
                doc=doc,
                chunk_index=i,
                heading=heading,
                extra_meta={"chunk_strategy": "section"},
            )
            if node:
                nodes.append(node)
        else:
            # Long section: sentence-split but prepend heading to every sub-chunk
            splitter = _get_sentence_splitter()
            sub_doc  = _slim_doc(doc, text=full_text)
            sub_nodes = splitter.get_nodes_from_documents([sub_doc])
            for j, sn in enumerate(sub_nodes):
                node = _make_node(
                    text=sn.text,
                    doc=doc,
                    chunk_index=i * 100 + j,
                    heading=heading,
                    extra_meta={"chunk_strategy": "section_split"},
                )
                if node:
                    nodes.append(node)

    # Fallback if no ## headings found
    if not nodes:
        nodes = _chunk_sentence(doc)

    return nodes


# ── Strategy 4: semantic splitting ────────────────────────────────────────────

def _chunk_semantic(doc: Document) -> list[TextNode]:
    """
    Use SemanticSplitterNodeParser to find topic boundaries using embeddings.
    Falls back to SentenceSplitter if the model is unavailable or text is short.
    Best for free-form research pages and general content.
    """
    splitter = _get_semantic_splitter()

    # Short docs: not worth running embeddings — use sentence splitter
    if splitter is None or word_count(doc.text) < 100:
        return _chunk_sentence(doc)

    try:
        raw_nodes = splitter.get_nodes_from_documents([_slim_doc(doc)])
        nodes: list[TextNode] = []
        heading_path = json.loads(doc.metadata.get("heading_path", "[]"))
        for i, rn in enumerate(raw_nodes):
            # Infer the closest heading for this chunk position
            heading = heading_path[min(i, len(heading_path) - 1)] if heading_path else ""
            node = _make_node(
                text=rn.text,
                doc=doc,
                chunk_index=i,
                heading=heading,
                extra_meta={"chunk_strategy": "semantic"},
            )
            if node:
                nodes.append(node)
        return nodes if nodes else _chunk_sentence(doc)
    except Exception as e:
        log.warning("SemanticSplitter failed on %s: %s — fallback", doc.metadata.get("url", "?"), e)
        return _chunk_sentence(doc)


# ── Strategy 5: sentence splitting (fallback) ─────────────────────────────────

def _chunk_sentence(doc: Document) -> list[TextNode]:
    """
    Standard SentenceSplitter — used as fallback and for PDFs.
    Preserves sentence boundaries, overlaps for context continuity.
    """
    splitter  = _get_sentence_splitter()
    raw_nodes = splitter.get_nodes_from_documents([_slim_doc(doc)])
    heading_path = json.loads(doc.metadata.get("heading_path", "[]"))

    nodes: list[TextNode] = []
    for i, rn in enumerate(raw_nodes):
        heading = heading_path[min(i, len(heading_path) - 1)] if heading_path else ""
        node = _make_node(
            text=rn.text,
            doc=doc,
            chunk_index=i,
            heading=heading,
            extra_meta={"chunk_strategy": "sentence"},
        )
        if node:
            nodes.append(node)
    return nodes


# ── Router ────────────────────────────────────────────────────────────────────

def chunk_document(doc: Document) -> list[TextNode]:
    """
    Route a Document to the right chunking strategy based on page_type.
    All strategies preserve full metadata on every output chunk.
    """
    ptype = doc.metadata.get("page_type", "general")

    if ptype in FACULTY_TYPES:
        return _chunk_faculty(doc)

    if ptype in COURSE_TYPES:
        return _chunk_courses(doc)

    if ptype in SECTION_TYPES:
        return _chunk_sections(doc)

    if ptype in SEMANTIC_TYPES:
        return _chunk_semantic(doc)

    if ptype in PDF_TYPES:
        return _chunk_sentence(doc)

    # Default fallback
    return _chunk_sentence(doc)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class ChunkingPipeline:

    def __init__(self) -> None:
        self.chunks: list[TextNode] = []

    def run(self, documents: list[Document]) -> list[TextNode]:
        t0 = time.perf_counter()
        by_strategy: dict[str, int] = {}
        by_type:     dict[str, int] = {}

        for doc in documents:
            ptype  = doc.metadata.get("page_type", "general")
            nodes  = chunk_document(doc)

            for node in nodes:
                strategy = node.metadata.get("chunk_strategy", "unknown")
                by_strategy[strategy] = by_strategy.get(strategy, 0) + 1
                by_type[ptype]        = by_type.get(ptype, 0) + 1

            self.chunks.extend(nodes)

        elapsed = (time.perf_counter() - t0) * 1000
        log.info(
            "Chunking complete: %d docs → %d chunks in %.0f ms",
            len(documents), len(self.chunks), elapsed,
        )
        log.info("By strategy: %s", by_strategy)
        log.info("By page type: %s", by_type)
        return self.chunks

    def save(self) -> Path:
        out = cfg.paths.processed / "chunks.jsonl"
        records = [
            {
                "id":       node.id_,
                "text":     node.text,
                "metadata": node.metadata,
            }
            for node in self.chunks
        ]
        save_jsonl(records, out)

        summary = {
            "total_chunks":      len(self.chunks),
            "total_words":       sum(node.metadata.get("chunk_words", 0) for node in self.chunks),
            "avg_chunk_words":   (
                sum(node.metadata.get("chunk_words", 0) for node in self.chunks) // max(len(self.chunks), 1)
            ),
            "by_strategy": {},
            "by_page_type": {},
        }
        for node in self.chunks:
            s = node.metadata.get("chunk_strategy", "unknown")
            p = node.metadata.get("page_type",       "unknown")
            summary["by_strategy"][s] = summary["by_strategy"].get(s, 0) + 1
            summary["by_page_type"][p] = summary["by_page_type"].get(p, 0) + 1

        (cfg.paths.processed / "chunks_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        log.info("Saved %d chunks → %s", len(self.chunks), out)
        return out

    @staticmethod
    def load_chunks(path: Path | None = None) -> list[TextNode]:
        """Re-load TextNodes from chunks.jsonl. Used by vector_store.py."""
        from utils import load_jsonl
        p = path or (cfg.paths.processed / "chunks.jsonl")
        if not p.exists():
            log.error("chunks.jsonl not found — run chunker.py first")
            return []
        records = load_jsonl(p)
        nodes = []
        for r in records:
            try:
                nodes.append(TextNode(text=r["text"], metadata=r["metadata"], id_=r["id"]))
            except Exception as e:
                log.warning("Skipping malformed chunk: %s", e)
        log.info("Loaded %d chunks from %s", len(nodes), p.name)
        return nodes


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    docs = IngestionPipeline.load_documents()
    if not docs:
        print("No documents found. Run ingestion_pipeline.py first.")
        return

    pipeline = ChunkingPipeline()
    pipeline.run(docs)
    out = pipeline.save()
    print(f"\n✅  Chunks → {out}")
    print(f"    Next   → python vector_store.py")


if __name__ == "__main__":
    main()