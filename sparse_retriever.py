"""
sparse_retriever.py — BM25 Sparse Retrieval for UB CSE Chatbot
==============================================================
Complements dense Qdrant search with exact keyword matching using BM25Okapi.

Why BM25 alongside dense retrieval:
  - Dense vectors fail on exact lookups: "CSE 574" and "CSE 574 prerequisites"
    have very similar embeddings to "CSE 500" — the model doesn't know course
    codes are discriminative identifiers.
  - "Who is Chunming Qiao?" — a rare faculty name will score poorly in dense
    space (few training examples) but BM25 matches the exact token.
  - GPA, credit hours, deadline dates — numeric exact matches that embedding
    models treat as semantically similar to any other numbers.
  - BM25 + dense fusion (RRF) consistently outperforms either alone on
    domain-specific academic corpora.

Boosting:
  When a query contains a CSE course code (CSE 574, CSE 115) or a known
  faculty name pattern, results containing that exact token in the text
  are given a score multiplier. This is the one place where a small custom
  rule is justified — it's deterministic normalization, not ML.

Index:
  - Built from chunks.jsonl once, saved to data/indexes/bm25.pkl
  - Tokenization: lowercase, split on whitespace + punctuation, filter
    stopwords to improve precision for academic domain terms
  - Reload is fast (pickle): ~50ms for 2000 chunks

HOW TO RUN:
  python sparse_retriever.py                          # build index
  python sparse_retriever.py --search "CSE 574"       # test query
  python sparse_retriever.py --search "Chunming Qiao" # faculty test
  python sparse_retriever.py --rebuild                # force rebuild
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from pathlib import Path

from rank_bm25 import BM25Okapi

from chunker import ChunkingPipeline
from config import cfg
from utils import get_logger, extract_course_codes

log = get_logger(__name__)

# ── Stopwords (minimal — keep academic terms) ─────────────────────────────────
# Standard English stopwords minus terms that matter in academic queries
# ("all", "no", "not" can be discriminative for requirement queries)
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "we", "you", "i", "me", "him", "her", "them", "us", "my", "your",
    "his", "our", "their", "which", "who", "what", "where", "when",
    "how", "as", "if", "so", "than", "more", "also", "about",
}

# ── Tokenizer ─────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_COURSE_NORM_RE = re.compile(r"\b(CSE)\s*(\d{3}[A-Za-z]?)\b", re.I)


def _normalize_course_tokens(text: str) -> str:
    """Normalize 'CSE574' → 'CSE 574' before tokenization so BM25 treats them identically."""
    return _COURSE_NORM_RE.sub(lambda m: f"CSE {m.group(2).upper()}", text)


def tokenize(text: str) -> list[str]:
    """
    Lowercase, normalize course codes, split on word boundaries,
    filter stopwords and single chars.
    Keeps: course codes, numbers, acronyms, domain terms.
    """
    text = _normalize_course_tokens(text.lower())
    tokens = _TOKEN_RE.findall(text)
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


# ── BM25 Index ────────────────────────────────────────────────────────────────

class BM25Index:
    """
    Wraps BM25Okapi with:
    - Persistent save/load (pickle)
    - Course-code and faculty-name exact-match score boosting
    - Returns scored result dicts compatible with retriever.py RRF fusion
    """

    def __init__(self) -> None:
        self._bm25:   BM25Okapi | None = None
        self._chunks: list[dict] = []   # [{id, text, metadata}, ...]
        self._corpus: list[list[str]] = []

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, chunks: list) -> None:
        """
        Build BM25 index from TextNode chunks.
        chunks: list of llama_index TextNode objects.
        """
        t0 = time.perf_counter()
        log.info("Building BM25 index from %d chunks...", len(chunks))

        self._chunks = [
            {
                "id":       c.id_,
                "text":     c.text,
                "metadata": dict(c.metadata),
            }
            for c in chunks
        ]
        self._corpus = [tokenize(c["text"]) for c in self._chunks]

        self._bm25 = BM25Okapi(
            self._corpus,
            k1=cfg.bm25.k1,
            b=cfg.bm25.b,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        log.info("BM25 index built: %d docs, vocab=%d, %.0f ms",
                 len(self._chunks),
                 len(self._bm25.idf),
                 elapsed)

    # ── Persist ────────────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> Path:
        p = path or cfg.paths.bm25
        payload = {
            "chunks":  self._chunks,
            "corpus":  self._corpus,
            "k1":      cfg.bm25.k1,
            "b":       cfg.bm25.b,
        }
        with open(p, "wb") as f:
            pickle.dump(payload, f)
        log.info("BM25 index saved → %s (%d KB)", p, p.stat().st_size // 1024)
        return p

    def load(self, path: Path | None = None) -> bool:
        p = path or cfg.paths.bm25
        if not p.exists():
            return False
        try:
            t0 = time.perf_counter()
            with open(p, "rb") as f:
                payload = pickle.load(f)
            self._chunks = payload["chunks"]
            self._corpus = payload["corpus"]
            self._bm25   = BM25Okapi(
                self._corpus,
                k1=payload.get("k1", cfg.bm25.k1),
                b=payload.get("b",  cfg.bm25.b),
            )
            elapsed = (time.perf_counter() - t0) * 1000
            log.info("BM25 index loaded: %d docs in %.0f ms", len(self._chunks), elapsed)
            return True
        except Exception as e:
            log.warning("BM25 load failed (%s) — will rebuild", e)
            return False

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int | None = None,
        boost_exact: bool = True,
    ) -> list[dict]:
        """
        BM25 keyword search.

        Args:
            query:       Natural language or keyword query
            top_k:       Number of results (default cfg.bm25.top_k)
            boost_exact: Apply course-code / faculty-name score multiplier

        Returns:
            List of dicts: {id, text, score, bm25_score, metadata, source}
            Sorted by descending score. Zero-score results are excluded.
        """
        if self._bm25 is None:
            log.error("BM25 index not loaded — call build() or load() first")
            return []

        k      = top_k or cfg.bm25.top_k
        q_norm = _normalize_course_tokens(query)
        tokens = tokenize(q_norm)

        if not tokens:
            log.warning("BM25: empty token list for query %r", query)
            return []

        t0     = time.perf_counter()
        scores = self._bm25.get_scores(tokens)
        elapsed = (time.perf_counter() - t0) * 1000

        # Apply exact-match boost for course codes and apparent proper names
        if boost_exact:
            boost_terms = _extract_boost_terms(query)
            if boost_terms:
                scores = _apply_boost(scores, self._chunks, boost_terms)
                log.debug("BM25 boost terms: %s", boost_terms)

        # Rank and filter zeros
        indexed = sorted(enumerate(scores), key=lambda x: -x[1])
        results = []
        for idx, score in indexed[:k]:
            if score <= 0.0:
                break
            chunk = self._chunks[idx]
            results.append({
                "id":         chunk["id"],
                "text":       chunk["text"],
                "bm25_score": round(float(score), 4),
                "score":      round(float(score), 4),   # alias for RRF fusion
                "metadata":   chunk["metadata"],
                "source":     "bm25",
            })

        log.info("BM25 search %r → %d results in %.1f ms", query[:60], len(results), elapsed)
        return results

    def __len__(self) -> int:
        return len(self._chunks)


# ── Boost helpers ─────────────────────────────────────────────────────────────

def _extract_boost_terms(query: str) -> set[str]:
    """
    Extract tokens from the query that deserve exact-match boosting:
    - Normalized CSE course codes (CSE 574, CSE 115)
    - Capitalized sequences that look like proper names (2+ words, each Title Case)
    """
    terms: set[str] = set()

    # Course codes
    for code in extract_course_codes(query):
        terms.add(code.lower())
        terms.add(code.replace(" ", "").lower())   # cse574 as well

    # Apparent proper names: "Chunming Qiao", "Rohini Srihari"
    name_re = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
    for m in name_re.finditer(query):
        for word in m.group(0).lower().split():
            if len(word) > 2:
                terms.add(word)

    return terms


def _apply_boost(
    scores:      "list[float] | any",
    chunks:      list[dict],
    boost_terms: set[str],
    multiplier:  float = 2.5,
) -> list[float]:
    """
    Multiply score by `multiplier` for chunks whose text contains any boost term.
    Deterministic — not ML — justified because course codes are discriminative IDs.
    """
    import numpy as np
    out = list(scores)
    for i, chunk in enumerate(chunks):
        text_lower = chunk["text"].lower()
        if any(term in text_lower for term in boost_terms):
            out[i] *= multiplier
    return out


# ── Singleton accessor ────────────────────────────────────────────────────────

_index: BM25Index | None = None


def get_bm25_index(rebuild: bool = False) -> BM25Index:
    """
    Return the singleton BM25Index, building/loading as needed.
    Called by retriever.py at query time.
    """
    global _index
    if _index is not None and not rebuild:
        return _index

    _index = BM25Index()
    if not rebuild and _index.load():
        return _index

    # Build from chunks
    chunks = ChunkingPipeline.load_chunks()
    if not chunks:
        log.error("No chunks found — run chunker.py first")
        return _index

    _index.build(chunks)
    _index.save()
    return _index


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 sparse retriever for UB CSE Chatbot")
    parser.add_argument("--search",  type=str, default="", help="Test query")
    parser.add_argument("--top-k",   type=int, default=5,  help="Results for --search")
    parser.add_argument("--rebuild", action="store_true",  help="Force rebuild from chunks.jsonl")
    args = parser.parse_args()

    index = get_bm25_index(rebuild=args.rebuild)
    print(f"\n✅  BM25 index: {len(index)} documents")

    if args.search:
        print(f"\n🔍  Query: {args.search!r}")
        results = index.search(args.search, top_k=args.top_k)
        if not results:
            print("  No results (all BM25 scores zero for this query)")
        for i, r in enumerate(results, 1):
            ptype = r["metadata"].get("page_type", "?")
            url   = r["metadata"].get("url", "?")[:65]
            print(f"  {i}. [bm25={r['bm25_score']:.3f}] [{ptype}] {url}")
            print(f"     {r['text'][:120].strip()}...")

    print(f"\n    Next → python graph_store.py")


if __name__ == "__main__":
    main()
