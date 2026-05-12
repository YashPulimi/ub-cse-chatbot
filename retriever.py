"""
retriever.py — Hybrid Retrieval with RRF Fusion for UB CSE Chatbot
==================================================================
Combines dense (Qdrant), sparse (BM25), and graph (Neo4j/NetworkX)
retrieval into a single ranked list using Reciprocal Rank Fusion (RRF).

Pipeline (per query):
  1. Run dense search    (Qdrant cosine similarity)
  2. Run BM25 search     (exact keyword match)
  3. Run graph retrieval (structured entity lookup)
  4. Fuse scores via RRF
  5. Apply intent-aware boosting for course codes / faculty names
  6. Return top-K candidates for reranker.py

Why RRF over score normalization:
  - Dense scores (cosine 0-1) and BM25 scores (unbounded TF-IDF) live on
    completely different scales. Score normalization requires knowing the
    full distribution — impractical in a streaming pipeline.
  - RRF only uses rank position, not raw score. It's parameter-stable
    (k=60 works well across domains) and handles missing results gracefully
    (a chunk absent from BM25 just gets no BM25 rank contribution).
  - Empirically: RRF outperforms linear interpolation on domain-specific
    corpora where one retriever dominates certain query types (BM25 wins
    on course codes, dense wins on paraphrased program questions).

Boosting:
  After RRF, chunks whose text contains an exact course code or faculty
  name from the query get a score multiplier. This is deterministic and
  justified — CSE course codes are discriminative identifiers, not
  semantic tokens. Boosting is applied before the reranker sees the list.

Concurrency:
  Dense and BM25 searches run concurrently via asyncio.gather().
  Graph retrieval runs concurrently with both.
  Total retrieval latency ≈ max(dense_ms, bm25_ms, graph_ms) instead of sum.

HOW TO USE:
  from retriever import HybridRetriever
  r = HybridRetriever()
  results = await r.retrieve("Who teaches CSE 574?")
  # results: list[dict] with keys id, text, score, metadata, source

HOW TO TEST:
  python retriever.py --query "prerequisites for CSE 574"
  python retriever.py --query "Rohini Srihari research areas"
  python retriever.py --query "MS program requirements"
  python retriever.py --query "CSE 115 instructor office hours"
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections import defaultdict
from typing import Any

from config import cfg
from graph_retriever import get_graph_retriever
from sparse_retriever import get_bm25_index
from utils import get_logger, extract_course_codes
from vector_store import VectorStore, get_qdrant_client

log = get_logger(__name__)

# ── Query expansion ──────────────────────────────────────────────────────────

import re as _re

_EXPANSIONS = {
    r'nlp':            'natural language processing text mining information extraction',
    r'cv':             'computer vision image processing pattern recognition',
    r'ml':             'machine learning deep learning artificial intelligence',
    r'ai':             'artificial intelligence machine learning',
    r'security':       'cybersecurity computer security',
    r'networks?':      'computer networks networking',
    r'hpc':            'high performance computing parallel',
    r'dbms':           'database systems',
}

def _expand_query(query: str) -> str:
    q = query
    for pattern, expansion in _EXPANSIONS.items():
        if _re.search(pattern, q, _re.IGNORECASE):
            q = q + ' ' + expansion
    return q.strip()


# ── Boost helpers (shared with sparse_retriever) ──────────────────────────────

import re

_FACULTY_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")


def _extract_boost_terms(query: str) -> set[str]:
    """
    Extract terms deserving exact-match score boosting:
    - Normalized CSE course codes
    - Apparent proper names (2+ Title Case words)
    """
    terms: set[str] = set()
    for code in extract_course_codes(query):
        terms.add(code.lower())
        terms.add(code.replace(" ", "").lower())   # cse574 variant
    for m in _FACULTY_NAME_RE.finditer(query):
        for word in m.group(0).lower().split():
            if len(word) > 2:
                terms.add(word)
    return terms


def _boost_score(result: dict, boost_terms: set[str], multiplier: float = 1.5) -> dict:
    """Return a copy of result with score multiplied if text contains a boost term."""
    text_lower = (result.get("text") or "").lower()
    if any(term in text_lower for term in boost_terms):
        return {**result, "score": result["score"] * multiplier}
    return result


# ── RRF fusion ────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int | None = None,
    weights: list[float] | None = None,
) -> list[dict]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score for document d:
        RRF(d) = Σ_i  weight_i / (k + rank_i(d))

    where rank_i(d) is 1-based position of d in list i (∞ if absent).

    Args:
        result_lists: One list per retriever, each item is a result dict
                      with at least {"id": str, "text": str, "metadata": dict}
        k:            RRF smoothing constant (default cfg.retrieval.rrf_k = 60)
        weights:      Per-list weights (default: cfg.retrieval dense/bm25 weights
                      for first two lists, 1.0 for remainder)

    Returns:
        Merged list sorted by RRF score descending, with rrf_score added.
        The original result dict from the highest-scoring source is kept
        (so text, metadata, source are preserved from the best retriever).
    """
    rrf_k = k or cfg.retrieval.rrf_k

    # Default weights: [dense_weight, bm25_weight, 1.0 for graph, ...]
    if weights is None:
        weights = [cfg.retrieval.dense_weight, cfg.retrieval.bm25_weight]
        while len(weights) < len(result_lists):
            weights.append(1.0)

    # Score accumulator: id → rrf_score
    scores:   dict[str, float] = defaultdict(float)
    # Best result per id (from whichever retriever had the highest single score)
    best_hit: dict[str, dict]  = {}

    for list_idx, (result_list, w) in enumerate(zip(result_lists, weights)):
        for rank, result in enumerate(result_list, start=1):
            rid = result.get("id") or result.get("text", "")[:64]
            scores[rid]   += w / (rrf_k + rank)
            # Keep the result dict from the retriever that scored it highest
            if rid not in best_hit or result.get("score", 0) > best_hit[rid].get("score", 0):
                best_hit[rid] = result

    # Sort by RRF score
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    fused  = []
    for rid, rrf_score in ranked:
        hit = dict(best_hit[rid])
        hit["rrf_score"]   = round(rrf_score, 6)
        hit["score"]       = round(rrf_score, 6)   # overwrite raw score with fused
        fused.append(hit)

    return fused


# ── Hybrid Retriever ──────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Runs dense + BM25 + graph concurrently and fuses results via RRF.

    Usage:
        retriever = HybridRetriever()
        results   = await retriever.retrieve("Who teaches CSE 574?")
        # or synchronously:
        results   = retriever.retrieve_sync("Who teaches CSE 574?")
    """

    def __init__(self) -> None:
        # Lazy-init: components are loaded on first use, not at import time
        self._vector_store: VectorStore | None   = None
        self._graph_retriever                    = None
        self._bm25_loaded: bool                  = False

    # ── Component accessors (lazy init) ───────────────────────────────────────

    def _get_vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore(client=get_qdrant_client())
        return self._vector_store

    def _get_graph_retriever(self):
        if self._graph_retriever is None:
            self._graph_retriever = get_graph_retriever()
        return self._graph_retriever

    def _get_bm25(self):
        return get_bm25_index()

    # ── Individual retrievers ─────────────────────────────────────────────────

    def _dense_search(self, query: str, top_k: int) -> list[dict]:
        try:
            results = self._get_vector_store().search(query, top_k=top_k)
            log.debug("Dense: %d results", len(results))
            return results
        except Exception as e:
            log.warning("Dense search failed: %s", e)
            return []

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        try:
            results = self._get_bm25().search(query, top_k=top_k)
            log.debug("BM25: %d results", len(results))
            return results
        except Exception as e:
            log.warning("BM25 search failed: %s", e)
            return []

    def _graph_search(self, query: str) -> list[dict]:
        try:
            results = self._get_graph_retriever().retrieve(query)
            log.debug("Graph: %d results", len(results))
            return results
        except Exception as e:
            log.warning("Graph retrieval failed: %s", e)
            return []

    # ── Core async retrieve ───────────────────────────────────────────────────

    async def retrieve(
        self,
        query:             str,
        top_k:             int | None   = None,
        use_graph:         bool         = True,
        dense_weight:      float | None = None,
        bm25_weight:       float | None = None,
        score_threshold:   float        = 0.0,
    ) -> list[dict]:
        """
        Full hybrid retrieval pipeline.

        Args:
            query:           Natural language query
            top_k:           Final number of candidates to return
                             (before reranking). Default: cfg.retrieval.top_k
            use_graph:       Whether to include graph evidence in results.
                             Automatically False if no graph-relevant intent.
            dense_weight:    Override for dense RRF weight
            bm25_weight:     Override for BM25 RRF weight
            score_threshold: Minimum RRF score to include a result

        Returns:
            List of result dicts sorted by fused RRF score, limited to top_k.
            Each dict: {id, text, score, rrf_score, metadata, source,
                        dense_rank?, bm25_rank?}
        """
        k = top_k or cfg.retrieval.top_k
        t0 = time.perf_counter()

        # Run dense and BM25 concurrently in a thread executor
        # (both are blocking/sync — we don't want them to serialize)
        loop = asyncio.get_event_loop()

        _eq = _expand_query(query)
        dense_future = loop.run_in_executor(None, self._dense_search, _eq, k)
        bm25_future  = loop.run_in_executor(None, self._bm25_search,  _eq, k)
        graph_future = (
            loop.run_in_executor(None, self._graph_search, query)
            if use_graph
            else asyncio.sleep(0, result=[])
        )

        dense_results, bm25_results, graph_results = await asyncio.gather(
            dense_future, bm25_future, graph_future
        )

        t_retrieve = (time.perf_counter() - t0) * 1000
        log.info(
            "Retrieval: dense=%d bm25=%d graph=%d in %.1f ms",
            len(dense_results), len(bm25_results), len(graph_results), t_retrieve,
        )

        # ── RRF fusion ────────────────────────────────────────────────────────
        dw = dense_weight or cfg.retrieval.dense_weight
        bw = bm25_weight  or cfg.retrieval.bm25_weight

        # Tag source before fusion for provenance
        for r in dense_results:
            r["source"]      = "dense"
        for r in bm25_results:
            r["source"]      = "bm25"
        for r in graph_results:
            r["source"]      = r.get("source", "graph")

        # Add rank metadata for UI display
        for i, r in enumerate(dense_results, 1):
            r["dense_rank"]  = i
        for i, r in enumerate(bm25_results, 1):
            r["bm25_rank"]   = i

        # Graph results are always injected at the top regardless of RRF
        # (they contain structured entity data, not retrieved chunks)
        # We add them as a fourth list with high weight so they stay visible
        result_lists = [dense_results, bm25_results]
        weights      = [dw, bw]
        if graph_results:
            result_lists.append(graph_results)
            weights.append(1.2)   # graph evidence is always highly relevant

        fused = reciprocal_rank_fusion(result_lists, weights=weights)

        # ── Exact-match boosting ───────────────────────────────────────────────
        boost_terms = _extract_boost_terms(query)
        if boost_terms:
            fused = [_boost_score(r, boost_terms) for r in fused]
            # Re-sort after boosting
            fused.sort(key=lambda x: -x["score"])
            log.debug("Boost terms applied: %s", boost_terms)

        # ── Filter and cap ────────────────────────────────────────────────────
        if score_threshold > 0:
            fused = [r for r in fused if r["score"] >= score_threshold]

        fused = fused[:k]

        total_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "Hybrid retrieve: %d candidates in %.1f ms (retrieve=%.1f ms)",
            len(fused), total_ms, t_retrieve,
        )

        return fused

    def retrieve_sync(self, query: str, **kwargs) -> list[dict]:
        """
        Synchronous wrapper for retrieve().
        Use this from non-async code (CLI, evaluator, tests).
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in an async context (e.g. Chainlit) — use nest_asyncio
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self.retrieve(query, **kwargs))
            else:
                return loop.run_until_complete(self.retrieve(query, **kwargs))
        except RuntimeError:
            return asyncio.run(self.retrieve(query, **kwargs))

    # ── Latency breakdown ─────────────────────────────────────────────────────

    async def retrieve_with_latency(self, query: str, **kwargs) -> tuple[list[dict], dict]:
        """
        Same as retrieve() but also returns a latency breakdown dict.
        Used by app.py to display per-component timing in the UI.
        Runs all three retrievers concurrently.
        """
        timings: dict[str, float] = {}
        loop  = asyncio.get_event_loop()
        top_k = kwargs.get("top_k", cfg.retrieval.top_k)

        t0 = time.perf_counter()

        # Run all three concurrently
        _eq = _expand_query(query)
        dense_future = loop.run_in_executor(None, self._dense_search, _eq, top_k)
        bm25_future  = loop.run_in_executor(None, self._bm25_search,  _eq, top_k)
        graph_future = loop.run_in_executor(None, self._graph_search, query)

        t_dense_start = time.perf_counter()
        dense_results, bm25_results, graph_results = await asyncio.gather(
            dense_future, bm25_future, graph_future,
            return_exceptions=True,
        )

        # Handle exceptions from gather
        if isinstance(dense_results, Exception):
            log.warning("Dense search failed: %s", dense_results)
            dense_results = []
        if isinstance(bm25_results, Exception):
            log.warning("BM25 search failed: %s", bm25_results)
            bm25_results = []
        if isinstance(graph_results, Exception):
            log.warning("Graph search failed: %s", graph_results)
            graph_results = []

        timings["dense_ms"]  = round((time.perf_counter() - t_dense_start) * 1000, 1)
        timings["bm25_ms"]   = timings["dense_ms"]   # concurrent — same wall time
        timings["graph_ms"]  = timings["dense_ms"]

        # Fuse
        tf = time.perf_counter()
        for r in dense_results: r["source"] = "dense"
        for r in bm25_results:  r["source"] = "bm25"
        for r in graph_results: r["source"] = r.get("source", "graph")
        for i, r in enumerate(dense_results, 1): r["dense_rank"] = i
        for i, r in enumerate(bm25_results,  1): r["bm25_rank"]  = i

        result_lists = [dense_results, bm25_results]
        weights      = [cfg.retrieval.dense_weight, cfg.retrieval.bm25_weight]
        if graph_results:
            result_lists.append(graph_results)
            weights.append(1.2)

        fused = reciprocal_rank_fusion(result_lists, weights=weights)
        boost_terms = _extract_boost_terms(query)
        if boost_terms:
            fused = [_boost_score(r, boost_terms) for r in fused]
            fused.sort(key=lambda x: -x["score"])

        fused = fused[:top_k]

        timings["fusion_ms"] = round((time.perf_counter() - tf) * 1000, 1)
        timings["total_ms"]  = round((time.perf_counter() - t0) * 1000, 1)
        timings["n_dense"]   = len(dense_results)
        timings["n_bm25"]    = len(bm25_results)
        timings["n_graph"]   = len(graph_results)
        timings["n_fused"]   = len(fused)

        log.info(
            "retrieve_with_latency: dense=%d bm25=%d graph=%d fused=%d in %.0f ms",
            len(dense_results), len(bm25_results), len(graph_results),
            len(fused), timings["total_ms"],
        )

        return fused, timings


# ── Singleton ─────────────────────────────────────────────────────────────────

_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


# ── Entry point ───────────────────────────────────────────────────────────────

def _print_results(results: list[dict], verbose: bool = False) -> None:
    print(f"\n{'='*70}")
    print(f"  {len(results)} results")
    print(f"{'='*70}")
    for i, r in enumerate(results, 1):
        src   = r.get("source", "?")
        score = r.get("score", 0)
        ptype = r.get("metadata", {}).get("page_type", "?")
        url   = r.get("metadata", {}).get("url", "")[:60]
        dr    = r.get("dense_rank", "-")
        br    = r.get("bm25_rank",  "-")
        print(f"\n  {i}. [{src:6}] score={score:.5f}  type={ptype}")
        print(f"     dense_rank={dr}  bm25_rank={br}")
        if url:
            print(f"     url: {url}")
        snippet = (r.get("text") or "")[:200].strip().replace("\n", " ")
        print(f"     {snippet}...")
        if verbose and r.get("source") == "graph":
            print(f"\n     [Full graph evidence]\n{r.get('text','')}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid retriever for UB CSE Chatbot")
    parser.add_argument("--query",   type=str, required=True, help="Query to test")
    parser.add_argument("--top-k",   type=int, default=10,    help="Number of results")
    parser.add_argument("--no-graph",action="store_true",     help="Disable graph retrieval")
    parser.add_argument("--verbose", action="store_true",     help="Show full graph evidence")
    args = parser.parse_args()

    retriever = HybridRetriever()

    print(f"\n🔍  Query: {args.query!r}")
    t0      = time.perf_counter()
    results = retriever.retrieve_sync(
        args.query,
        top_k=args.top_k,
        use_graph=not args.no_graph,
    )
    elapsed = (time.perf_counter() - t0) * 1000

    _print_results(results, verbose=args.verbose)
    print(f"\n⏱   Total: {elapsed:.0f} ms")
    print(f"\n    Next → python reranker.py")


if __name__ == "__main__":
    main()