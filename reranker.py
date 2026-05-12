"""
reranker.py — Cross-Encoder Reranker for UB CSE Chatbot
========================================================
Takes the top-K candidates from retriever.py and re-scores them
using a cross-encoder model, returning the top-N most relevant chunks
for the generator.

Why rerank after RRF:
  - RRF fusion ranks by position across retrievers — it has no notion
    of query-document relevance, only of consensus rank.
  - A bi-encoder (bge-small) embeds query and document independently,
    so it misses fine-grained interactions ("office hours" in query vs
    "office hours: MWF 2-3pm" in chunk vs "office: Davis 338").
  - A cross-encoder reads the (query, document) pair jointly, giving
    it full attention over both — dramatically better at distinguishing
    "CSE 574 prerequisites" from "CSE 574 description" when both rank
    equally in RRF.
  - Latency trade-off: cross-encoder is ~10x slower per pair than
    bi-encoder, so we only run it on the top-20 RRF candidates, not
    the full corpus. top-20 → top-5 is the standard RAG pattern.

Model choice (cross-encoder/ms-marco-MiniLM-L-6-v2):
  - 22M params, runs on CPU in ~80ms for 20 pairs
  - Trained on MS MARCO (web QA) — generalises well to academic domain
  - Upgrade: BAAI/bge-reranker-base → better precision, 2x slower
  - Both are drop-in via cfg.reranker.model

Graph evidence handling:
  - Graph result dicts (source="graph") contain structured entity data,
    not prose — cross-encoder scores them poorly because they look like
    key-value dumps, not natural text.
  - Strategy: always keep graph results in the final output regardless
    of their cross-encoder score. They are injected as a separate
    "context" block in the generator prompt, not interleaved with chunks.

HOW TO USE:
  from reranker import get_reranker
  reranker = get_reranker()
  ranked = await reranker.rerank("Who teaches CSE 574?", candidates)
  # ranked: list[dict] — top cfg.reranker.top_k results, score updated

HOW TO TEST:
  python reranker.py --query "prerequisites for CSE 574"
  python reranker.py --query "Rohini Srihari office hours"
  python reranker.py --query "MS admission requirements" --top-k 5
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from sentence_transformers import CrossEncoder

from config import cfg
from utils import get_logger

log = get_logger(__name__)


# ── Cross-encoder wrapper ─────────────────────────────────────────────────────

class Reranker:
    """
    Cross-encoder reranker with:
    - Lazy model loading (first call only)
    - Async interface via run_in_executor (non-blocking)
    - Graph evidence passthrough (always kept regardless of score)
    - Score logging for UI display (app.py backend panel)
    - Singleton pattern
    """

    _singleton: "Reranker | None" = None

    def __init__(self) -> None:
        self._model: CrossEncoder | None = None
        self._model_name = cfg.reranker.model
        self._top_k      = cfg.reranker.top_k

    @classmethod
    def instance(cls) -> "Reranker":
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_model(self) -> CrossEncoder:
        if self._model is None:
            log.info("Loading reranker model: %s", self._model_name)
            t0 = time.perf_counter()
            self._model = CrossEncoder(
                self._model_name,
                max_length=512,
                device=cfg.embedding.device,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            log.info("Reranker loaded in %.0f ms", elapsed)
        return self._model

    # ── Sync rerank (runs in executor) ────────────────────────────────────────

    def _rerank_sync(
        self,
        query:      str,
        candidates: list[dict],
        top_k:      int,
    ) -> list[dict]:
        """
        Blocking rerank — called via run_in_executor from async context.

        Separates graph evidence from text chunks:
          - Text chunks  → scored by cross-encoder, top_k kept
          - Graph results → always passed through, prepended to output

        Returns combined list: graph_results + top_k text chunks
        """
        if not candidates:
            return []

        model = self._load_model()

        # Split graph evidence from regular chunks
        graph_results = [r for r in candidates if r.get("source") == "graph"]
        text_results  = [r for r in candidates if r.get("source") != "graph"]

        if not text_results:
            log.debug("Reranker: only graph results, skipping cross-encoder")
            return graph_results

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, r.get("text", "")[:512]) for r in text_results]

        t0     = time.perf_counter()
        scores = model.predict(pairs, show_progress_bar=False)
        elapsed = (time.perf_counter() - t0) * 1000

        log.info(
            "Reranker scored %d pairs in %.0f ms (%.1f ms/pair)",
            len(pairs), elapsed, elapsed / max(len(pairs), 1),
        )

        # Attach cross-encoder score and original RRF rank to each result
        scored = []
        for result, ce_score in zip(text_results, scores):
            r = dict(result)
            r["ce_score"]   = round(float(ce_score), 4)
            r["rrf_score"]  = r.get("rrf_score", r.get("score", 0.0))
            r["score"]      = r["ce_score"]   # overwrite with cross-encoder score
            scored.append(r)

        # If all CE scores are very negative, the cross-encoder has no signal.
        # Fall back to original RRF order (retrieval was better than reranker).
        max_ce = max(r["ce_score"] for r in scored) if scored else 0
        if max_ce < -3.0:
            log.info("All CE scores negative (max=%.2f) — using RRF order", max_ce)
            scored.sort(key=lambda x: -x.get("rrf_score", 0))
        else:
            # Apply page-type boost/penalty exactly once before sorting.
            _HIGH_VALUE = {"faculty_profile", "faculty", "research", "faculty_website",
                           "courses", "catalog_course", "degree_requirements", "cv_pdf"}
            for r in scored:
                pt  = r.get("metadata", {}).get("page_type", "")
                url = r.get("metadata", {}).get("url", "").lower()
                if pt in _HIGH_VALUE:
                    r["ce_score"] += 0.5
                if pt in {"general", "hub_curriculum"}:
                    r["ce_score"] -= 2.0
                if "apply" in url or ("admission" in url and "faculty" not in url):
                    r["ce_score"] -= 1.5
            scored.sort(key=lambda x: -x["ce_score"])

        # Log score distribution for UI backend panel
        if scored:
            top_scores = [f"{r['ce_score']:.3f}" for r in scored[:5]]
            log.info("Reranker top-5 CE scores: %s", top_scores)

        # Keep top_k text chunks + all graph results
        final_text   = scored[:top_k]
        final_result = graph_results + final_text

        log.info(
            "Reranked %d → %d chunks (+ %d graph results)",
            len(text_results), len(final_text), len(graph_results),
        )
        return final_result

    # ── Async interface ───────────────────────────────────────────────────────

    async def rerank(
        self,
        query:      str,
        candidates: list[dict],
        top_k:      int | None = None,
    ) -> list[dict]:
        """
        Async rerank: runs cross-encoder in a thread executor so the
        Chainlit event loop stays unblocked during inference.

        Args:
            query:      The user's question
            candidates: top-K results from retriever.py
            top_k:      How many text chunks to keep (default cfg.reranker.top_k)

        Returns:
            graph_results (all) + top_k text chunks, sorted by ce_score
        """
        k    = top_k or self._top_k
        loop = asyncio.get_event_loop()

        if cfg.reranker.use_executor:
            result = await loop.run_in_executor(
                None, self._rerank_sync, query, candidates, k
            )
        else:
            result = self._rerank_sync(query, candidates, k)

        return result

    def rerank_sync(
        self,
        query:      str,
        candidates: list[dict],
        top_k:      int | None = None,
    ) -> list[dict]:
        """Synchronous wrapper for non-async callers (evaluator, CLI)."""
        k = top_k or self._top_k
        return self._rerank_sync(query, candidates, k)

    # ── Score summary (for app.py UI panel) ───────────────────────────────────

    @staticmethod
    def score_summary(results: list[dict]) -> list[dict]:
        """
        Return a compact score summary for each result.
        Used by app.py to render the reranking panel in the UI.

        Returns list of dicts: {rank, source, ce_score, rrf_score,
                                page_type, url_snippet, text_snippet}
        """
        summary = []
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            summary.append({
                "rank":         i,
                "source":       r.get("source", "?"),
                "ce_score":     r.get("ce_score",  "n/a"),
                "rrf_score":    round(r.get("rrf_score", 0), 5),
                "dense_rank":   r.get("dense_rank",  "-"),
                "bm25_rank":    r.get("bm25_rank",   "-"),
                "page_type":    meta.get("page_type", "?"),
                "url":          meta.get("url", "")[:70],
                "text_snippet": (r.get("text") or "")[:120].replace("\n", " "),
            })
        return summary


# ── Singleton accessor ────────────────────────────────────────────────────────

def get_reranker() -> Reranker:
    return Reranker.instance()


# ── Entry point ───────────────────────────────────────────────────────────────

async def _run_test(query: str, top_k: int) -> None:
    """Test reranker end-to-end using the live hybrid retriever."""
    from retriever import get_retriever

    print(f"\n🔍  Query: {query!r}")

    # Retrieve candidates
    retriever  = get_retriever()
    t0         = time.perf_counter()
    candidates = retriever.retrieve_sync(query, top_k=cfg.retrieval.top_k)
    t_ret      = (time.perf_counter() - t0) * 1000
    print(f"    Retrieved {len(candidates)} candidates in {t_ret:.0f} ms")

    # Rerank
    reranker = get_reranker()
    t1       = time.perf_counter()
    ranked   = await reranker.rerank(query, candidates, top_k=top_k)
    t_rank   = (time.perf_counter() - t1) * 1000
    print(f"    Reranked  → {len(ranked)} results in {t_rank:.0f} ms")

    # Print score summary
    print(f"\n{'='*70}")
    summary = Reranker.score_summary(ranked)
    for s in summary:
        ce  = f"{s['ce_score']:.4f}" if isinstance(s["ce_score"], float) else s["ce_score"]
        rrf = f"{s['rrf_score']:.5f}"
        print(
            f"  {s['rank']:2}. [{s['source']:6}] CE={ce:>8}  RRF={rrf}"
            f"  [{s['page_type']}]"
        )
        print(f"      {s['text_snippet']}...")
    print(f"\n⏱   Retrieve={t_ret:.0f}ms  Rerank={t_rank:.0f}ms"
          f"  Total={t_ret+t_rank:.0f}ms")
    print(f"\n    Next → python guardrails.py")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Cross-encoder reranker for UB CSE Chatbot")
    parser.add_argument("--query",  type=str, required=True, help="Query to test")
    parser.add_argument("--top-k",  type=int, default=cfg.reranker.top_k,
                        help=f"Final results to keep (default {cfg.reranker.top_k})")
    args = parser.parse_args()

    asyncio.run(_run_test(args.query, args.top_k))


if __name__ == "__main__":
    main()