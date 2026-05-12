"""
vector_store.py — Qdrant Vector Store for UB CSE Chatbot
=========================================================
Indexes chunks from chunker.py into a local Qdrant collection.
Provides dense vector search with metadata filtering.

Why Qdrant over ChromaDB:
  - Proper payload filter API: filter by page_type, faculty_email,
    course_code, source — all at query time, not post-hoc in Python
  - Named vectors allow adding sparse (SPLADE) later without reindexing
  - Local on-disk mode: no Docker needed for dev (QdrantClient(path=...))
  - 2-5x faster ANN search at >50K vectors vs ChromaDB
  - Production-ready: same client works local or remote Docker/Cloud

Collection schema:
  - One point per chunk
  - Vector: 384-dim bge-small normalized float
  - Payload (metadata): all chunk metadata fields

HOW TO RUN:
  python vector_store.py           # build index from chunks.jsonl
  python vector_store.py --reset   # drop collection and rebuild
  python vector_store.py --search "who teaches CSE 574"  # quick test

NEXT: python sparse_retriever.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from chunker import ChunkingPipeline
from config import cfg
from embeddings import get_embed_model
from ingestion_pipeline import IngestionPipeline
from utils import get_logger, async_timed

log = get_logger(__name__)

# ── Client factory ────────────────────────────────────────────────────────────

def get_qdrant_client() -> QdrantClient:
    """
    Return a QdrantClient in local on-disk mode (default) or
    remote mode if QDRANT_HOST is set in .env.
    Local mode needs no Docker — persists to data/indexes/qdrant_db/.
    """
    if cfg.qdrant.host:
        log.info("Connecting to Qdrant at %s:%d", cfg.qdrant.host, cfg.qdrant.port)
        return QdrantClient(
            host=cfg.qdrant.host,
            port=cfg.qdrant.port,
            grpc_port=6334 if cfg.qdrant.use_grpc else None,
            prefer_grpc=cfg.qdrant.use_grpc,
        )
    else:
        path = cfg.qdrant.local_path
        log.info("Using Qdrant local on-disk mode at %s", path)
        return QdrantClient(path=path)


# ── Collection management ─────────────────────────────────────────────────────

class VectorStore:

    def __init__(self, client: QdrantClient | None = None) -> None:
        self.client     = client or get_qdrant_client()
        self.collection = cfg.qdrant.collection
        self.embed      = get_embed_model()

    def collection_exists(self) -> bool:
        try:
            self.client.get_collection(self.collection)
            return True
        except Exception:
            return False

    def create_collection(self, overwrite: bool = False) -> None:
        """Create Qdrant collection. Skips if already exists unless overwrite=True."""
        if self.collection_exists():
            if not overwrite:
                info = self.client.get_collection(self.collection)
                log.info(
                    "Collection '%s' already exists (%d points) — skipping create",
                    self.collection,
                    info.points_count,
                )
                return
            log.info("Dropping collection '%s' for rebuild", self.collection)
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=self.embed.dim,
                distance=Distance.COSINE,
            ),
        )
        # Create payload indexes for fast metadata filtering
        # These allow: filter(page_type="faculty_profile") at query time
        for field, schema in [
            ("page_type",       PayloadSchemaType.KEYWORD),
            ("source",          PayloadSchemaType.KEYWORD),
            ("chunk_strategy",  PayloadSchemaType.KEYWORD),
            ("faculty_rank",    PayloadSchemaType.KEYWORD),
            ("course_code",     PayloadSchemaType.KEYWORD),
        ]:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                pass  # index may already exist

        log.info("Collection '%s' created (dim=%d, distance=COSINE)",
                 self.collection, self.embed.dim)

    def reset(self) -> None:
        """Drop and recreate the collection — full reindex."""
        self.create_collection(overwrite=True)

    # ── Indexing ──────────────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list, batch_size: int = 64) -> int:
        """
        Embed and upsert TextNode chunks into Qdrant.
        Returns total points upserted.
        Uses content_hash as the point ID seed for idempotent upserts.
        """
        if not chunks:
            log.warning("No chunks to upsert")
            return 0

        total = 0
        t0    = time.perf_counter()

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            texts = [c.text for c in batch]

            # Embed batch
            vectors = self.embed.embed_texts(texts, batch_size=batch_size)

            # Build Qdrant points
            points = []
            for chunk, vec in zip(batch, vectors):
                # Use first 16 chars of uuid as numeric-compatible ID
                # Qdrant supports string IDs (UUIDs)
                payload = _sanitize_payload(dict(chunk.metadata))
                points.append(PointStruct(
                    id=chunk.id_,
                    vector=vec,
                    payload=payload,
                ))

            self.client.upsert(
                collection_name=self.collection,
                points=points,
                wait=True,
            )
            total += len(batch)
            elapsed = (time.perf_counter() - t0) * 1000
            log.info("Upserted %d/%d chunks (%.0f ms elapsed)",
                     total, len(chunks), elapsed)

        # Save embedding cache after full index build
        self.embed.save_cache()

        total_elapsed = (time.perf_counter() - t0) * 1000
        log.info("Indexing complete: %d chunks in %.0f ms (%.1f ms/chunk)",
                 total, total_elapsed, total_elapsed / max(total, 1))
        return total

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filter_page_type: str | None = None,
        filter_source: str | None = None,
        score_threshold: float = 0.0,
    ) -> list[dict]:
        """
        Dense vector search with optional metadata filters.

        Args:
            query:             Natural language query string
            top_k:             Number of results (default: cfg.retrieval.top_k)
            filter_page_type:  Only return chunks with this page_type
            filter_source:     Only return chunks from this source domain
            score_threshold:   Minimum cosine similarity score

        Returns:
            List of dicts with keys: text, score, metadata, id
        """
        k   = top_k or cfg.retrieval.top_k
        vec = self.embed.embed_query(query)

        # Build optional filter
        conditions = []
        if filter_page_type:
            conditions.append(FieldCondition(
                key="page_type",
                match=MatchValue(value=filter_page_type),
            ))
        if filter_source:
            conditions.append(FieldCondition(
                key="source",
                match=MatchValue(value=filter_source),
            ))
        qdrant_filter = Filter(must=conditions) if conditions else None

        t0 = time.perf_counter()
        # Qdrant v1.12+ uses query_points() instead of search()
        response = self.client.query_points(
            collection_name=self.collection,
            query=vec,
            limit=k,
            query_filter=qdrant_filter,
            score_threshold=score_threshold if score_threshold > 0.0 else None,
            with_payload=True,
        )
        hits = response.points
        elapsed = (time.perf_counter() - t0) * 1000
        log.info("Dense search for %r → %d hits in %.1f ms", query[:60], len(hits), elapsed)

        return [
            {
                "id":       hit.id,
                "score":    round(hit.score, 4),
                "text":     hit.payload.get("text", "") if hit.payload else "",
                "metadata": hit.payload or {},
                "source":   "dense",
            }
            for hit in hits
        ]

    def search_with_text(
        self,
        query: str,
        chunks_lookup: dict[str, str],
        top_k: int | None = None,
        **kwargs,
    ) -> list[dict]:
        """
        Search and enrich results with chunk text from a lookup dict.
        chunks_lookup: {chunk_id → chunk_text}
        """
        results = self.search(query, top_k=top_k, **kwargs)
        for r in results:
            r["text"] = chunks_lookup.get(r["id"], r.get("text", ""))
        return results

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return collection statistics."""
        try:
            info = self.client.get_collection(self.collection)
            # points_count may be None in local mode before first commit
            return {
                "collection":    self.collection,
                "points_count":  info.points_count or 0,
                "vectors_count": getattr(info, "vectors_count", None) or 0,
                "status":        str(info.status),
            }
        except Exception as e:
            return {"error": str(e)}


# ── Payload sanitizer ─────────────────────────────────────────────────────────

def _sanitize_payload(meta: dict) -> dict:
    """
    Qdrant payload values must be scalar, list[scalar], or nested dict.
    Remove or stringify any non-serializable values.
    Also store the chunk text in payload so retrieval is self-contained.
    """
    import json as _json
    # Parse JSON-string list fields into real Python lists so Qdrant
    # can use them for metadata filters (graph-guided chunk retrieval)
    _json_list_keys = ("courses_in_chunk", "courses_mentioned", "faculty_names",
                       "heading_path", "pdf_paths", "faculty_research_areas")
    parsed = dict(meta)
    for jk in _json_list_keys:
        if jk in parsed and isinstance(parsed[jk], str):
            try:
                parsed[jk] = _json.loads(parsed[jk])
            except Exception:
                pass
    meta = parsed

    clean: dict = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, list):
            # Keep only scalar lists
            scalar_list = [x for x in v if isinstance(x, (str, int, float, bool))]
            if scalar_list:
                clean[k] = scalar_list
        elif isinstance(v, dict):
            # Flatten one level
            for dk, dv in v.items():
                if isinstance(dv, (str, int, float, bool)):
                    clean[f"{k}.{dk}"] = dv
        else:
            clean[k] = str(v)
    return clean


# ── Build index pipeline ──────────────────────────────────────────────────────

def build_index(reset: bool = False) -> VectorStore:
    """Full pipeline: load chunks → create collection → upsert → return store."""

    # Load chunks
    chunks = ChunkingPipeline.load_chunks()
    if not chunks:
        log.info("No chunks found — running chunker pipeline first")
        docs    = IngestionPipeline.load_documents()
        pipeline = ChunkingPipeline()
        chunks  = pipeline.run(docs)
        pipeline.save()

    # Store chunk text in payload so search() is self-contained
    chunks_with_text = []
    for c in chunks:
        c.metadata["text"] = c.text   # store text in payload
        chunks_with_text.append(c)

    # Build vector store
    store = VectorStore()
    store.create_collection(overwrite=reset)

    # Only upsert if collection is empty or reset
    stats = store.stats()
    if not reset and stats.get("points_count", 0) >= len(chunks):
        log.info("Collection already has %d points (chunks=%d) — skipping upsert",
                 stats["points_count"], len(chunks))
        return store

    n = store.upsert_chunks(chunks_with_text)
    log.info("Index built: %d points in collection '%s'", n, store.collection)
    return store


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build or query Qdrant vector index")
    parser.add_argument("--reset",  action="store_true", help="Drop and rebuild the collection")
    parser.add_argument("--search", type=str, default="",  help="Quick search test query")
    parser.add_argument("--top-k",  type=int, default=5,   help="Results for --search")
    args = parser.parse_args()

    store = build_index(reset=args.reset)

    stats = store.stats()
    print(f"\n✅  Vector index: {stats}")

    if args.search:
        print(f"\n🔍  Searching: {args.search!r}")
        results = store.search(args.search, top_k=args.top_k)
        for i, r in enumerate(results, 1):
            ptype = r["metadata"].get("page_type", "?")
            url   = r["metadata"].get("url", "?")[:70]
            print(f"  {i}. [{r['score']:.3f}] [{ptype}] {url}")
            print(f"     {r['text'][:120]}...")

    print(f"\n    Next → python sparse_retriever.py")


if __name__ == "__main__":
    main()