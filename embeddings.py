"""
embeddings.py — Embedding model wrapper with batching and cache
===============================================================
Provides a single EmbeddingModel class used by:
  - vector_store.py   (indexing chunks)
  - retriever.py      (query embedding at query time)

Design:
  - Uses HuggingFace bge-small-en-v1.5 (default) or any sentence-transformers model
  - Batches encode calls to maximise GPU/CPU throughput
  - Caches embeddings keyed by content_hash to avoid re-embedding unchanged chunks
  - Thread-safe for use from async code via run_in_executor
  - Query embeddings cached in an LRU dict (TTL-based)

WHY bge-small-en-v1.5:
  - 33M params, 384-dim, fits in 200 MB RAM
  - ~2000 sentences/sec on modern CPU
  - Strong MTEB performance for retrieval tasks
  - Upgrade path: swap EMBED_MODEL=BAAI/bge-base-en-v1.5 for 768-dim / better recall

HOW TO USE:
  from embeddings import get_embed_model
  model = get_embed_model()
  vecs = model.embed_texts(["hello", "world"])   # returns list[list[float]]
  q_vec = model.embed_query("What is CSE 574?")  # returns list[float]
"""

from __future__ import annotations

import hashlib
import pickle
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from config import cfg
from utils import get_logger

log = get_logger(__name__)

# ── Model singleton ───────────────────────────────────────────────────────────

_model: SentenceTransformer | None = None


def get_embed_model() -> "EmbeddingModel":
    """Return the singleton EmbeddingModel. Thread-safe (GIL)."""
    return EmbeddingModel.instance()


class EmbeddingModel:
    """
    Thin wrapper around SentenceTransformer with:
    - batch encoding
    - persistent disk cache keyed by text hash
    - in-memory LRU query cache
    """
    _singleton: "EmbeddingModel | None" = None

    def __init__(self) -> None:
        log.info("Loading embedding model: %s on %s", cfg.embedding.model, cfg.embedding.device)
        t0 = time.perf_counter()
        self._model = SentenceTransformer(
            cfg.embedding.model,
            device=cfg.embedding.device,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        log.info("Embedding model loaded in %.0f ms", elapsed)

        # Persistent disk cache: {text_hash → embedding_list}
        self._cache_path = cfg.paths.embed_cache
        self._cache: dict[str, list[float]] = self._load_cache()
        self._cache_dirty = False

        # In-memory LRU for repeated query embeddings (TTL not needed — queries repeat)
        self._query_cache: dict[str, list[float]] = {}
        self._query_cache_max = cfg.cache.embed_max_size

    @classmethod
    def instance(cls) -> "EmbeddingModel":
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    # ── Cache helpers ──────────────────────────────────────────────────────────

    def _load_cache(self) -> dict[str, list[float]]:
        if self._cache_path.exists():
            try:
                with open(self._cache_path, "rb") as f:
                    cache = pickle.load(f)
                log.info("Embedding cache loaded: %d entries from %s",
                         len(cache), self._cache_path.name)
                return cache
            except Exception as e:
                log.warning("Embedding cache corrupted (%s) — starting fresh", e)
        return {}

    def save_cache(self) -> None:
        """Write cache to disk. Call after batch indexing completes."""
        if not self._cache_dirty:
            return
        try:
            with open(self._cache_path, "wb") as f:
                pickle.dump(self._cache, f)
            log.info("Embedding cache saved: %d entries → %s",
                     len(self._cache), self._cache_path.name)
            self._cache_dirty = False
        except Exception as e:
            log.warning("Failed to save embedding cache: %s", e)

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    # ── Core encode ───────────────────────────────────────────────────────────

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Embed a list of texts. Hits cache first; only encodes cache misses.
        Returns embeddings in the same order as input texts.
        """
        bs = batch_size or cfg.embedding.batch_size
        results: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts:   list[str] = []

        # Cache lookup
        for i, text in enumerate(texts):
            key = self._text_hash(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                miss_indices.append(i)
                miss_texts.append(text)

        cache_hits = len(texts) - len(miss_indices)
        if cache_hits:
            log.debug("Embed cache: %d hits / %d total", cache_hits, len(texts))

        # Batch encode misses
        if miss_texts:
            t0 = time.perf_counter()
            encoded = self._model.encode(
                miss_texts,
                batch_size=bs,
                show_progress_bar=show_progress,
                normalize_embeddings=True,   # cosine similarity → dot product
                convert_to_numpy=True,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            log.info("Encoded %d texts in %.0f ms (%.1f ms/text)",
                     len(miss_texts), elapsed, elapsed / max(len(miss_texts), 1))

            for idx, (orig_i, text) in enumerate(zip(miss_indices, miss_texts)):
                vec = encoded[idx].tolist()
                key = self._text_hash(text)
                self._cache[key] = vec
                self._cache_dirty = True
                results[orig_i] = vec

        return results  # type: ignore[return-value]

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string with in-memory LRU cache.
        Queries are short and repeated often (same question → cached embedding).
        """
        if query in self._query_cache:
            return self._query_cache[query]

        vec = self._model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        # Simple LRU: evict oldest when full
        if len(self._query_cache) >= self._query_cache_max:
            oldest = next(iter(self._query_cache))
            del self._query_cache[oldest]
        self._query_cache[query] = vec
        return vec

    @property
    def dim(self) -> int:
        return cfg.embedding.dim
