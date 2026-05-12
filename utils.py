"""
utils.py — Shared helpers for UB CSE Chatbot
=============================================
Minimal glue: logging setup, timing decorator, content hashing,
course-code normalization, and simple file I/O helpers.

Rules:
- No business logic here — only generic utilities.
- Regex is limited to deterministic normalization (course codes, emails, URLs).
- Everything else (entity extraction, chunking, retrieval) lives in its own module.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from config import cfg

F = TypeVar("F", bound=Callable[..., Any])


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Return a logger that writes to both console and a rotating file.
    Call once at the top of each module:
        log = get_logger(__name__)
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler — one log file per module name
    log_file = cfg.paths.logs / f"{name.split('.')[-1]}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── Timing decorators ─────────────────────────────────────────────────────────

def timed(label: str | None = None) -> Callable[[F], F]:
    """
    Decorator that logs wall-clock time for sync functions.

    Usage:
        @timed("embed_batch")
        def embed(texts): ...
    """
    def decorator(fn: F) -> F:
        name = label or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log = get_logger(fn.__module__)
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = (time.perf_counter() - t0) * 1000
            log.info("⏱  %s took %.1f ms", name, elapsed)
            return result

        return wrapper  # type: ignore[return-value]
    return decorator


def async_timed(label: str | None = None) -> Callable[[F], F]:
    """
    Decorator that logs wall-clock time for async functions.

    Usage:
        @async_timed("qdrant_search")
        async def search(query): ...
    """
    def decorator(fn: F) -> F:
        name = label or fn.__name__

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            log = get_logger(fn.__module__)
            t0 = time.perf_counter()
            result = await fn(*args, **kwargs)
            elapsed = (time.perf_counter() - t0) * 1000
            log.info("⏱  %s took %.1f ms", name, elapsed)
            return result

        return wrapper  # type: ignore[return-value]
    return decorator


class LatencyTracker:
    """
    Lightweight context-manager for per-request latency breakdowns.

    Usage:
        tracker = LatencyTracker()
        with tracker.step("retrieval"):
            results = await retrieve(query)
        with tracker.step("rerank"):
            ranked = rerank(results)
        print(tracker.report())
    """

    def __init__(self) -> None:
        self._steps: dict[str, float] = {}
        self._start: dict[str, float] = {}
        self.total_start = time.perf_counter()

    def step(self, name: str) -> "LatencyTracker":
        self._current = name
        return self

    def __enter__(self) -> "LatencyTracker":
        self._start[self._current] = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        elapsed = (time.perf_counter() - self._start[self._current]) * 1000
        self._steps[self._current] = round(elapsed, 1)

    def report(self) -> dict[str, float]:
        total = round((time.perf_counter() - self.total_start) * 1000, 1)
        return {**self._steps, "total_ms": total}


# ── Content hashing ───────────────────────────────────────────────────────────

def content_hash(text: str) -> str:
    """MD5 of normalized text — used for deduplication across crawl runs."""
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


# ── Course-code normalization (only deterministic regex allowed) ──────────────

_COURSE_CODE_RE = re.compile(
    r"\b(CSE|cse)\s*(\d{3}[A-Za-z]?)\b"
)

def normalize_course_code(raw: str) -> str:
    """
    CSE574, cse 574, CSE-574, cse574A  →  CSE 574, CSE 574A
    This is the only regex normalization in the project.
    """
    def _fix(m: re.Match) -> str:
        return f"CSE {m.group(2).upper()}"
    return _COURSE_CODE_RE.sub(_fix, raw)


def extract_course_codes(text: str) -> list[str]:
    """Return sorted, deduplicated list of normalized course codes found in text."""
    codes = {normalize_course_code(m.group(0)) for m in _COURSE_CODE_RE.finditer(text)}
    return sorted(codes)


# ── URL helpers ───────────────────────────────────────────────────────────────

def normalize_url(url: str) -> str:
    """Strip trailing slash, fragment, and common tracking params."""
    from urllib.parse import urlparse, urlunparse, urlencode, parse_qs
    try:
        p = urlparse(url.strip())
        # Drop tracking params
        _DROP = {"utm_source", "utm_medium", "utm_campaign", "ref", "fbclid"}
        qs = {k: v for k, v in parse_qs(p.query).items() if k not in _DROP}
        clean_query = urlencode({k: v[0] for k, v in qs.items()}, doseq=False)
        return urlunparse((p.scheme, p.netloc, p.path.rstrip("/"), "", clean_query, ""))
    except Exception:
        return url


# ── File I/O ─────────────────────────────────────────────────────────────────

def load_jsonl(path: str | Path) -> list[dict]:
    """Load a .jsonl file into a list of dicts."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: list[dict], path: str | Path) -> None:
    """Save a list of dicts to a .jsonl file, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


# ── Async helpers ─────────────────────────────────────────────────────────────

async def run_in_executor(fn: Callable[..., Any], *args: Any) -> Any:
    """
    Run a blocking (sync) function in the default thread executor
    without blocking the async event loop.

    Usage:
        result = await run_in_executor(reranker.predict, pairs)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)


# ── Text helpers ──────────────────────────────────────────────────────────────

def truncate(text: str, max_chars: int = 300) -> str:
    """Truncate text for log display."""
    return text[:max_chars] + "…" if len(text) > max_chars else text


def word_count(text: str) -> int:
    return len(text.split())
