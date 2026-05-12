"""
server.py — FastAPI backend for UB CSE Chatbot
================================================
Replaces app.py (Chainlit). Serves the custom UB-branded UI and
exposes a streaming SSE endpoint for the chat pipeline.

Run:
    pip install fastapi uvicorn[standard]
    python server.py
    # opens at http://localhost:8000

Endpoints:
    GET  /              → serves index.html
    POST /chat          → streaming SSE: guardrails → retrieve → rerank → generate
    POST /reset         → clear session memory
    GET  /health        → status check
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from config import cfg
from generator import get_generator
from guardrails import get_guardrails
from memory import get_memory, clear_memory
from reranker import get_reranker, Reranker
from retriever import get_retriever
from utils import get_logger

log = get_logger(__name__)
app = FastAPI(title="UB CSE Assistant", version="1.0")

# Serve static files from public/ OR static/ dir
_public = Path(__file__).parent / "public"
_static = Path(__file__).parent / "static"
if _static.exists():
    app.mount("/static", StaticFiles(directory=str(_static)), name="static")
elif _public.exists():
    app.mount("/static", StaticFiles(directory=str(_public)), name="static")

# Pre-warm on startup
@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_retriever)
    await loop.run_in_executor(None, get_reranker)
    await loop.run_in_executor(None, get_guardrails)
    get_generator()
    log.info("All components pre-warmed")


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {"status": "ok", "model": cfg.llm.model}


@app.post("/reset")
async def reset_session(request: Request):
    body = await request.json()
    sid  = body.get("session_id", "")
    if sid:
        clear_memory(sid)
    return {"ok": True}


@app.post("/chat")
async def chat(request: Request):
    body       = await request.json()
    query      = body.get("message", "").strip()
    session_id = body.get("session_id") or str(uuid.uuid4())

    if not query:
        return JSONResponse({"error": "empty message"}, status_code=400)

    return StreamingResponse(
        _stream_pipeline(query, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


async def _stream_pipeline(query: str, session_id: str):
    """
    Yields SSE events:
      data: {"type": "token",    "content": "..."}
      data: {"type": "sources",  "content": [...]}
      data: {"type": "debug",    "content": {...}}
      data: {"type": "latency",  "ttft": N, "total": N, "rerank": N}
      data: {"type": "done"}
      data: {"type": "error",    "content": "..."}
    """
    def sse(obj: dict) -> str:
        return f"data: {json.dumps(obj)}\n\n"

    t_start = time.perf_counter()
    mem     = get_memory(session_id)

    try:
        # ── Personalization ───────────────────────────────────────────────────
        resolved = mem.process_user_turn(query)

        if resolved == "__PERSONALIZE_YES__":
            mem.add_assistant_turn("Personalization on.")
            yield sse({"type": "token", "content": "Great! Tell me your program (MS/PhD/BS) and areas of interest."})
            yield sse({"type": "done"})
            return

        if resolved == "__PERSONALIZE_NO__":
            mem.add_assistant_turn("Continuing without personalization.")
            yield sse({"type": "token", "content": "No problem! Ask me anything about UB CSE."})
            yield sse({"type": "done"})
            return

        # ── Guardrails ────────────────────────────────────────────────────────
        decision = await get_guardrails().check(resolved)

        if decision.small_talk:
            mem.add_assistant_turn(decision.canned_reply)
            yield sse({"type": "token", "content": decision.canned_reply})
            yield sse({"type": "done"})
            return

        if not decision.allowed:
            mem.add_assistant_turn(decision.redirect_msg)
            yield sse({"type": "token", "content": decision.redirect_msg})
            yield sse({"type": "done"})
            return

        # ── Retrieval ─────────────────────────────────────────────────────────
        try:
            candidates, timings = await get_retriever().retrieve_with_latency(resolved)
        except Exception as e:
            log.error("Retrieval error: %s", e, exc_info=True)
            candidates, timings = [], {}

        # ── Reranking ─────────────────────────────────────────────────────────
        t_rr = time.perf_counter()
        try:
            ranked = await get_reranker().rerank(resolved, candidates)
        except Exception as e:
            log.error("Rerank error: %s", e, exc_info=True)
            ranked = candidates[:cfg.reranker.top_k]
        rerank_ms     = round((time.perf_counter() - t_rr) * 1000, 1)
        score_summary = Reranker.score_summary(ranked)

        # ── Send debug panel data ─────────────────────────────────────────────
        text_chunks  = [r for r in ranked if r.get("source") != "graph"]
        graph_chunks = [r for r in ranked if r.get("source") == "graph"]

        debug_payload = {
            "timings":      timings,
            "rerank_ms":    rerank_ms,
            "guardrail":    {"reason": decision.reason, "latency_ms": decision.latency_ms},
            "scores":       score_summary,
            "chunks": [
                {
                    "rank":      i + 1,
                    "text":      (r.get("text") or "")[:800],
                    "ce_score":  r.get("ce_score", None),
                    "rrf_score": r.get("rrf_score", r.get("score", 0)),
                    "source":    r.get("source", "?"),
                    "page_type": r.get("metadata", {}).get("page_type", "?"),
                    "url":       r.get("metadata", {}).get("url", ""),
                }
                for i, r in enumerate(text_chunks[:5])
            ],
            "graph": [r.get("text", "") for r in graph_chunks],
        }
        yield sse({"type": "debug", "content": debug_payload})

        # ── Stream tokens ─────────────────────────────────────────────────────
        history   = mem.build_prompt_context()
        generator = get_generator()

        full_resp = []
        ttft_ms   = None
        t_gen     = time.perf_counter()

        async for token in generator.stream(resolved, ranked, history):
            if ttft_ms is None:
                ttft_ms = round((time.perf_counter() - t_gen) * 1000, 1)
            full_resp.append(token)
            yield sse({"type": "token", "content": token})

        total_ms = round((time.perf_counter() - t_start) * 1000, 1)

        # ── Sources ───────────────────────────────────────────────────────────
        sources = []
        seen    = set()
        for r in ranked:
            url = r.get("metadata", {}).get("url", "")
            if url and url not in seen:
                seen.add(url)
                sources.append(url)
        yield sse({"type": "sources", "content": sources[:5]})

        # ── Latency ───────────────────────────────────────────────────────────
        yield sse({"type": "latency", "ttft": ttft_ms or 0,
                   "total": total_ms, "rerank": rerank_ms})
        yield sse({"type": "done"})

        # ── Memory ────────────────────────────────────────────────────────────
        mem.add_assistant_turn("".join(full_resp))
        log.info("Done: %.0f ms | %s", total_ms, session_id)

        # ── Personalization prompt ────────────────────────────────────────────
        if mem.should_ask_personalize():
            yield sse({"type": "personalize", "content": mem.get_personalize_prompt()})

    except Exception as e:
        log.error("Pipeline error: %s", e, exc_info=True)
        yield sse({"type": "error", "content": str(e)})
        yield sse({"type": "done"})


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)