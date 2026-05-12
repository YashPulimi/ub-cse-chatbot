"""
app.py — UB CSE Chatbot
========================
UB brand: #005bbb blue, white, Montserrat headers — matches engineering.buffalo.edu
Run:  python -m chainlit run app.py
"""

from __future__ import annotations
import time
import uuid

import chainlit as cl

from config import cfg
from generator import get_generator
from guardrails import get_guardrails
from memory import get_memory, clear_memory
from reranker import get_reranker, Reranker
from retriever import get_retriever
from utils import get_logger

log = get_logger(__name__)


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(label="Who teaches computer vision?",
                   message="Who teaches computer vision at UB CSE?", icon="👁️"),
        cl.Starter(label="MS degree requirements",
                   message="What are the MS CSE degree requirements?", icon="🎓"),
        cl.Starter(label="CSE 574 prerequisites",
                   message="What are the prerequisites for CSE 574?", icon="📚"),
        cl.Starter(label="PhD admissions GPA",
                   message="What GPA is required for the CSE PhD program?", icon="🔬"),
    ]


@cl.on_chat_start
async def on_chat_start():
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("message_count", 0)

    try:
        get_retriever(); get_reranker(); get_guardrails(); get_generator()
    except Exception as e:
        log.warning("Pre-warm failed: %s", e)

    await cl.Message(
        content=_welcome(),
        author="UB CSE Assistant",
    ).send()


@cl.on_chat_end
async def on_chat_end():
    sid = cl.user_session.get("session_id", "")
    if sid:
        clear_memory(sid)


def _welcome() -> str:
    return (
        "---\n"
        "### 🎓 Department of Computer Science and Engineering\n"
        "**University at Buffalo** — School of Engineering and Applied Sciences\n\n"
        "---\n\n"
        "Ask me anything about UB CSE:\n\n"
        "- **Courses** — prerequisites, credits, descriptions\n"
        "- **Faculty** — office hours, research areas, contact info\n"
        "- **Programs** — MS, PhD, BS requirements and specializations\n"
        "- **Research** — labs, research areas, faculty interests\n"
        "- **Admissions** — GPA requirements, deadlines, required documents\n\n"
        f"*Model: `{cfg.llm.model}`*"
    )


@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id", str(uuid.uuid4()))
    mem        = get_memory(session_id)
    t_start    = time.perf_counter()

    count = cl.user_session.get("message_count", 0) + 1
    cl.user_session.set("message_count", count)

    query = message.content.strip()
    if not query:
        return

    await _pipeline(query, mem, session_id, t_start)


async def _pipeline(query, mem, session_id, t_start):
    # ── Personalization ───────────────────────────────────────────────────────
    resolved = mem.process_user_turn(query)
    if resolved == "__PERSONALIZE_YES__":
        mem.add_assistant_turn("Personalization on.")
        await cl.Message(
            content="Great! Tell me your program (MS/PhD/BS) and interests.",
            author="UB CSE Assistant").send()
        return
    if resolved == "__PERSONALIZE_NO__":
        mem.add_assistant_turn("Continuing without personalization.")
        await cl.Message(
            content="No problem! Ask me anything about UB CSE.",
            author="UB CSE Assistant").send()
        return

    # ── Guardrails ────────────────────────────────────────────────────────────
    decision = await get_guardrails().check(resolved)
    if decision.small_talk:
        mem.add_assistant_turn(decision.canned_reply)
        await cl.Message(content=decision.canned_reply, author="UB CSE Assistant").send()
        return
    if not decision.allowed:
        mem.add_assistant_turn(decision.redirect_msg)
        await cl.Message(content=decision.redirect_msg, author="UB CSE Assistant").send()
        return

    # ── Retrieval ─────────────────────────────────────────────────────────────
    try:
        candidates, timings = await get_retriever().retrieve_with_latency(resolved)
    except Exception as e:
        log.error("Retrieval error: %s", e, exc_info=True)
        candidates, timings = [], {}

    # ── Reranking ─────────────────────────────────────────────────────────────
    t_rr = time.perf_counter()
    try:
        ranked = await get_reranker().rerank(resolved, candidates)
    except Exception as e:
        log.error("Rerank error: %s", e, exc_info=True)
        ranked = candidates[:cfg.reranker.top_k]
    rerank_ms     = round((time.perf_counter() - t_rr) * 1000, 1)
    score_summary = Reranker.score_summary(ranked)

    # ── Side panels ───────────────────────────────────────────────────────────
    elements = _build_elements(ranked, score_summary, timings, rerank_ms, decision)

    # ── Stream answer ─────────────────────────────────────────────────────────
    history = mem.build_prompt_context()
    msg     = cl.Message(content="", author="UB CSE Assistant", elements=elements)
    await msg.send()

    full_resp = []
    ttft_ms   = None
    t_gen     = time.perf_counter()

    async for token in get_generator().stream(resolved, ranked, history):
        if ttft_ms is None:
            ttft_ms = round((time.perf_counter() - t_gen) * 1000, 1)
        await msg.stream_token(token)
        full_resp.append(token)

    total_ms = round((time.perf_counter() - t_start) * 1000, 1)
    await msg.stream_token(_latency_footer(ttft_ms or 0, total_ms, rerank_ms))
    await msg.update()

    mem.add_assistant_turn("".join(full_resp))
    log.info("Done: %.0f ms | %s", total_ms, session_id)

    if mem.should_ask_personalize():
        await cl.Message(
            content=mem.get_personalize_prompt(),
            author="UB CSE Assistant").send()


def _build_elements(ranked, score_summary, timings, rerank_ms, decision):
    elements = []
    text_chunks  = [r for r in ranked if r.get("source") != "graph"]
    graph_chunks = [r for r in ranked if r.get("source") == "graph"]

    # ── Top chunks ────────────────────────────────────────────────────────────
    if text_chunks:
        lines = []
        for i, r in enumerate(text_chunks[:5], 1):
            meta   = r.get("metadata", {})
            url    = meta.get("url", "")
            ptype  = meta.get("page_type", "?")
            ce     = r.get("ce_score", "—")
            rrf    = r.get("rrf_score", r.get("score", 0))
            ce_str = f"{ce:.4f}" if isinstance(ce, float) else str(ce)
            text   = (r.get("text") or "").strip()

            lines.append(f"{'━'*58}")
            lines.append(f"Chunk {i}  ·  {ptype}  ·  CE={ce_str}  ·  RRF={rrf:.5f}")
            if url:
                lines.append(f"Source: {url}")
            lines.append("")
            lines.append(text[:800] + ("…" if len(text) > 800 else ""))
            lines.append("")

        elements.append(cl.Text(
            name="📄 Top Retrieved Chunks",
            content="\n".join(lines),
            display="side",
        ))

    # ── Graph evidence ────────────────────────────────────────────────────────
    if graph_chunks:
        elements.append(cl.Text(
            name="🕸️ Graph Evidence",
            content="\n\n".join(r.get("text", "") for r in graph_chunks),
            display="side",
        ))

    # ── Reranking + latency ───────────────────────────────────────────────────
    table = [
        "RERANKING SCORES",
        "─" * 62,
        f"  {'#':>2}  {'src':>6}  {'CE':>8}  {'RRF':>10}  "
        f"{'d#':>4}  {'b#':>4}  page_type",
        "  " + "─" * 56,
    ]
    for s in score_summary:
        ce  = f"{s['ce_score']:.4f}" if isinstance(s["ce_score"], float) else "   graph"
        rrf = f"{s['rrf_score']:.5f}"
        table.append(
            f"  {s['rank']:>2}  {s['source']:>6}  {ce:>8}  {rrf:>10}  "
            f"{str(s['dense_rank']):>4}  {str(s['bm25_rank']):>4}  {s['page_type'][:20]}"
        )
    table += [
        "",
        "LATENCY BREAKDOWN",
        "─" * 40,
        f"  Dense  : {timings.get('dense_ms',  0):.0f} ms  ({timings.get('n_dense', 0)} hits)",
        f"  BM25   : {timings.get('bm25_ms',   0):.0f} ms  ({timings.get('n_bm25',  0)} hits)",
        f"  Graph  : {timings.get('graph_ms',  0):.0f} ms  ({timings.get('n_graph', 0)} hits)",
        f"  Fusion : {timings.get('fusion_ms', 0):.0f} ms  → {timings.get('n_fused', 0)} fused",
        f"  Rerank : {rerank_ms:.0f} ms",
        f"  Guard  : {decision.latency_ms:.0f} ms  [{decision.reason}]",
    ]
    elements.append(cl.Text(
        name="📊 Reranking & Latency",
        content="\n".join(table),
        display="side",
    ))

    return elements


def _latency_footer(ttft_ms: float, total_ms: float, rerank_ms: float) -> str:
    return (
        f"\n\n---\n"
        f"⏱ **TTFT** {ttft_ms:.0f} ms  ·  "
        f"**Total** {total_ms:.0f} ms  ·  "
        f"**Rerank** {rerank_ms:.0f} ms"
    )