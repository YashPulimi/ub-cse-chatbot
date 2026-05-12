"""
generator.py — Grounded Response Generator for UB CSE Chatbot
=============================================================
Takes reranked chunks + graph evidence + conversation history
and generates a grounded, cited response via a local Ollama LLM.

Design:
  - Prompt is strictly grounded: LLM is instructed to answer ONLY
    from the provided context, never from general knowledge.
  - Graph evidence is injected as a separate structured block so the
    LLM can answer relational queries (prerequisites, taught-by) that
    chunks alone may not cover.
  - Streaming: tokens are yielded as an async generator so Chainlit
    can stream them to the UI token-by-token (Time to First Token < 2s).
  - Fallback: if no relevant context is found, returns a polite
    "not found" message rather than hallucinating.
  - Citations: source URLs are appended to the response so the user
    can verify the answer.

Prompt structure:
  ┌──────────────────────────────────────────────────────────┐
  │ SYSTEM                                                   │
  │   You are the UB CSE department assistant...             │
  │   Answer ONLY from the provided context.                 │
  │   If the answer is not in the context, say so.           │
  ├──────────────────────────────────────────────────────────┤
  │ CONTEXT                                                  │
  │   [GRAPH] structured entity evidence [/GRAPH]            │
  │   [1] chunk text  (url)                                  │
  │   [2] chunk text  (url)                                  │
  │   ...                                                    │
  ├──────────────────────────────────────────────────────────┤
  │ CONVERSATION HISTORY                                     │
  │   User: ...                                              │
  │   Assistant: ...                                         │
  ├──────────────────────────────────────────────────────────┤
  │ USER                                                     │
  │   {query}                                                │
  └──────────────────────────────────────────────────────────┘

HOW TO USE:
  from generator import get_generator
  gen = get_generator()

  # Streaming (for Chainlit):
  async for token in gen.stream(query, ranked_results, history):
      print(token, end="", flush=True)

  # Non-streaming (for evaluator):
  response = await gen.generate(query, ranked_results, history)

HOW TO TEST (requires Ollama running):
  python generator.py --query "Who teaches CSE 574?"
  python generator.py --query "What are the MS admission requirements?"
  python generator.py --query "What is the weather today?"   # out-of-scope
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator

import httpx

from config import cfg
from utils import get_logger, truncate

log = get_logger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a helpful assistant for UB's CSE department. Be conversational and concise — like a knowledgeable friend, not a textbook.

Rules:
- Answer directly. Never start with "Based on the information provided."
- Be concise: 2-4 sentences for simple questions, a short list for complex ones.
- Include the key facts: names, emails, office locations, course codes, credits.
- Use natural language. Contractions are fine (it's, she's, here's).
- If asked a follow-up, refer back naturally without repeating everything.
- If you don't know, say it briefly and suggest the department website.
- Never invent facts."""

_NO_CONTEXT_RESPONSE = (
    "Hmm, I couldn't find specific information about that in the UB CSE website data I have. "
    "You might want to check the department website directly at "
    "https://engineering.buffalo.edu/computer-science-engineering.html, "
    "or reach out to the department at cse-info@buffalo.edu. "
    "Is there something else about UB CSE I can help you with?"
)

# Minimum number of context chunks required to attempt an answer
_MIN_CONTEXT_CHUNKS = 1


# ── Context builder ───────────────────────────────────────────────────────────

def build_context_block(results: list[dict], max_chunks: int = 5) -> tuple[str, list[str]]:
    """
    Build the CONTEXT block injected into the LLM prompt.

    Args:
        results:    Reranked result dicts from reranker.py
        max_chunks: Max text chunks to include (graph results are separate)

    Returns:
        (context_string, source_urls)
        source_urls: ordered list of URLs for citation footer
    """
    graph_blocks: list[str] = []
    text_chunks:  list[str] = []
    source_urls:  list[str] = []

    chunk_idx = 1

    for result in results:
        source = result.get("source", "")
        text   = (result.get("text") or "").strip()
        meta   = result.get("metadata", {})

        if not text:
            continue

        if source == "graph":
            # Graph evidence: inject as-is (already formatted)
            graph_blocks.append(text)

        else:
            # Text chunk: numbered for citations
            if chunk_idx > max_chunks:
                continue
            url = meta.get("url", "")
            if url:
                source_urls.append(url)
                text_chunks.append(f"[{chunk_idx}] {text}\n(Source: {url})")
            else:
                text_chunks.append(f"[{chunk_idx}] {text}")
            chunk_idx += 1

    # Assemble context block
    parts: list[str] = []
    if graph_blocks:
        parts.append("\n\n".join(graph_blocks))
    if text_chunks:
        parts.append("\n\n".join(text_chunks))

    context = "\n\n".join(parts)
    return context, source_urls


def build_prompt(
    query:   str,
    context: str,
    history: str = "",
) -> str:
    """
    Universal prompt builder — works for Qwen2.5 and Llama3.2.
    Detects model from cfg and uses correct chat format.
    """
    model = cfg.llm.model.lower()
    is_llama = "llama" in model or "phi" in model

    ctx_block = ""
    if context.strip():
        ctx_block = f"\n\n[Context from UB CSE website:]\n{context}"

    hist_block = ""
    if history.strip():
        hist_block = f"\n\n[Conversation so far:]\n{history}"

    if is_llama:
        # Llama 3.2 format
        user_content = f"{hist_block}{ctx_block}\n\n{query}".strip()
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{_SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"{user_content}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )
    else:
        # Qwen2.5 im_start format
        parts = [f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>"]

        if history.strip():
            for line in history.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                if line.startswith("User:"):
                    parts.append(f"<|im_start|>user\n{line[5:].strip()}<|im_end|>")
                elif line.startswith("Assistant:"):
                    parts.append(f"<|im_start|>assistant\n{line[10:].strip()}<|im_end|>")

        user_content = f"{query}{ctx_block}".strip()
        parts.append(f"<|im_start|>user\n{user_content}<|im_end|>")
        parts.append("<|im_start|>assistant")
        return "\n".join(parts)


# ── Ollama client ─────────────────────────────────────────────────────────────

class OllamaClient:
    """
    Thin async wrapper around the Ollama /api/generate endpoint.
    Supports both streaming and non-streaming modes.
    """

    def __init__(self) -> None:
        self.base_url = cfg.llm.ollama_base_url
        self.model    = cfg.llm.model
        self.timeout  = httpx.Timeout(
            connect=5.0,
            read=120.0,   # streaming responses can be slow
            write=10.0,
            pool=5.0,
        )

    async def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """
        Stream tokens from Ollama as they are generated.
        Yields individual token strings.
        """
        payload = {
            "model":   self.model,
            "prompt":  prompt,
            "stream":  True,
            "options": {
                "temperature":  cfg.llm.temperature,
                "num_predict":  max(cfg.llm.max_new_tokens, 800),
                "num_ctx":      cfg.llm.context_window,
                "stop":         ["<|user|>", "<|system|>"],
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data  = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
        except httpx.ConnectError:
            log.error("Ollama not running at %s", self.base_url)
            yield "[Error: Ollama is not running. Start it with: ollama serve]"
        except Exception as e:
            log.error("Ollama stream error: %s", e)
            yield f"[Error generating response: {e}]"

    async def generate(self, prompt: str) -> str:
        """Non-streaming generation. Returns complete response string."""
        payload = {
            "model":   self.model,
            "prompt":  prompt,
            "stream":  False,
            "options": {
                "temperature": cfg.llm.temperature,
                "num_predict": cfg.llm.max_new_tokens,
                "num_ctx":     cfg.llm.context_window,
                "stop":        ["<|im_end|>", "<|im_start|>", "<|eot_id|>"],
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                r.raise_for_status()
                return r.json().get("response", "").strip()
        except httpx.ConnectError:
            return "[Error: Ollama is not running. Start it with: ollama serve]"
        except Exception as e:
            log.error("Ollama generate error: %s", e)
            return f"[Error: {e}]"


# ── Generator ─────────────────────────────────────────────────────────────────

class Generator:
    """
    Grounded response generator.
    Wraps OllamaClient with context building, fallback handling,
    and citation footer assembly.
    """

    _singleton: "Generator | None" = None

    def __init__(self) -> None:
        self._client = OllamaClient()

    @classmethod
    def instance(cls) -> "Generator":
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    # ── Streaming (for Chainlit) ───────────────────────────────────────────────

    async def stream(
        self,
        query:   str,
        results: list[dict],
        history: str = "",
    ) -> AsyncIterator[str]:
        """
        Stream a grounded response token by token.
        Yields str tokens ending with a citations footer.

        Usage in app.py:
            async for token in gen.stream(query, ranked, history):
                await msg.stream_token(token)
        """
        t0 = time.perf_counter()

        # Check for empty context — only fall back if truly nothing
        if not results:
            log.info("No context — returning fallback response")
            yield _NO_CONTEXT_RESPONSE
            return

        # Build context and prompt
        context, source_urls = build_context_block(results)
        prompt               = build_prompt(query, context, history)

        log.info(
            "Generating response | model=%s  context_chars=%d  history_chars=%d",
            cfg.llm.model, len(context), len(history),
        )
        # Log first 500 chars of context for debugging
        log.info("CONTEXT PREVIEW: %s", context[:500].replace("\n", " | "))

        # Stream tokens
        full_response = []
        ttft_logged   = False

        async for token in self._client.stream(prompt):
            if not ttft_logged:
                ttft = (time.perf_counter() - t0) * 1000
                log.info("Time to first token: %.0f ms", ttft)
                ttft_logged = True
            full_response.append(token)
            yield token

        # Sources are sent via SSE 'sources' event in server.py — not appended to stream

        total_ms = (time.perf_counter() - t0) * 1000
        response_text = "".join(full_response)

        # If model returned nothing, yield a fallback
        if not response_text.strip():
            log.warning("Empty response from LLM — yielding fallback")
            yield ("I found some relevant information but had trouble formulating a response. "
                   "Try rephrasing your question or ask me something more specific about UB CSE.")

        log.info(
            "Generation complete: %d tokens in %.0f ms",
            len(response_text.split()), total_ms,
        )

    # ── Non-streaming (for evaluator) ─────────────────────────────────────────

    async def generate(
        self,
        query:   str,
        results: list[dict],
        history: str = "",
    ) -> str:
        """
        Non-streaming generation. Returns complete response string.
        Used by evaluator.py and CLI tests.
        """
        if not results:
            return _NO_CONTEXT_RESPONSE

        context, source_urls = build_context_block(results)
        prompt               = build_prompt(query, context, history)

        t0       = time.perf_counter()
        response = await self._client.generate(prompt)
        elapsed  = (time.perf_counter() - t0) * 1000
        log.info("Generation (non-stream): %.0f ms", elapsed)

        # sources returned separately

        return response

    async def check_ollama(self) -> bool:
        """Returns True if Ollama is available."""
        return await self._client.is_available()


# ── Citation footer ───────────────────────────────────────────────────────────

def _build_citations_footer(urls: list[str]) -> str:
    if not urls:
        return ""
    lines = ["\n\n---\n**Sources:**"]
    for i, url in enumerate(urls, 1):
        lines.append(f"[{i}] {url}")
    return "\n".join(lines)


# ── Singleton accessor ────────────────────────────────────────────────────────

def get_generator() -> Generator:
    return Generator.instance()


# ── Entry point ───────────────────────────────────────────────────────────────

async def _run_test(query: str) -> None:
    from retriever import get_retriever
    from reranker  import get_reranker

    gen = Generator()

    # Check Ollama
    if not await gen.check_ollama():
        print("\n❌  Ollama is not running.")
        print("    Start it with:  ollama serve")
        print("    Then pull model: ollama pull qwen2.5:3b")
        return

    print(f"\n🔍  Query: {query!r}")

    # Retrieve + rerank
    retriever  = get_retriever()
    reranker   = get_reranker()
    candidates = retriever.retrieve_sync(query)
    ranked     = reranker.rerank_sync(query, candidates)

    print(f"    Context: {len(ranked)} results "
          f"({sum(1 for r in ranked if r.get('source')=='graph')} graph, "
          f"{sum(1 for r in ranked if r.get('source')!='graph')} chunks)\n")

    # Stream response
    print("─" * 60)
    t0 = time.perf_counter()
    async for token in gen.stream(query, ranked):
        print(token, end="", flush=True)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"\n{'─'*60}")
    print(f"\n⏱   {elapsed:.0f} ms total")
    print(f"\n    Next → python memory.py  (demo) or python app.py")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generator for UB CSE Chatbot")
    parser.add_argument("--query", type=str,
                        default="Who teaches CSE 574?",
                        help="Query to test")
    args = parser.parse_args()
    asyncio.run(_run_test(args.query))


if __name__ == "__main__":
    main()