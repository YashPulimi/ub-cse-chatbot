"""
app.py — UB CSE Conversational Chatbot
========================================
Pipeline:
  1. Hybrid retrieval     — BM25 + dense + RRF fusion
  2. Cross-encoder rerank — ms-marco-MiniLM-L-6-v2
  3. Neo4j KG lookup      — structured facts for faculty/course queries
  4. Local LLM generation — Ollama (con'
  
  versational, warm tone)
  5. Sliding window memory— last 5 turns
  6. Guardrails           — blocks off-topic queries
  7. Chainlit UI          — streaming, source citations

HOW TO RUN:
  chainlit run app.py
"""

import asyncio
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import chainlit as cl
import chromadb
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from neo4j import GraphDatabase
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

log = logging.getLogger("app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_DIR     = Path("data/chroma")
COLLECTION     = "ub_cse"
EMBED_MODEL    = "nomic-embed-text"
# LLM_MODEL      = "qwen2.5:3b"
LLM_MODEL = "llama3.2:3b"
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password123"
RERANK_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MEMORY_K       = 5
TOP_K          = 20
FINAL_TOP_N    = 5

# ── BM25 ──────────────────────────────────────────────────────────────────────

class BM25Index:
    def __init__(self):
        self.ids = []
        self.index = None

    def build(self, chunks: list[dict]):
        tokenize = lambda t: re.sub(r"[^\w\s]", " ", t.lower()).split()
        self.ids = [c["id"] for c in chunks]
        self.index = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, query: str, top_k=10) -> list[tuple[str, float]]:
        tokens = re.sub(r"[^\w\s]", " ", query.lower()).split()
        scores = self.index.get_scores(tokens)
        ranked = sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True)
        return [(cid, float(s)) for cid, s in ranked[:top_k] if s > 0]


# ── Hybrid Retriever ──────────────────────────────────────────────────────────

class HybridRetriever:
    """BM25 + dense search + RRF + cross-encoder reranking."""

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_collection(COLLECTION)
        self.embedder = OllamaEmbeddings(model=EMBED_MODEL)

        all_chunks = self._load_chunks()
        self.bm25 = BM25Index()
        self.bm25.build(all_chunks)
        self.chunk_map = {c["id"]: c for c in all_chunks}
        self.reranker = CrossEncoder(RERANK_MODEL, max_length=512)

        log.info(f"Retriever ready — {self.collection.count()} chunks")

    def _load_chunks(self) -> list[dict]:
        r = self.collection.get(include=["documents", "metadatas"])
        return [
            {"id": cid, "text": doc, **meta}
            for cid, doc, meta in zip(r["ids"], r["documents"], r["metadatas"])
        ]

    def search(self, query: str, top_n=FINAL_TOP_N) -> list[dict]:
        vec = self.embedder.embed_query(query)
        dense = self.collection.query(
            query_embeddings=[vec], n_results=TOP_K, include=["distances"]
        )
        dense_hits = list(zip(dense["ids"][0], dense["distances"][0]))

        bm25_hits = self.bm25.search(query, top_k=TOP_K)

        scores: dict = defaultdict(float)
        for rank, (cid, _) in enumerate(dense_hits, 1):
            scores[cid] += 0.6 / (60 + rank)
        for rank, (cid, _) in enumerate(bm25_hits, 1):
            scores[cid] += 0.4 / (60 + rank)
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        candidates = [
            {**self.chunk_map[cid], "rrf_score": round(score, 5)}
            for cid, score in fused[:TOP_K]
            if cid in self.chunk_map
        ]

        if candidates:
            pairs = [(query, c["text"]) for c in candidates]
            rscores = self.reranker.predict(pairs)
            ranked = sorted(zip(candidates, rscores), key=lambda x: x[1], reverse=True)
            for chunk, score in ranked:
                chunk["rerank_score"] = round(float(score), 4)
            candidates = [c for c, _ in ranked[:top_n]]

        return candidates


# ── Knowledge Graph ───────────────────────────────────────────────────────────

class KnowledgeGraph:
    """Query Neo4j for structured facts to augment LLM context."""

    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            self.driver.verify_connectivity()
            self.available = True
            log.info("Neo4j connected")
        except Exception as e:
            log.warning(f"Neo4j unavailable: {e}")
            self.available = False

    def _run(self, cypher: str, **params) -> list[dict]:
        if not self.available:
            return []
        try:
            with self.driver.session() as s:
                return [dict(r) for r in s.run(cypher, **params)]
        except Exception:
            return []

    def get_facts(self, query: str) -> str:
        facts = []
        q = query.lower()

        for num in re.findall(r"cse\s*(\d{3}[a-z]?)", q, re.IGNORECASE):
            code = f"CSE {num.upper()}"

            rows = self._run("""
                MATCH (p:Professor)-[:TEACHES]->(c:Course {code:$code})
                RETURN p.name AS name, p.email AS email, p.office AS office, c.title AS title
            """, code=code)
            for r in rows:
                facts.append(
                    f"{r['name']} teaches {code} ({r.get('title', '')})."
                    + (f" Email: {r['email']}." if r.get("email") else "")
                    + (f" Office: {r['office']}." if r.get("office") else "")
                )

            rows = self._run("""
                MATCH (pre:Course)-[:PREREQ_FOR]->(c:Course {code:$code})
                RETURN pre.code AS code, pre.title AS title
            """, code=code)
            for r in rows:
                facts.append(f"Prerequisite for {code}: {r['code']} {r.get('title', '')}")

        for pattern in [
            r"professor\s+(\w+)",
            r"dr\.?\s+(\w+)",
            r"(\w+)'s\s+(?:office|email|course)"
        ]:
            m = re.search(pattern, q)
            if m:
                rows = self._run("""
                    MATCH (p:Professor)
                    WHERE toLower(p.name) CONTAINS $name
                    OPTIONAL MATCH (p)-[:TEACHES]->(c:Course)
                    RETURN p.name AS name, p.email AS email,
                           p.office AS office, p.rank AS rank,
                           collect(c.code) AS courses
                    LIMIT 3
                """, name=m.group(1).lower())
                for r in rows:
                    facts.append(
                        f"{r['name']} ({r.get('rank', 'Professor')}). "
                        f"Email: {r.get('email', 'N/A')}. "
                        f"Office: {r.get('office', 'N/A')}. "
                        f"Courses: {', '.join(r['courses']) if r['courses'] else 'N/A'}."
                    )

        area_m = re.search(
            r"(machine learning|deep learning|computer vision|nlp|"
            r"natural language|security|database|network|algorithm|ai\b)",
            q,
            re.IGNORECASE
        )
        if area_m and any(kw in q for kw in ["who", "research", "professor", "faculty", "work"]):
            rows = self._run("""
                MATCH (p:Professor)-[:WORKS_IN]->(r:ResearchArea)
                WHERE toLower(r.name) CONTAINS $area
                RETURN p.name AS name LIMIT 8
            """, area=area_m.group(1).lower())
            if rows:
                names = [r["name"] for r in rows]
                facts.append(f"Faculty working on {area_m.group(1)}: {', '.join(names)}.")

        return "\n".join(facts)


# ── Guardrail ─────────────────────────────────────────────────────────────────

class Guardrail:
    CSE_KEYWORDS = {
        "cse", "course", "class", "program", "degree", "professor",
        "faculty", "research", "lab", "graduate", "undergraduate",
        "admission", "deadline", "credit", "gpa", "phd", "ms", "master",
        "thesis", "capstone", "syllabus", "prerequisite", "requirement",
        "buffalo", "ub", "engineering", "computer science", "algorithm",
        "machine learning", "ai", "security", "davis hall", "capen hall",
    }
    HARMFUL = [
        r"ignore.{0,20}instruction", r"jailbreak",
        r"forget.{0,20}instruction", r"ignore previous",
    ]
    OUT_OF_SCOPE = [
        r"\bpizza\b", r"\bweather\b", r"\bsports\b",
        r"\bpolitics\b", r"\bjoke\b",
    ]

    def check(self, query: str) -> str:
        q = query.lower()
        for p in self.HARMFUL:
            if re.search(p, q):
                return "HARMFUL"
        for p in self.OUT_OF_SCOPE:
            if re.search(p, q):
                return "OUT_SCOPE"
        if any(kw in q for kw in self.CSE_KEYWORDS):
            return "IN_SCOPE"
        if len(query.split()) <= 5:
            return "IN_SCOPE"
        return "IN_SCOPE"


# ── Generator ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a warm, helpful assistant for UB's Computer Science
and Engineering department. You're like a knowledgeable friend — approachable,
positive, and genuinely excited to help students navigate UB CSE.

Guidelines:
- Be conversational and encouraging, not robotic
- Answer only from the provided context and knowledge graph facts
- If you don't have the info, say so warmly and point to the official site
- Keep answers clear and concise
- For deadlines and requirements, remind students to verify on the official site
- Knowledge graph facts are highly reliable — prioritize them for faculty/course info"""


class Generator:
    def __init__(self):
        self.llm = OllamaLLM(model=LLM_MODEL, temperature=0.3)
        self.history: list[dict] = []

    def generate(self, query: str, chunks: list[dict], kg_facts: str) -> str:
        context = "\n\n---\n\n".join(
            f"[{i}] {c.get('title', '')}\n{c['text']}"
            for i, c in enumerate(chunks, 1)
        )
        history = "".join(
            f"{'Student' if t['role'] == 'user' else 'Assistant'}: {t['content']}\n"
            for t in self.history[-(MEMORY_K * 2):]
        )

        conv_part = f"Conversation so far:\n{history}\n" if history else ""
        kg_part = f"Knowledge Graph Facts (reliable):\n{kg_facts}\n\n" if kg_facts else ""

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"{conv_part}"
            f"{kg_part}"
            f"Context:\n{context}\n\n"
            f"Student: {query}\n"
            f"Assistant:"
        )

        answer = self.llm.invoke(prompt).strip()
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def reset(self):
        self.history = []


# ── Globals ───────────────────────────────────────────────────────────────────

_retriever: HybridRetriever | None = None
_generator: Generator | None = None
_kg: KnowledgeGraph | None = None
_guardrail = Guardrail()


def get_retriever():
    global _retriever
    if not _retriever:
        _retriever = HybridRetriever()
    return _retriever


def get_generator():
    global _generator
    if not _generator:
        _generator = Generator()
    return _generator


def get_kg():
    global _kg
    if not _kg:
        _kg = KnowledgeGraph()
    return _kg


# ── Chainlit UI ───────────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="⏳ Starting up...").send()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_retriever)
    await loop.run_in_executor(None, get_generator)
    await loop.run_in_executor(None, get_kg)
    get_generator().reset()

    await cl.Message(content=(
        "👋 Hey! I'm your **UB CSE Assistant** — here to help you navigate "
        "everything about the CS & Engineering department at Buffalo!\n\n"
        "I can help with:\n"
        "- 📚 MS, PhD, and undergraduate programs\n"
        "- 📅 Admissions deadlines and requirements\n"
        "- 👨‍🏫 Faculty info, office hours, and research\n"
        "- 📖 Courses, prerequisites, and credit requirements\n\n"
        "What can I help you with today? 😊"
    )).send()


@cl.on_message
async def on_message(message: cl.Message):
    query = message.content.strip()
    label = _guardrail.check(query)

    if label == "HARMFUL":
        await cl.Message(
            content="That's not something I can help with! Ask me anything about UB CSE instead 😊"
        ).send()
        return

    if label == "OUT_SCOPE":
        await cl.Message(
            content="Ha, I wish I could help with that! I'm strictly a UB CSE expert — ask me about courses, faculty, or programs!"
        ).send()
        return

    loop = asyncio.get_event_loop()

    kg = get_kg()
    kg_facts = await loop.run_in_executor(None, kg.get_facts, query)

    async with cl.Step(name="Searching knowledge base...") as step:
        chunks = await loop.run_in_executor(None, get_retriever().search, query)
        step.output = "\n".join(
            f"{c.get('page_type', '?')} | rerank={c.get('rerank_score', '?')}"
            for c in chunks[:3]
        )

    answer = await loop.run_in_executor(
        None, get_generator().generate, query, chunks, kg_facts
    )

    elements = []
    if kg_facts:
        elements.append(
            cl.Text(
                name=" Knowledge Graph",
                content=f"**Facts:**\n{kg_facts}",
                display="inline"
            )
        )

    sources = list({c.get("url", "") for c in chunks if c.get("url")})[:3]
    if sources:
        elements.append(
            cl.Text(
                name="🔗 Sources",
                content="**Sources:**\n" + "\n".join(f"- [{u}]({u})" for u in sources),
                display="inline",
            )
        )

    await cl.Message(content=answer, elements=elements).send()


@cl.on_chat_end
async def on_chat_end():
    if _generator:
        _generator.reset()