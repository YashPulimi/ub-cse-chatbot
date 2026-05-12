"""
graph_retriever.py — Real GraphRAG for UB CSE Chatbot
======================================================
Implements the full GraphRAG pipeline as described in the architecture:

  Query
    → Query Understanding (intent + entity extraction + entity linking)
    → Seed Node Lookup
    → k-Hop Subgraph Expansion
    → Path Ranking (by query intent)
    → Graph Evidence Construction (structured facts)
    → Graph-Guided Chunk Retrieval (chunks MENTIONING graph nodes)
    → Returns: graph_evidence_dicts + graph_guided_chunk_dicts

The graph_guided_chunk_dicts are injected into the RRF candidate pool
in retriever.py alongside dense and BM25 results, so the graph
actively shapes which text chunks reach the LLM — not just adds a
separate fact block.

Pipeline difference vs "KG fact injection":

  OLD (KG fact injection):
    Query → detect course code → fetch row → format text → send to LLM

  NEW (Real GraphRAG):
    Query → link entities → expand subgraph → rank paths →
    fetch chunks attached to subgraph nodes → merge into retrieval pool

Graph schema (from graph_store.py):
  Nodes: Course, Faculty, Program, ResearchArea, Lab, Topic
  Edges: TEACHES, REQUIRES, PART_OF, COVERS, RESEARCHES,
         AFFILIATED_WITH, LEADS, HAS_AREA

HOW TO TEST:
  python graph_retriever.py --query "CSE 574"
  python graph_retriever.py --query "who teaches NLP courses"
  python graph_retriever.py --query "faculty related to machine learning"
  python graph_retriever.py --query "MS program requirements"
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from config import cfg
from graph_store import get_graph_store, Neo4jGraphStore, NetworkXGraphStore
from utils import get_logger, extract_course_codes

log = get_logger(__name__)


# ── 1. Query Understanding ────────────────────────────────────────────────────

# Intent patterns
_INTENT_PATTERNS = [
    ("prereq",          re.compile(r"\b(prereq|pre-?req|required? (for|before)|need (for|before)|before taking)\b", re.I)),
    ("teaches",         re.compile(r"\b(who teach|instructor|professor|taught by|who is the (prof|instructor))\b", re.I)),
    ("faculty_research",re.compile(r"\b(research(es?| area| interest| topic)|work(ing)? on|specializ|expert|who work)\b", re.I)),
    ("course_info",     re.compile(r"\b(about|describe|what is|credits?|syllabus|overview)\b", re.I)),
    ("program_req",     re.compile(r"\b(ms\b|phd\b|bachelor|undergrad|degree req|core course|curriculum|elective|program)\b", re.I)),
    ("related_faculty", re.compile(r"\b(related (faculty|professor)|faculty (related|for)|suggest (faculty|professor))\b", re.I)),
    ("lab_research",    re.compile(r"\b(lab|center|group|institute|research lab)\b", re.I)),
]

# Entity patterns
_COURSE_RE   = re.compile(r"\bCSE\s*(\d{3}[A-Z]?)\b", re.I)
_FACULTY_RE  = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+)\b")
_PROGRAM_RE  = re.compile(r"\b(ms\b|m\.s\.|phd\b|ph\.d\.|bachelor|bs\b|undergrad(?:uate)?|master)\b", re.I)

# Research area vocabulary for entity linking
_AREA_VOCAB = {
    "nlp":                          "Natural Language Processing",
    "natural language":             "Natural Language Processing",
    "natural language processing":  "Natural Language Processing",
    "machine learning":             "Machine Learning",
    "ml":                           "Machine Learning",
    "deep learning":                "Machine Learning",
    "computer vision":              "Computer Vision",
    "cv":                           "Computer Vision",
    "image processing":             "Computer Vision",
    "databases":                    "Databases",
    "database":                     "Databases",
    "security":                     "Cybersecurity",
    "cybersecurity":                "Cybersecurity",
    "networks":                     "Computer Networks",
    "networking":                   "Computer Networks",
    "hci":                          "Human-Computer Interaction",
    "systems":                      "Computer Systems",
    "distributed":                  "Distributed Systems",
    "algorithms":                   "Algorithms & Theory",
    "theory":                       "Algorithms & Theory",
    "bioinformatics":               "Bioinformatics",
    "data mining":                  "Data Mining",
    "robotics":                     "Robotics",
    "software engineering":         "Software Engineering",
}


@dataclass
class QueryUnderstanding:
    """Structured output of query analysis."""
    raw:            str
    intents:        list[str]        = field(default_factory=list)
    course_codes:   list[str]        = field(default_factory=list)
    faculty_names:  list[str]        = field(default_factory=list)
    research_areas: list[str]        = field(default_factory=list)
    program_level:  str              = ""    # "ms" | "phd" | "bs" | ""
    seed_nodes:     list[dict]       = field(default_factory=list)  # {id, type}

    @property
    def has_entities(self) -> bool:
        return bool(
            self.course_codes or self.faculty_names
            or self.research_areas or self.program_level
        )


def understand_query(query: str) -> QueryUnderstanding:
    """
    Full query understanding: intent classification + entity extraction
    + entity linking to graph node types.
    """
    qu = QueryUnderstanding(raw=query)

    # Intent classification
    for intent_name, pattern in _INTENT_PATTERNS:
        if pattern.search(query):
            qu.intents.append(intent_name)

    # Course codes
    qu.course_codes = extract_course_codes(query)

    # Faculty names (2+ Title Case words, filter false positives)
    _STOPWORDS = {"University", "Buffalo", "Computer", "Science", "Engineering",
                  "Department", "School", "College", "Program", "Course"}
    raw_names = _FACULTY_RE.findall(query)
    qu.faculty_names = [
        n for n in raw_names
        if not any(w in _STOPWORDS for w in n.split())
        and len(n.split()) >= 2
    ]

    # Research area entity linking
    q_lower = query.lower()
    for phrase, canonical in _AREA_VOCAB.items():
        if phrase in q_lower and canonical not in qu.research_areas:
            qu.research_areas.append(canonical)

    # Program level
    pm = _PROGRAM_RE.search(query)
    if pm:
        raw = pm.group(1).lower()
        if raw in ("phd", "ph.d.", "doctoral"):
            qu.program_level = "phd"
        elif raw in ("ms", "m.s.", "master"):
            qu.program_level = "ms"
        elif raw in ("bs", "bachelor", "undergraduate", "undergrad"):
            qu.program_level = "bs"

    # Build seed node list (for graph lookup)
    for code in qu.course_codes:
        qu.seed_nodes.append({"id": code, "type": "Course"})
    for name in qu.faculty_names:
        qu.seed_nodes.append({"id": name.lower(), "type": "Faculty"})
    for area in qu.research_areas:
        qu.seed_nodes.append({"id": area.lower(), "type": "ResearchArea"})
    if qu.program_level:
        qu.seed_nodes.append({"id": qu.program_level, "type": "Program"})

    log.debug(
        "QueryUnderstanding: intents=%s courses=%s faculty=%s areas=%s program=%s",
        qu.intents, qu.course_codes, qu.faculty_names,
        qu.research_areas, qu.program_level,
    )
    return qu


# ── 2. Subgraph ───────────────────────────────────────────────────────────────

@dataclass
class SubgraphNode:
    id:    str
    type:  str        # Course | Faculty | ResearchArea | Program | Lab | Topic
    props: dict = field(default_factory=dict)
    hop:   int  = 0   # distance from seed

@dataclass
class SubgraphEdge:
    src: str
    rel: str
    dst: str

@dataclass
class Subgraph:
    nodes: dict[str, SubgraphNode] = field(default_factory=dict)
    edges: list[SubgraphEdge]      = field(default_factory=list)

    def add_node(self, nid: str, ntype: str, props: dict, hop: int) -> None:
        if nid not in self.nodes:
            self.nodes[nid] = SubgraphNode(id=nid, type=ntype, props=props, hop=hop)

    def add_edge(self, src: str, rel: str, dst: str) -> None:
        e = SubgraphEdge(src=src, rel=rel, dst=dst)
        if e not in self.edges:
            self.edges.append(e)

    @property
    def node_ids_by_type(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for nid, n in self.nodes.items():
            out[n.type].append(nid)
        return dict(out)


# ── 3. Path ranking ───────────────────────────────────────────────────────────

# Paths that match certain intents get priority
_PATH_SCORES: dict[str, dict[str, float]] = {
    "prereq":           {"REQUIRES": 3.0, "PART_OF": 1.0},
    "teaches":          {"TEACHES": 3.0, "AFFILIATED_WITH": 1.5},
    "faculty_research": {"RESEARCHES": 3.0, "AFFILIATED_WITH": 2.0, "LEADS": 1.5, "COVERS": 1.0},
    "course_info":      {"COVERS": 2.0, "PART_OF": 1.5, "REQUIRES": 1.5},
    "program_req":      {"PART_OF": 3.0, "HAS_AREA": 1.5},
    "related_faculty":  {"RESEARCHES": 3.0, "COVERS": 2.5, "TEACHES": 2.0},
    "lab_research":     {"LEADS": 3.0, "RESEARCHES": 2.0},
}

def score_path(rel: str, intents: list[str]) -> float:
    """Return relevance score for a relationship given the query intents."""
    if not intents:
        return 1.0
    total = 0.0
    for intent in intents:
        scores = _PATH_SCORES.get(intent, {})
        total += scores.get(rel, 0.5)
    return total / len(intents)


# ── 4. Neo4j subgraph expander ────────────────────────────────────────────────

class Neo4jSubgraphExpander:

    def __init__(self, store: Neo4jGraphStore) -> None:
        self._store = store

    def _run(self, cypher: str, **params) -> list[dict]:
        with self._store._driver.session(database=self._store._db) as s:
            return [dict(r) for r in s.run(cypher, **params)]

    def expand(self, qu: QueryUnderstanding, max_hops: int = 2) -> Subgraph:
        sg = Subgraph()
        t0 = time.perf_counter()

        # ── Seed node lookup ──────────────────────────────────────────────────
        for seed in qu.seed_nodes:
            sid  = seed["id"]
            stype = seed["type"]

            if stype == "Course":
                rows = self._run(
                    "MATCH (c:Course {code: $code}) RETURN c",
                    code=sid.upper()
                )
                for r in rows:
                    c = dict(r["c"])
                    sg.add_node(sid.upper(), "Course", c, hop=0)

            elif stype == "Faculty":
                rows = self._run(
                    "MATCH (f:Faculty) WHERE toLower(f.name) CONTAINS $name RETURN f LIMIT 3",
                    name=sid.lower()
                )
                for r in rows:
                    f = dict(r["f"])
                    sg.add_node(f.get("name", sid).lower(), "Faculty", f, hop=0)

            elif stype == "ResearchArea":
                rows = self._run(
                    "MATCH (r:ResearchArea) WHERE toLower(r.name) CONTAINS $name RETURN r LIMIT 3",
                    name=sid.lower()
                )
                for r in rows:
                    ra = dict(r["r"])
                    sg.add_node(ra.get("name", sid).lower(), "ResearchArea", ra, hop=0)

            elif stype == "Program":
                rows = self._run(
                    "MATCH (p:Program) WHERE toLower(p.level) = $level RETURN p LIMIT 5",
                    level=sid.lower()
                )
                for r in rows:
                    p = dict(r["p"])
                    sg.add_node(p.get("name", sid).lower(), "Program", p, hop=0)

        if not sg.nodes:
            log.debug("No seed nodes found in graph for query: %r", qu.raw)
            return sg

        # ── k-hop expansion ───────────────────────────────────────────────────
        for hop in range(1, max_hops + 1):
            current_nodes = [
                (nid, n) for nid, n in sg.nodes.items() if n.hop == hop - 1
            ]

            for nid, node in current_nodes:
                ntype = node.type

                if ntype == "Course":
                    self._expand_course(nid, sg, hop, qu.intents)
                elif ntype == "Faculty":
                    self._expand_faculty(nid, sg, hop, qu.intents)
                elif ntype == "ResearchArea":
                    self._expand_area(nid, sg, hop, qu.intents)
                elif ntype == "Program":
                    self._expand_program(nid, sg, hop, qu.intents)

        elapsed = (time.perf_counter() - t0) * 1000
        log.info(
            "Subgraph expanded: %d nodes, %d edges in %.1f ms",
            len(sg.nodes), len(sg.edges), elapsed
        )
        return sg

    def _expand_course(self, code: str, sg: Subgraph, hop: int, intents: list[str]) -> None:
        # Prerequisites
        if score_path("REQUIRES", intents) >= 1.0:
            rows = self._run(
                "MATCH (c:Course {code:$code})-[:REQUIRES]->(p:Course) RETURN p",
                code=code.upper()
            )
            for r in rows:
                p = dict(r["p"])
                pid = p.get("code", "")
                sg.add_node(pid, "Course", p, hop)
                sg.add_edge(code.upper(), "REQUIRES", pid)

        # Faculty who teach it
        if score_path("TEACHES", intents) >= 1.0:
            rows = self._run(
                "MATCH (f:Faculty)-[:TEACHES]->(c:Course {code:$code}) RETURN f",
                code=code.upper()
            )
            for r in rows:
                f = dict(r["f"])
                fid = f.get("name", "").lower()
                sg.add_node(fid, "Faculty", f, hop)
                sg.add_edge(fid, "TEACHES", code.upper())

        # Topics covered
        if score_path("COVERS", intents) >= 1.0:
            rows = self._run(
                "MATCH (c:Course {code:$code})-[:COVERS]->(t:Topic) RETURN t",
                code=code.upper()
            )
            for r in rows:
                t = dict(r["t"])
                tid = t.get("name", "").lower()
                sg.add_node(tid, "Topic", t, hop)
                sg.add_edge(code.upper(), "COVERS", tid)

        # Programs it belongs to
        if score_path("PART_OF", intents) >= 1.0:
            rows = self._run(
                "MATCH (c:Course {code:$code})-[:PART_OF]->(p:Program) RETURN p",
                code=code.upper()
            )
            for r in rows:
                p = dict(r["p"])
                pid = p.get("name", "").lower()
                sg.add_node(pid, "Program", p, hop)
                sg.add_edge(code.upper(), "PART_OF", pid)

    def _expand_faculty(self, fid: str, sg: Subgraph, hop: int, intents: list[str]) -> None:
        # Research areas
        if score_path("RESEARCHES", intents) >= 1.0:
            rows = self._run(
                "MATCH (f:Faculty)-[:RESEARCHES]->(r:ResearchArea) "
                "WHERE toLower(f.name) CONTAINS $name RETURN r",
                name=fid.lower()
            )
            for r in rows:
                ra = dict(r["r"])
                rid = ra.get("name", "").lower()
                sg.add_node(rid, "ResearchArea", ra, hop)
                sg.add_edge(fid, "RESEARCHES", rid)

        # Courses taught
        if score_path("TEACHES", intents) >= 1.0:
            rows = self._run(
                "MATCH (f:Faculty)-[:TEACHES]->(c:Course) "
                "WHERE toLower(f.name) CONTAINS $name RETURN c",
                name=fid.lower()
            )
            for r in rows:
                c = dict(r["c"])
                cid = c.get("code", "")
                sg.add_node(cid, "Course", c, hop)
                sg.add_edge(fid, "TEACHES", cid)

        # Labs led
        if score_path("LEADS", intents) >= 1.0:
            rows = self._run(
                "MATCH (f:Faculty)-[:LEADS]->(l:Lab) "
                "WHERE toLower(f.name) CONTAINS $name RETURN l",
                name=fid.lower()
            )
            for r in rows:
                lab = dict(r["l"])
                lid = lab.get("name", "").lower()
                sg.add_node(lid, "Lab", lab, hop)
                sg.add_edge(fid, "LEADS", lid)

    def _expand_area(self, aid: str, sg: Subgraph, hop: int, intents: list[str]) -> None:
        # Faculty researching this area
        if score_path("RESEARCHES", intents) >= 1.0:
            rows = self._run(
                "MATCH (f:Faculty)-[:RESEARCHES]->(r:ResearchArea) "
                "WHERE toLower(r.name) CONTAINS $name RETURN f",
                name=aid.lower()
            )
            for r in rows:
                f = dict(r["f"])
                fid = f.get("name", "").lower()
                sg.add_node(fid, "Faculty", f, hop)
                sg.add_edge(fid, "RESEARCHES", aid)

        # Topics in this area (via courses that cover them)
        rows = self._run(
            "MATCH (c:Course)-[:COVERS]->(t:Topic) "
            "WHERE toLower(t.name) CONTAINS $name RETURN c, t LIMIT 5",
            name=aid.lower()
        )
        for r in rows:
            c = dict(r["c"])
            t = dict(r["t"])
            cid = c.get("code", "")
            tid = t.get("name", "").lower()
            sg.add_node(cid, "Course", c, hop)
            sg.add_node(tid, "Topic", t, hop)
            sg.add_edge(cid, "COVERS", tid)

    def _expand_program(self, pid: str, sg: Subgraph, hop: int, intents: list[str]) -> None:
        rows = self._run(
            "MATCH (c:Course)-[:PART_OF]->(p:Program) "
            "WHERE toLower(p.level) = $level OR toLower(p.name) CONTAINS $name "
            "RETURN c LIMIT 20",
            level=pid.lower(), name=pid.lower()
        )
        for r in rows:
            c = dict(r["c"])
            cid = c.get("code", "")
            if cid:
                sg.add_node(cid, "Course", c, hop)
                sg.add_edge(cid, "PART_OF", pid)


# ── 5. NetworkX subgraph expander ─────────────────────────────────────────────

class NetworkXSubgraphExpander:

    def __init__(self, store: NetworkXGraphStore) -> None:
        self._G = store.G

    def _neighbors(self, nid: str, rel: str, direction: str = "out") -> list[tuple[str, dict]]:
        if nid not in self._G:
            return []
        edges = self._G.out_edges(nid, data=True) if direction == "out" else self._G.in_edges(nid, data=True)
        result = []
        for u, v, d in edges:
            if d.get("rel") == rel:
                neighbor = v if direction == "out" else u
                if neighbor in self._G:
                    result.append((neighbor, dict(self._G.nodes[neighbor])))
        return result

    def _fuzzy_nodes(self, fragment: str, label: str) -> list[tuple[str, dict]]:
        return [
            (nid, dict(self._G.nodes[nid]))
            for nid in self._G.nodes
            if self._G.nodes[nid].get("label") == label
            and fragment.lower() in nid.lower()
        ]

    def expand(self, qu: QueryUnderstanding, max_hops: int = 2) -> Subgraph:
        sg = Subgraph()
        t0 = time.perf_counter()

        # Seed lookup
        for seed in qu.seed_nodes:
            sid   = seed["id"]
            stype = seed["type"]

            if stype == "Course":
                nid = sid.upper()
                if nid in self._G:
                    sg.add_node(nid, "Course", dict(self._G.nodes[nid]), hop=0)
            else:
                label_map = {"Faculty": "Faculty", "ResearchArea": "ResearchArea",
                             "Program": "Program", "Lab": "Lab"}
                label = label_map.get(stype, stype)
                for nid, props in self._fuzzy_nodes(sid, label)[:3]:
                    sg.add_node(nid, stype, props, hop=0)

        if not sg.nodes:
            return sg

        # k-hop expansion
        all_rels = [
            ("REQUIRES", "out"), ("TEACHES", "in"), ("COVERS", "out"),
            ("PART_OF", "out"), ("RESEARCHES", "out"), ("LEADS", "out"),
            ("AFFILIATED_WITH", "out"),
        ]
        for hop in range(1, max_hops + 1):
            current = [(nid, n) for nid, n in sg.nodes.items() if n.hop == hop - 1]
            for nid, node in current:
                for rel, direction in all_rels:
                    if score_path(rel, qu.intents) < 0.8:
                        continue
                    for neighbor_id, neighbor_props in self._neighbors(nid, rel, direction):
                        label = neighbor_props.get("label", "Unknown")
                        sg.add_node(neighbor_id, label, neighbor_props, hop)
                        if direction == "out":
                            sg.add_edge(nid, rel, neighbor_id)
                        else:
                            sg.add_edge(neighbor_id, rel, nid)

        elapsed = (time.perf_counter() - t0) * 1000
        log.info("NetworkX subgraph: %d nodes, %d edges in %.1f ms",
                 len(sg.nodes), len(sg.edges), elapsed)
        return sg


# ── 6. Graph-guided chunk retrieval ──────────────────────────────────────────

def graph_guided_chunks(
    subgraph: Subgraph,
    top_k: int = 10,
) -> list[dict]:
    """
    Retrieve text chunks from Qdrant whose metadata mentions
    entities found in the subgraph. These are injected into
    the RRF candidate pool so the graph shapes text retrieval.

    Uses Qdrant metadata filters on:
      - courses_in_chunk  (course codes)
      - faculty_names     (faculty mentioned)
      - page_type         (for program/research nodes)
    """
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue
        from vector_store import get_qdrant_client
        from config import cfg as _cfg

        client     = get_qdrant_client()
        collection = _cfg.qdrant.collection
        by_type    = subgraph.node_ids_by_type

        all_results: list[dict] = []
        seen_ids: set[str]      = set()

        # ── Fetch chunks mentioning course codes ──────────────────────────────
        course_ids = [nid.upper() for nid in by_type.get("Course", [])[:6]]
        if course_ids:
            try:
                hits = client.scroll(
                    collection_name=collection,
                    scroll_filter=Filter(
                        should=[
                            FieldCondition(
                                key="courses_in_chunk",
                                match=MatchAny(any=course_ids),
                            )
                        ]
                    ),
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False,
                )[0]
                for hit in hits:
                    if hit.id not in seen_ids:
                        seen_ids.add(hit.id)
                        payload = hit.payload or {}
                        all_results.append({
                            "id":       str(hit.id),
                            "text":     payload.get("text", ""),
                            "score":    0.85,
                            "rrf_score": 0.85,
                            "metadata": payload,
                            "source":   "graph_chunk",
                        })
            except Exception as e:
                log.debug("graph_chunk course filter failed: %s", e)

        # ── Fetch chunks from faculty pages ───────────────────────────────────
        faculty_ids = by_type.get("Faculty", [])[:4]
        if faculty_ids:
            try:
                hits = client.scroll(
                    collection_name=collection,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="page_type",
                                match=MatchValue(value="faculty_profile"),
                            )
                        ]
                    ),
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False,
                )[0]
                for hit in hits:
                    payload = hit.payload or {}
                    # Check if this chunk mentions any of the faculty
                    text_lower = (payload.get("text", "") + payload.get("title", "")).lower()
                    if any(fid.split()[-1].lower() in text_lower for fid in faculty_ids):
                        if hit.id not in seen_ids:
                            seen_ids.add(hit.id)
                            all_results.append({
                                "id":       str(hit.id),
                                "text":     payload.get("text", ""),
                                "score":    0.8,
                                "rrf_score": 0.8,
                                "metadata": payload,
                                "source":   "graph_chunk",
                            })
            except Exception as e:
                log.debug("graph_chunk faculty filter failed: %s", e)

        # ── Fetch chunks from research pages (for area queries) ───────────────
        area_ids = by_type.get("ResearchArea", [])
        if area_ids:
            try:
                hits = client.scroll(
                    collection_name=collection,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="page_type",
                                match=MatchValue(value="research"),
                            )
                        ]
                    ),
                    limit=8,
                    with_payload=True,
                    with_vectors=False,
                )[0]
                for hit in hits:
                    if hit.id not in seen_ids:
                        seen_ids.add(hit.id)
                        payload = hit.payload or {}
                        all_results.append({
                            "id":       str(hit.id),
                            "text":     payload.get("text", ""),
                            "score":    0.75,
                            "rrf_score": 0.75,
                            "metadata": payload,
                            "source":   "graph_chunk",
                        })
            except Exception as e:
                log.debug("graph_chunk research filter failed: %s", e)

        log.info("Graph-guided chunks: %d results from Qdrant metadata filters", len(all_results))
        return all_results[:top_k]

    except Exception as e:
        log.warning("graph_guided_chunks failed: %s", e)
        return []


# ── 7. Graph evidence formatter ───────────────────────────────────────────────

def format_subgraph_evidence(sg: Subgraph, qu: QueryUnderstanding) -> str:
    """
    Format the subgraph into a structured evidence string for the LLM.
    Groups by node type, shows relationships, ranks by intent.
    """
    if not sg.nodes:
        return ""

    lines = ["[GRAPH EVIDENCE]"]
    by_type = sg.node_ids_by_type

    # Courses
    for cid in by_type.get("Course", [])[:5]:
        n = sg.nodes[cid]
        p = n.props
        title   = p.get("title", "")
        credits = p.get("credits", "")
        desc    = (p.get("description", "") or "")[:200]
        url     = p.get("url", "")
        line    = f"Course: {cid}"
        if title:   line += f" — {title}"
        if credits: line += f" ({credits} credits)"
        lines.append(line)
        if desc: lines.append(f"  Description: {desc}")
        if url:  lines.append(f"  Source: {url}")

        # Show relationships for this course
        course_edges = [e for e in sg.edges if e.src == cid or e.dst == cid]
        for e in course_edges[:8]:
            if e.rel == "REQUIRES" and e.src == cid:
                req_node = sg.nodes.get(e.dst)
                req_title = req_node.props.get("title","") if req_node else ""
                lines.append(f"  Prerequisite: {e.dst}" + (f" ({req_title})" if req_title else ""))
            elif e.rel == "TEACHES" and e.dst == cid:
                fac_node = sg.nodes.get(e.src)
                if fac_node:
                    fp = fac_node.props
                    email = fp.get("email","")
                    lines.append(f"  Taught by: {fp.get('name', e.src)}" + (f" <{email}>" if email else ""))
            elif e.rel == "COVERS" and e.src == cid:
                lines.append(f"  Covers topic: {e.dst.title()}")
            elif e.rel == "PART_OF" and e.src == cid:
                prog_node = sg.nodes.get(e.dst)
                pname = prog_node.props.get("name","") if prog_node else e.dst
                lines.append(f"  Part of program: {pname}")

    # Faculty
    for fid in by_type.get("Faculty", [])[:4]:
        n = sg.nodes[fid]
        p = n.props
        name   = p.get("name", fid)
        rank   = p.get("rank", "")
        email  = p.get("email", "")
        office = p.get("office", "")
        url    = p.get("url", "")
        line   = f"Faculty: {name}"
        if rank:  line += f" ({rank})"
        lines.append(line)
        if email:  lines.append(f"  Email: {email}")
        if office: lines.append(f"  Office: {office}")
        if url:    lines.append(f"  Profile: {url}")

        # Research areas
        research_edges = [e for e in sg.edges if e.src == fid and e.rel in ("RESEARCHES","AFFILIATED_WITH")]
        if research_edges:
            areas = [e.dst.title() for e in research_edges[:4]]
            lines.append(f"  Research areas: {', '.join(areas)}")

        # Courses taught
        teaches_edges = [e for e in sg.edges if e.src == fid and e.rel == "TEACHES"]
        if teaches_edges:
            courses = [e.dst for e in teaches_edges[:5]]
            lines.append(f"  Teaches: {', '.join(courses)}")

    # Research Areas
    for aid in by_type.get("ResearchArea", [])[:3]:
        n = sg.nodes[aid]
        lines.append(f"Research Area: {n.props.get('name', aid).title()}")
        faculty_in_area = [e.src for e in sg.edges if e.dst == aid and e.rel == "RESEARCHES"]
        if faculty_in_area:
            fac_names = []
            for fid in faculty_in_area[:5]:
                fn = sg.nodes.get(fid)
                fac_names.append(fn.props.get("name", fid) if fn else fid)
            lines.append(f"  Faculty: {', '.join(fac_names)}")

    # Programs
    for pid in by_type.get("Program", [])[:2]:
        n = sg.nodes[pid]
        p = n.props
        lines.append(f"Program: {p.get('name', pid)} (level: {p.get('level','')})")
        courses_in_prog = [e.src for e in sg.edges if e.dst == pid and e.rel == "PART_OF"]
        if courses_in_prog:
            lines.append(f"  Courses: {', '.join(courses_in_prog[:8])}")

    lines.append("[/GRAPH EVIDENCE]")
    return "\n".join(lines)


# ── 8. Public GraphRAG retriever ──────────────────────────────────────────────

class GraphRAGRetriever:
    """
    Full GraphRAG pipeline:
      1. Query understanding (intent + entity linking)
      2. Seed node lookup + k-hop subgraph expansion
      3. Path ranking by intent
      4. Graph evidence construction (structured block for LLM)
      5. Graph-guided chunk retrieval (Qdrant metadata filters)

    Returns two lists:
      - graph_evidence_results: structured subgraph text (source="graph")
      - graph_chunk_results:    text chunks from graph-connected pages (source="graph_chunk")

    Both are injected into retriever.py's RRF fusion pool.
    """

    def __init__(self) -> None:
        self._store    = get_graph_store()
        self._backend  = "neo4j" if isinstance(self._store, Neo4jGraphStore) else "networkx"
        self._expander = (
            Neo4jSubgraphExpander(self._store)
            if self._backend == "neo4j"
            else NetworkXSubgraphExpander(self._store)
        )
        log.info("GraphRAGRetriever initialized (backend=%s)", self._backend)

    def retrieve(self, query: str) -> list[dict]:
        """
        Main entry point called by retriever.py.

        Returns combined list of:
          - graph evidence dict (source="graph", structured subgraph text)
          - graph-guided chunk dicts (source="graph_chunk", from Qdrant filters)
        """
        t0 = time.perf_counter()

        # Step 1: Query understanding
        qu = understand_query(query)
        if not qu.has_entities:
            log.debug("No graph-relevant entities detected — skipping GraphRAG")
            return []

        # Step 2: Subgraph expansion
        sg = self._expander.expand(qu, max_hops=cfg.retrieval.graph_hops)
        if not sg.nodes:
            log.debug("Empty subgraph — no matching nodes in graph")
            return []

        results: list[dict] = []

        # Step 3: Format structured graph evidence
        evidence_text = format_subgraph_evidence(sg, qu)
        if evidence_text:
            results.append({
                "id":       f"graph:subgraph:{hash(query) % 999999}",
                "text":     evidence_text,
                "score":    1.2,
                "rrf_score": 1.2,
                "metadata": {
                    "page_type":   "graph_evidence",
                    "graph_type":  "subgraph",
                    "n_nodes":     len(sg.nodes),
                    "n_edges":     len(sg.edges),
                    "intents":     json.dumps(qu.intents),
                    "source":      "graph",
                },
                "source": "graph",
            })

        # Step 4: Graph-guided chunk retrieval
        guided_chunks = graph_guided_chunks(sg, top_k=8)
        results.extend(guided_chunks)

        elapsed = (time.perf_counter() - t0) * 1000
        log.info(
            "GraphRAG: %d nodes, %d edges, %d evidence + %d guided_chunks in %.1f ms",
            len(sg.nodes), len(sg.edges), 1 if evidence_text else 0,
            len(guided_chunks), elapsed,
        )
        return results

    def close(self) -> None:
        if hasattr(self._store, "close"):
            self._store.close()


# ── Singleton ──────────────────────────────────────────────────────────────────

_retriever: GraphRAGRetriever | None = None

def get_graph_retriever() -> GraphRAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = GraphRAGRetriever()
    return _retriever


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="GraphRAG retriever for UB CSE Chatbot")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--hops",  type=int, default=2)
    args = parser.parse_args()

    gr = GraphRAGRetriever()

    print(f"\n🔍  Query: {args.query!r}\n")

    qu = understand_query(args.query)
    print(f"Intent:    {qu.intents}")
    print(f"Courses:   {qu.course_codes}")
    print(f"Faculty:   {qu.faculty_names}")
    print(f"Areas:     {qu.research_areas}")
    print(f"Program:   {qu.program_level}")
    print(f"Seeds:     {qu.seed_nodes}\n")

    sg = gr._expander.expand(qu, max_hops=args.hops)
    print(f"Subgraph:  {len(sg.nodes)} nodes, {len(sg.edges)} edges")
    print(f"Node types: {dict(sg.node_ids_by_type)}\n")

    results = gr.retrieve(args.query)
    for r in results:
        src = r.get("source","?")
        print(f"{'='*60}")
        print(f"[{src}] score={r['score']:.3f}")
        print(r["text"][:400])

    gr.close()


if __name__ == "__main__":
    main()