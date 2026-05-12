"""
graph_store.py — Knowledge Graph Builder for UB CSE Chatbot
============================================================
Extracts entities and relationships from ingested Documents and
populates a Neo4j property graph. Also provides a NetworkX fallback
for dev environments without Docker.

Graph Schema
────────────
Nodes:
  (:Course)       {code, title, credits, description, prereqs, url}
  (:Faculty)      {name, email, rank, office, phone, url}
  (:Program)      {name, level, url}               level = "ms"|"phd"|"bs"|"minor"
  (:ResearchArea) {name}
  (:Lab)          {name, url}
  (:Topic)        {name}                            NLP-extracted keywords

Relationships:
  (Faculty)-[:TEACHES]->(Course)
  (Faculty)-[:RESEARCHES]->(ResearchArea)
  (Faculty)-[:LEADS]->(Lab)
  (Course)-[:REQUIRES]->(Course)                   prerequisite edges
  (Course)-[:PART_OF]->(Program)
  (Program)-[:HAS_AREA]->(ResearchArea)
  (Course)-[:COVERS]->(Topic)
  (Faculty)-[:AFFILIATED_WITH]->(ResearchArea)

Why a Knowledge Graph here:
  - Relational queries like "which faculty teach courses in NLP?" require
    traversal — a vector index cannot join Faculty ↔ Course ↔ ResearchArea.
  - Prerequisite chains: "what do I need before CSE 574?" needs graph hops,
    not embedding similarity.
  - The bonus requirement (suggest related labs/faculty from a course query)
    is a 2-hop Cypher query: (Course)-[:COVERS]->(:Topic)<-[:RESEARCHES]-(Faculty).
  - GraphRAG: retrieved graph context is injected alongside vector chunks
    to let the LLM answer relational questions it otherwise hallucinates.

Extraction strategy:
  - Structured metadata from ingestion_pipeline.py is the primary source
    (course_codes, faculty_names, faculty_email, etc.) — no NLP needed.
  - Prerequisite text is regex-parsed from course description fields.
  - Research area / lab names are extracted from research page headings.
  - Topic extraction uses a lightweight keyword approach (no heavy NLP dep).

NetworkX fallback:
  - If NEO4J_URI is unreachable, the graph is saved as a .gpickle file.
  - graph_retriever.py detects which backend is available and uses it.
  - Cypher queries are translated to NetworkX traversals automatically.

HOW TO RUN:
  # With Neo4j running:
  python graph_store.py

  # Without Neo4j (NetworkX fallback):
  NEO4J_URI="" python graph_store.py

  # Force rebuild:
  python graph_store.py --rebuild

  # Quick stats:
  python graph_store.py --stats

  # Test a query:
  python graph_store.py --query "CSE 574"

  Next → python retriever.py
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from config import cfg
from ingestion_pipeline import IngestionPipeline
from utils import get_logger, extract_course_codes, load_json, save_json

log = get_logger(__name__)

# ── Prerequisite parser ───────────────────────────────────────────────────────

_PREREQ_RE = re.compile(
    r"(?:prerequisite[s]?|pre-?req[s]?|requires?)[:\s]+([^\n.;]{5,200})",
    re.IGNORECASE,
)
_COREQ_RE = re.compile(
    r"(?:corequisite[s]?|co-?req[s]?)[:\s]+([^\n.;]{5,200})",
    re.IGNORECASE,
)


def _parse_prereqs(text: str) -> list[str]:
    """Extract prerequisite course codes from a course description string."""
    prereq_codes: list[str] = []
    for m in _PREREQ_RE.finditer(text):
        prereq_codes.extend(extract_course_codes(m.group(1)))
    return sorted(set(prereq_codes))


def _parse_coreqs(text: str) -> list[str]:
    coreq_codes: list[str] = []
    for m in _COREQ_RE.finditer(text):
        coreq_codes.extend(extract_course_codes(m.group(1)))
    return sorted(set(coreq_codes))


# ── Research area / lab name normalization ────────────────────────────────────

_AREA_ALIASES: dict[str, str] = {
    "ai":                           "Artificial Intelligence",
    "artificial intelligence":      "Artificial Intelligence",
    "machine learning":             "Machine Learning",
    "ml":                           "Machine Learning",
    "nlp":                          "Natural Language Processing",
    "natural language processing":  "Natural Language Processing",
    "computer vision":              "Computer Vision",
    "cv":                           "Computer Vision",
    "databases":                    "Databases",
    "database systems":             "Databases",
    "networks":                     "Computer Networks",
    "computer networks":            "Computer Networks",
    "security":                     "Cybersecurity",
    "cybersecurity":                "Cybersecurity",
    "hci":                          "Human-Computer Interaction",
    "human-computer interaction":   "Human-Computer Interaction",
    "systems":                      "Computer Systems",
    "distributed systems":          "Distributed Systems",
    "bioinformatics":               "Bioinformatics",
    "data mining":                  "Data Mining",
    "algorithms":                   "Algorithms & Theory",
    "theory":                       "Algorithms & Theory",
    "robotics":                     "Robotics",
    "software engineering":         "Software Engineering",
    "programming languages":        "Programming Languages",
}


def _normalize_area(raw: str) -> str:
    key = raw.strip().lower()
    return _AREA_ALIASES.get(key, raw.strip().title())


# ── Keyword topic extraction (lightweight, no spaCy) ─────────────────────────

_TOPIC_STOP = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
    "is", "are", "be", "this", "that", "with", "from", "by", "as", "it",
    "not", "will", "can", "may", "such", "other", "also", "use", "used",
    "course", "students", "study", "topics", "include", "including",
    "department", "university", "buffalo", "cse", "ub",
}

_TOPIC_RE = re.compile(r"\b[A-Za-z][a-z]{3,}\b")


def _extract_topics(text: str, max_topics: int = 6) -> list[str]:
    """
    Extract candidate topic words from a course description.
    Returns the top-N most frequent non-stopword content words.
    Used to create (:Course)-[:COVERS]->(:Topic) edges.
    """
    words = _TOPIC_RE.findall(text.lower())
    freq: dict[str, int] = {}
    for w in words:
        if w not in _TOPIC_STOP and len(w) >= 4:
            freq[w] = freq.get(w, 0) + 1

    # Sort by frequency descending, return top N
    ranked = sorted(freq.items(), key=lambda x: -x[1])
    return [w.title() for w, _ in ranked[:max_topics]]


# ── Entity extraction from Documents ─────────────────────────────────────────

class EntityExtractor:
    """
    Extracts structured node and edge data from LlamaIndex Documents.
    Uses metadata-first extraction (no heavy NLP) — all structured fields
    were populated by ingestion_pipeline.py from the crawler's structured dicts.
    """

    def __init__(self) -> None:
        # Node stores: keyed by normalized ID
        self.courses:       dict[str, dict] = {}   # code → props
        self.faculty:       dict[str, dict] = {}   # name → props
        self.programs:      dict[str, dict] = {}   # name → props
        self.research_areas: dict[str, dict] = {}  # name → props
        self.labs:          dict[str, dict] = {}   # name → props
        self.topics:        dict[str, dict] = {}   # name → props

        # Edge stores: list of (from_id, to_id, rel_type, props)
        self.edges: list[tuple[str, str, str, dict]] = []

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _add_course(self, code: str, props: dict) -> None:
        code = code.upper().strip()
        if code not in self.courses:
            self.courses[code] = {"code": code, **props}
        else:
            # Merge: keep non-empty fields from both
            for k, v in props.items():
                if v and not self.courses[code].get(k):
                    self.courses[code][k] = v

    def _add_faculty(self, name: str, props: dict) -> None:
        key = name.strip().lower()
        if key not in self.faculty:
            self.faculty[key] = {"name": name.strip(), **props}
        else:
            for k, v in props.items():
                if v and not self.faculty[key].get(k):
                    self.faculty[key][k] = v

    def _add_program(self, name: str, level: str, url: str = "") -> None:
        key = name.strip().lower()
        if key not in self.programs:
            self.programs[key] = {"name": name.strip(), "level": level, "url": url}

    def _add_research_area(self, raw_name: str) -> str:
        name = _normalize_area(raw_name)
        key  = name.lower()
        if key not in self.research_areas:
            self.research_areas[key] = {"name": name}
        return name

    def _add_lab(self, name: str, url: str = "") -> None:
        key = name.strip().lower()
        if key not in self.labs:
            self.labs[key] = {"name": name.strip(), "url": url}

    def _add_topic(self, name: str) -> None:
        key = name.strip().lower()
        if key not in self.topics:
            self.topics[key] = {"name": name.strip()}

    def _add_edge(self, from_id: str, to_id: str, rel: str, props: dict | None = None) -> None:
        self.edges.append((from_id.strip(), to_id.strip(), rel, props or {}))

    # ── Document processors ────────────────────────────────────────────────────

    def process_course_doc(self, doc: Any) -> None:
        """Extract courses from a course-type Document."""
        meta     = doc.metadata
        url      = meta.get("url", "")
        text     = doc.text
        page_type = meta.get("page_type", "")

        # Course codes mentioned on this page
        codes_json = meta.get("courses_mentioned", "[]")
        try:
            codes = json.loads(codes_json) if isinstance(codes_json, str) else codes_json
        except (json.JSONDecodeError, TypeError):
            codes = []

        # Also scan the text directly
        text_codes = extract_course_codes(text)
        all_codes  = sorted(set(codes) | set(text_codes))

        # Try to parse structured course data from text blocks
        # Pattern: "CSE 574 - Machine Learning (3 credits)\nDescription..."
        course_block_re = re.compile(
            r"(CSE\s*\d{3}[A-Z]?)"           # code
            r"[^\n]{0,80}\n"                  # rest of title line
            r"((?:.+\n){0,15})",              # up to 15 lines of description
            re.MULTILINE,
        )

        found_in_text: set[str] = set()
        for m in course_block_re.finditer(text):
            code_raw   = m.group(1).replace(" ", " ").strip()
            # normalize spacing
            code_clean = re.sub(r"CSE\s*(\d{3}[A-Z]?)", r"CSE \1", code_raw, flags=re.I).upper()
            block      = m.group(0)
            found_in_text.add(code_clean)

            # Credits
            credits_m = re.search(r"(\d+)\s*credit", block, re.I)
            credits   = int(credits_m.group(1)) if credits_m else None

            # Title: first line after code
            title_m = re.search(
                r"CSE\s*\d{3}[A-Z]?\s*[-–:]\s*(.+?)(?:\(|$)",
                block.split("\n")[0], re.I,
            )
            title = title_m.group(1).strip() if title_m else ""

            prereqs = _parse_prereqs(block)
            topics  = _extract_topics(block)

            self._add_course(code_clean, {
                "title":       title,
                "credits":     credits,
                "description": block.strip()[:600],
                "prereqs":     json.dumps(prereqs),
                "url":         url,
            })
            for prereq in prereqs:
                self._add_course(prereq, {"url": ""})   # stub
                self._add_edge(code_clean, prereq, "REQUIRES")

            for topic_name in topics:
                self._add_topic(topic_name)
                self._add_edge(code_clean, topic_name, "COVERS")

        # Add stubs for any codes mentioned but not parsed as blocks
        for code in all_codes:
            if code not in found_in_text:
                self._add_course(code, {"url": url, "title": "", "description": ""})

    def process_faculty_doc(self, doc: Any) -> None:
        """Extract faculty entity and their edges from a faculty-type Document."""
        meta  = doc.metadata
        url   = meta.get("url", "")
        name  = meta.get("title", "").strip()

        if not name or len(name) < 3:
            # Try to extract name from faculty_names metadata
            try:
                names = json.loads(meta.get("faculty_names", "[]"))
                name  = names[0] if names else ""
            except (json.JSONDecodeError, TypeError):
                pass

        if not name:
            return

        props = {
            "email":   meta.get("faculty_email", ""),
            "rank":    meta.get("faculty_rank", ""),
            "office":  meta.get("faculty_office", ""),
            "phone":   meta.get("faculty_phone", ""),
            "url":     url,
        }
        self._add_faculty(name, props)
        fkey = name.strip().lower()

        # Research areas
        try:
            areas_raw = meta.get("faculty_research_areas", "[]")
            areas = json.loads(areas_raw) if isinstance(areas_raw, str) else areas_raw
        except (json.JSONDecodeError, TypeError):
            areas = []

        for area_raw in (areas or []):
            area_name = self._add_research_area(area_raw)
            self._add_edge(fkey, area_name.lower(), "RESEARCHES")
            self._add_edge(fkey, area_name.lower(), "AFFILIATED_WITH")

        # Courses mentioned on faculty page → TEACHES edges
        try:
            codes_raw = meta.get("courses_mentioned", "[]")
            codes = json.loads(codes_raw) if isinstance(codes_raw, str) else codes_raw
        except (json.JSONDecodeError, TypeError):
            codes = []

        text_codes = extract_course_codes(doc.text)
        all_codes  = sorted(set(codes) | set(text_codes))

        for code in all_codes:
            self._add_course(code, {"url": ""})
            self._add_edge(fkey, code, "TEACHES")

    def process_research_doc(self, doc: Any) -> None:
        """Extract research areas and labs from research-type Documents."""
        meta = doc.metadata
        url  = meta.get("url", "")
        text = doc.text

        # Extract heading-based research area names
        try:
            heading_path = json.loads(meta.get("heading_path", "[]"))
        except (json.JSONDecodeError, TypeError):
            heading_path = []

        for heading in heading_path:
            h = heading.strip()
            if len(h) > 3 and not h.lower().startswith(("cse", "computer science")):
                area_name = self._add_research_area(h)
                _ = area_name  # stored via _add_research_area

        # Lab names: lines containing "Lab", "Center", "Group", "Institute"
        lab_re = re.compile(
            r"^(.{5,80}(?:Lab(?:oratory)?|Center|Group|Institute|Cluster))\b",
            re.MULTILINE | re.IGNORECASE,
        )
        for m in lab_re.finditer(text):
            lab_name = m.group(1).strip()
            # Filter nav/footer noise
            if len(lab_name) < 8 or lab_name.lower().startswith(("skip", "search")):
                continue
            self._add_lab(lab_name, url)

        # Faculty mentioned on research page
        try:
            fnames = json.loads(meta.get("faculty_names", "[]"))
        except (json.JSONDecodeError, TypeError):
            fnames = []

        for fname in fnames:
            self._add_faculty(fname, {"url": ""})

    def process_program_doc(self, doc: Any) -> None:
        """Extract program nodes and their course memberships."""
        meta  = doc.metadata
        url   = meta.get("url", "")
        title = meta.get("title", "").strip()
        ptype = meta.get("page_type", "")

        # Infer program level from URL / title
        level = "unknown"
        url_l = url.lower()
        if "phd" in url_l or "doctoral" in title.lower():
            level = "phd"
        elif "ms" in url_l or "master" in title.lower():
            level = "ms"
        elif "undergraduate" in url_l or "bs" in url_l or "bachelor" in title.lower():
            level = "bs"
        elif "minor" in url_l or "minor" in title.lower():
            level = "minor"

        if title:
            self._add_program(title, level, url)

        # Courses mentioned in this program doc → PART_OF edges
        codes = extract_course_codes(doc.text)
        for code in codes:
            self._add_course(code, {"url": ""})
            if title:
                self._add_edge(code, title.strip().lower(), "PART_OF")

    # ── Top-level runner ───────────────────────────────────────────────────────

    def extract_all(self, documents: list[Any]) -> None:
        """Route every document to the right extractor based on page_type."""
        counts: dict[str, int] = defaultdict(int)

        for doc in documents:
            ptype = doc.metadata.get("page_type", "general")

            if ptype in {"courses", "catalog_course", "undergraduate_courses", "graduate_courses"}:
                self.process_course_doc(doc)
                counts["course_docs"] += 1

            elif ptype in {"faculty_profile", "faculty_website", "faculty_profile_pdf", "faculty"}:
                self.process_faculty_doc(doc)
                counts["faculty_docs"] += 1

            elif ptype in {"research", "catalog_department"}:
                self.process_research_doc(doc)
                counts["research_docs"] += 1

            elif ptype in {"degree_requirements", "catalog_program", "graduate_programs",
                           "undergraduate_programs", "admissions", "hub_curriculum"}:
                self.process_program_doc(doc)
                counts["program_docs"] += 1

            else:
                # General / PDF pages: just extract course codes for stubs
                for code in extract_course_codes(doc.text):
                    self._add_course(code, {"url": doc.metadata.get("url", "")})
                counts["general_docs"] += 1

        log.info(
            "Extraction complete — courses=%d, faculty=%d, programs=%d, "
            "areas=%d, labs=%d, topics=%d, edges=%d | by_type=%s",
            len(self.courses), len(self.faculty), len(self.programs),
            len(self.research_areas), len(self.labs), len(self.topics),
            len(self.edges), dict(counts),
        )

    def to_graph_data(self) -> dict:
        """Serialize all extracted entities and edges to a plain dict."""
        return {
            "nodes": {
                "courses":        list(self.courses.values()),
                "faculty":        list(self.faculty.values()),
                "programs":       list(self.programs.values()),
                "research_areas": list(self.research_areas.values()),
                "labs":           list(self.labs.values()),
                "topics":         list(self.topics.values()),
            },
            "edges": [
                {"from": f, "to": t, "rel": r, "props": p}
                for f, t, r, p in self.edges
            ],
        }


# ── Neo4j backend ─────────────────────────────────────────────────────────────

class Neo4jGraphStore:
    """
    Ingests extracted entities into Neo4j using the official Python driver.
    Uses MERGE so re-runs are fully idempotent.
    """

    def __init__(self) -> None:
        from neo4j import GraphDatabase  # type: ignore
        self._driver = GraphDatabase.driver(
            cfg.neo4j.uri,
            auth=(cfg.neo4j.user, cfg.neo4j.password),
        )
        self._db = cfg.neo4j.database
        log.info("Neo4j connected: %s (db=%s)", cfg.neo4j.uri, self._db)

    def close(self) -> None:
        self._driver.close()

    def verify(self) -> bool:
        try:
            with self._driver.session(database=self._db) as s:
                s.run("RETURN 1")
            return True
        except Exception as e:
            log.error("Neo4j connection failed: %s", e)
            return False

    def create_constraints(self) -> None:
        """Create uniqueness constraints for fast MERGE operations."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Course) REQUIRE c.code IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Faculty) REQUIRE f.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Program) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:ResearchArea) REQUIRE r.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Lab) REQUIRE l.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
        ]
        with self._driver.session(database=self._db) as s:
            for c in constraints:
                try:
                    s.run(c)
                except Exception as e:
                    log.debug("Constraint (may exist): %s", e)
        log.info("Constraints created")

    def clear(self) -> None:
        with self._driver.session(database=self._db) as s:
            s.run("MATCH (n) DETACH DELETE n")
        log.info("Neo4j graph cleared")

    # ── Node upserts ───────────────────────────────────────────────────────────

    def upsert_courses(self, courses: list[dict]) -> int:
        q = """
        UNWIND $rows AS row
        MERGE (c:Course {code: row.code})
        SET c.title       = coalesce(row.title, c.title, ''),
            c.credits     = coalesce(row.credits, c.credits),
            c.description = coalesce(row.description, c.description, ''),
            c.prereqs     = coalesce(row.prereqs, c.prereqs, '[]'),
            c.url         = coalesce(row.url, c.url, '')
        """
        with self._driver.session(database=self._db) as s:
            s.run(q, rows=courses)
        log.info("Upserted %d courses", len(courses))
        return len(courses)

    def upsert_faculty(self, faculty: list[dict]) -> int:
        q = """
        UNWIND $rows AS row
        MERGE (f:Faculty {name: row.name})
        SET f.email  = coalesce(row.email, f.email, ''),
            f.rank   = coalesce(row.rank, f.rank, ''),
            f.office = coalesce(row.office, f.office, ''),
            f.phone  = coalesce(row.phone, f.phone, ''),
            f.url    = coalesce(row.url, f.url, '')
        """
        with self._driver.session(database=self._db) as s:
            s.run(q, rows=faculty)
        log.info("Upserted %d faculty", len(faculty))
        return len(faculty)

    def upsert_programs(self, programs: list[dict]) -> int:
        q = """
        UNWIND $rows AS row
        MERGE (p:Program {name: row.name})
        SET p.level = coalesce(row.level, p.level, ''),
            p.url   = coalesce(row.url, p.url, '')
        """
        with self._driver.session(database=self._db) as s:
            s.run(q, rows=programs)
        log.info("Upserted %d programs", len(programs))
        return len(programs)

    def upsert_research_areas(self, areas: list[dict]) -> int:
        q = """
        UNWIND $rows AS row
        MERGE (r:ResearchArea {name: row.name})
        """
        with self._driver.session(database=self._db) as s:
            s.run(q, rows=areas)
        log.info("Upserted %d research areas", len(areas))
        return len(areas)

    def upsert_labs(self, labs: list[dict]) -> int:
        q = """
        UNWIND $rows AS row
        MERGE (l:Lab {name: row.name})
        SET l.url = coalesce(row.url, l.url, '')
        """
        with self._driver.session(database=self._db) as s:
            s.run(q, rows=labs)
        log.info("Upserted %d labs", len(labs))
        return len(labs)

    def upsert_topics(self, topics: list[dict]) -> int:
        q = """
        UNWIND $rows AS row
        MERGE (t:Topic {name: row.name})
        """
        with self._driver.session(database=self._db) as s:
            s.run(q, rows=topics)
        log.info("Upserted %d topics", len(topics))
        return len(topics)

    # ── Relationship upserts ───────────────────────────────────────────────────

    def upsert_edges(self, edges: list[dict]) -> int:
        """
        Batch-upsert all edges. Routes by relationship type to the correct Cypher.
        Edges with unknown from/to IDs are silently skipped (nodes may not exist yet).
        """
        # Group by relationship type for efficient batching
        by_rel: dict[str, list[dict]] = defaultdict(list)
        for e in edges:
            by_rel[e["rel"]].append(e)

        total = 0
        for rel, batch in by_rel.items():
            try:
                n = self._upsert_rel_batch(rel, batch)
                total += n
            except Exception as exc:
                log.warning("Edge batch failed for %s (%d edges): %s", rel, len(batch), exc)

        log.info("Upserted %d edges across %d relationship types", total, len(by_rel))
        return total

    def _upsert_rel_batch(self, rel: str, batch: list[dict]) -> int:
        """Route edge batch to the right Cypher template."""
        queries: dict[str, str] = {
            "TEACHES": """
                UNWIND $rows AS row
                MATCH (f:Faculty {name: row.from_name})
                MATCH (c:Course  {code: row.to_code})
                MERGE (f)-[:TEACHES]->(c)
            """,
            "REQUIRES": """
                UNWIND $rows AS row
                MATCH (a:Course {code: row.from_code})
                MATCH (b:Course {code: row.to_code})
                MERGE (a)-[:REQUIRES]->(b)
            """,
            "RESEARCHES": """
                UNWIND $rows AS row
                MATCH (f:Faculty      {name: row.from_name})
                MATCH (r:ResearchArea {name: row.to_name})
                MERGE (f)-[:RESEARCHES]->(r)
            """,
            "AFFILIATED_WITH": """
                UNWIND $rows AS row
                MATCH (f:Faculty      {name: row.from_name})
                MATCH (r:ResearchArea {name: row.to_name})
                MERGE (f)-[:AFFILIATED_WITH]->(r)
            """,
            "PART_OF": """
                UNWIND $rows AS row
                MATCH (c:Course  {code: row.from_code})
                MATCH (p:Program {name: row.to_name})
                MERGE (c)-[:PART_OF]->(p)
            """,
            "COVERS": """
                UNWIND $rows AS row
                MATCH (c:Course {code: row.from_code})
                MATCH (t:Topic  {name: row.to_name})
                MERGE (c)-[:COVERS]->(t)
            """,
            "LEADS": """
                UNWIND $rows AS row
                MATCH (f:Faculty {name: row.from_name})
                MATCH (l:Lab     {name: row.to_name})
                MERGE (f)-[:LEADS]->(l)
            """,
            "HAS_AREA": """
                UNWIND $rows AS row
                MATCH (p:Program      {name: row.from_name})
                MATCH (r:ResearchArea {name: row.to_name})
                MERGE (p)-[:HAS_AREA]->(r)
            """,
        }

        if rel not in queries:
            return 0

        # Prepare rows based on relationship type
        rows = []
        for e in batch:
            frm = e["from"]
            to  = e["to"]
            if rel in ("TEACHES", "RESEARCHES", "AFFILIATED_WITH", "LEADS"):
                # from = faculty key (lowercase name), to = code/area/lab name
                # Resolve faculty name from key
                rows.append({"from_name": frm.title(), "to_code": to.upper(),
                              "to_name": _normalize_area(to) if rel in ("RESEARCHES", "AFFILIATED_WITH")
                                         else to.strip().title()})
            elif rel == "REQUIRES":
                rows.append({"from_code": frm.upper(), "to_code": to.upper()})
            elif rel == "PART_OF":
                rows.append({"from_code": frm.upper(), "to_name": to.strip().title()})
            elif rel == "COVERS":
                rows.append({"from_code": frm.upper(), "to_name": to.strip().title()})
            elif rel in ("HAS_AREA",):
                rows.append({"from_name": frm.strip().title(), "to_name": to.strip().title()})

        if not rows:
            return 0

        with self._driver.session(database=self._db) as s:
            s.run(queries[rel], rows=rows)
        return len(rows)

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._driver.session(database=self._db) as s:
            result = s.run("""
                MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt
                UNION ALL
                MATCH ()-[r]->() RETURN type(r) AS label, count(r) AS cnt
            """)
            return {row["label"]: row["cnt"] for row in result}

    def ingest(self, data: dict, rebuild: bool = False) -> None:
        """Full ingestion: create constraints → (clear) → upsert nodes → upsert edges."""
        self.create_constraints()
        if rebuild:
            self.clear()

        nodes = data["nodes"]
        self.upsert_courses(nodes["courses"])
        self.upsert_faculty(nodes["faculty"])
        self.upsert_programs(nodes["programs"])
        self.upsert_research_areas(nodes["research_areas"])
        self.upsert_labs(nodes["labs"])
        self.upsert_topics(nodes["topics"])
        self.upsert_edges(data["edges"])


# ── NetworkX fallback ─────────────────────────────────────────────────────────

class NetworkXGraphStore:
    """
    Local fallback when Neo4j is unavailable.
    Stores a directed multigraph as a .gpickle file.
    Provides the same query interface as Neo4jGraphStore so
    graph_retriever.py can use either backend transparently.
    """

    def __init__(self) -> None:
        import networkx as nx  # type: ignore
        self._nx = nx
        self.G = nx.MultiDiGraph()
        self._path = cfg.paths.graph
        log.info("Using NetworkX fallback graph store at %s", self._path)

    def load(self) -> bool:
        if self._path.exists():
            try:
                import pickle
                with open(self._path, "rb") as _f:
                    self.G = pickle.load(_f)
                log.info("NetworkX graph loaded: %d nodes, %d edges",
                         self.G.number_of_nodes(), self.G.number_of_edges())
                return True
            except Exception as e:
                log.warning("NetworkX graph load failed: %s", e)
        return False

    def save(self) -> None:
        import pickle
        with open(self._path, "wb") as _f:
            pickle.dump(self.G, _f)
        log.info("NetworkX graph saved → %s (%d nodes, %d edges)",
                 self._path, self.G.number_of_nodes(), self.G.number_of_edges())

    def ingest(self, data: dict, rebuild: bool = False) -> None:
        if not rebuild and self.load():
            return

        nodes = data["nodes"]

        # Add nodes with labels encoded in the "label" attribute
        for course in nodes["courses"]:
            self.G.add_node(course["code"], label="Course", **course)

        for fac in nodes["faculty"]:
            self.G.add_node(fac["name"].lower(), label="Faculty", **fac)

        for prog in nodes["programs"]:
            self.G.add_node(prog["name"].lower(), label="Program", **prog)

        for area in nodes["research_areas"]:
            self.G.add_node(area["name"].lower(), label="ResearchArea", **area)

        for lab in nodes["labs"]:
            self.G.add_node(lab["name"].lower(), label="Lab", **lab)

        for topic in nodes["topics"]:
            self.G.add_node(topic["name"].lower(), label="Topic", **topic)

        # Add edges
        for e in data["edges"]:
            frm = e["from"]
            to  = e["to"]
            if frm in self.G and to in self.G:
                self.G.add_edge(frm, to, rel=e["rel"], **e.get("props", {}))

        log.info("NetworkX graph built: %d nodes, %d edges",
                 self.G.number_of_nodes(), self.G.number_of_edges())
        self.save()

    def stats(self) -> dict:
        from collections import Counter
        label_counts = Counter(
            self.G.nodes[n].get("label", "Unknown") for n in self.G.nodes
        )
        rel_counts = Counter(
            self.G.edges[e].get("rel", "UNKNOWN") for e in self.G.edges
        )
        return {**dict(label_counts), **dict(rel_counts)}


# ── Graph store factory ───────────────────────────────────────────────────────

def get_graph_store(prefer_neo4j: bool = True):
    """
    Return the best available graph store.
    Tries Neo4j first; falls back to NetworkX if connection fails.
    """
    if prefer_neo4j and cfg.neo4j.uri:
        try:
            store = Neo4jGraphStore()
            if store.verify():
                return store
            store.close()
            log.warning("Neo4j unreachable — falling back to NetworkX")
        except Exception as e:
            log.warning("Neo4j init failed (%s) — falling back to NetworkX", e)

    return NetworkXGraphStore()


# ── Pipeline ──────────────────────────────────────────────────────────────────

class GraphBuildPipeline:
    """
    End-to-end: load documents → extract entities → ingest into graph store.
    Saves raw graph_data.json alongside the graph for inspection/debugging.
    """

    def __init__(self, rebuild: bool = False) -> None:
        self.rebuild = rebuild

    def run(self) -> dict:
        t0 = time.perf_counter()

        # Load documents
        docs = IngestionPipeline.load_documents()
        if not docs:
            log.error("No documents — run ingestion_pipeline.py first")
            return {}

        log.info("Extracting entities from %d documents...", len(docs))
        extractor = EntityExtractor()
        extractor.extract_all(docs)

        graph_data = extractor.to_graph_data()

        # Save raw extracted data for debugging
        graph_json_path = cfg.paths.indexes / "graph_data.json"
        save_json(graph_data, graph_json_path)
        log.info("Graph data saved → %s", graph_json_path)

        # Ingest
        store = get_graph_store()
        store.ingest(graph_data, rebuild=self.rebuild)

        elapsed = (time.perf_counter() - t0) * 1000
        log.info("Graph build complete in %.0f ms", elapsed)

        # Summary
        try:
            stats = store.stats()
            log.info("Graph stats: %s", stats)
        except Exception:
            pass

        if hasattr(store, "close"):
            store.close()

        return graph_data


# ── Quick query test (standalone) ─────────────────────────────────────────────

def _test_query_neo4j(query_term: str) -> None:
    """Quick test query against Neo4j for --query flag."""
    store = Neo4jGraphStore()
    if not store.verify():
        print("Neo4j not available")
        store.close()
        return

    with store._driver.session(database=store._db) as s:
        # Try as course code first
        codes = extract_course_codes(query_term)
        if codes:
            code = codes[0]
            print(f"\n=== Course: {code} ===")
            r = s.run(
                "MATCH (c:Course {code: $code}) RETURN c",
                code=code,
            ).single()
            if r:
                print(dict(r["c"]))

            print("\n--- Taught by ---")
            for row in s.run(
                "MATCH (f:Faculty)-[:TEACHES]->(c:Course {code: $code}) RETURN f.name, f.email",
                code=code,
            ):
                print(f"  {row['f.name']} <{row['f.email']}>")

            print("\n--- Prerequisites ---")
            for row in s.run(
                "MATCH (c:Course {code: $code})-[:REQUIRES]->(p:Course) RETURN p.code, p.title",
                code=code,
            ):
                print(f"  {row['p.code']} — {row['p.title']}")

            print("\n--- Related faculty (via topics, BONUS) ---")
            for row in s.run(
                """
                MATCH (c:Course {code: $code})-[:COVERS]->(t:Topic)
                      <-[:RESEARCHES]-(f:Faculty)
                RETURN DISTINCT f.name, f.email LIMIT 5
                """,
                code=code,
            ):
                print(f"  {row['f.name']} <{row['f.email']}>")
        else:
            # Try as faculty name
            print(f"\n=== Faculty matching: {query_term} ===")
            for row in s.run(
                "MATCH (f:Faculty) WHERE toLower(f.name) CONTAINS toLower($name) "
                "RETURN f.name, f.email, f.rank LIMIT 5",
                name=query_term,
            ):
                print(f"  {row['f.name']} | {row['f.rank']} | {row['f.email']}")

            print("\n--- Courses taught ---")
            for row in s.run(
                "MATCH (f:Faculty)-[:TEACHES]->(c:Course) "
                "WHERE toLower(f.name) CONTAINS toLower($name) "
                "RETURN c.code, c.title LIMIT 10",
                name=query_term,
            ):
                print(f"  {row['c.code']} — {row['c.title']}")

    store.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build UB CSE Knowledge Graph")
    parser.add_argument("--rebuild", action="store_true",
                        help="Drop existing graph and rebuild from scratch")
    parser.add_argument("--stats",   action="store_true",
                        help="Print graph statistics and exit")
    parser.add_argument("--query",   type=str, default="",
                        help="Test query (course code or faculty name)")
    args = parser.parse_args()

    if args.stats:
        store = get_graph_store()
        print("\n📊  Graph stats:")
        for k, v in store.stats().items():
            print(f"    {k}: {v}")
        if hasattr(store, "close"):
            store.close()
        return

    if args.query:
        _test_query_neo4j(args.query)
        return

    pipeline = GraphBuildPipeline(rebuild=args.rebuild)
    data     = pipeline.run()

    if data:
        nodes = data.get("nodes", {})
        edges = data.get("edges", [])
        print(f"\n✅  Knowledge Graph built:")
        print(f"    Courses:        {len(nodes.get('courses', []))}")
        print(f"    Faculty:        {len(nodes.get('faculty', []))}")
        print(f"    Programs:       {len(nodes.get('programs', []))}")
        print(f"    Research Areas: {len(nodes.get('research_areas', []))}")
        print(f"    Labs:           {len(nodes.get('labs', []))}")
        print(f"    Topics:         {len(nodes.get('topics', []))}")
        print(f"    Edges:          {len(edges)}")
        print(f"\n    Next → python graph_retriever.py  (or python retriever.py)")


if __name__ == "__main__":
    main()