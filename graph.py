"""
graph.py — UB CSE Knowledge Graph Builder (Neo4j)
===================================================
Reads:  data/raw/crawl_*.jsonl       (web pages)
        data/raw/faculty_emails.json  (clean faculty data from directory)

Builds: Neo4j graph

Nodes:       Professor, Course, Lab, Program, ResearchArea, Department
Relationships: TEACHES, MEMBER_OF, WORKS_IN, PREREQ_FOR, PART_OF, OFFERED_BY

HOW TO RUN:
  python fetch_faculty.py   (once, to get clean emails)
  python graph.py
"""

import json
import logging
import re
from pathlib import Path

from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("graph")

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password123"
RAW_DIR        = Path("data/raw")
FACULTY_JSON   = RAW_DIR / "faculty_emails.json"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_faculty() -> list[dict]:
    """
    Load faculty from the clean JSON extracted by fetch_faculty.py.
    This is the single source of truth for faculty emails/offices/phones.
    Emails come from mailto: links in the faculty directory — 100% accurate.
    """
    if not FACULTY_JSON.exists():
        log.error("faculty_emails.json not found. Run fetch_faculty.py first.")
        return []

    raw = json.loads(FACULTY_JSON.read_text())
    faculty = []
    for name, info in raw.items():
        # Clean trailing comma from name
        clean_name = name.rstrip(",").strip()
        faculty.append({
            "name":            clean_name,
            "email":           info.get("email", ""),
            "office":          info.get("office", ""),
            "phone":           info.get("phone", ""),
            "rank":            info.get("rank", "Faculty"),
            "research_topics": info.get("research_topics", []),
            "url":             info.get("profile_url", ""),
        })

    log.info(f"Loaded {len(faculty)} faculty from JSON")
    return faculty


def load_pages() -> list[dict]:
    """Load crawled pages from latest JSONL."""
    candidates = sorted(RAW_DIR.glob("crawl_*.jsonl"), reverse=True)
    if not candidates:
        log.error("No crawl file found. Run crawler.py first.")
        return []
    pages = [json.loads(l) for l in candidates[0].read_text().splitlines() if l.strip()]
    log.info(f"Loaded {len(pages)} pages from {candidates[0].name}")
    return pages


# ── Entity extractors ─────────────────────────────────────────────────────────

def extract_courses(pages: list[dict]) -> list[dict]:
    """Extract courses from catalog and degree requirement pages."""
    courses = {}
    pattern = re.compile(
        r"\b(CSE\s*\d{3}[A-Z]?)\b"
        r"(?:[:\s\-–—LEC]+)"
        r"([A-Z][A-Za-z ,&:()/]{4,60}?)"
        r"(?:\s*Credits?:\s*(\d))?",
        re.MULTILINE,
    )
    valid_types = {
        "courses", "degree_requirements", "catalog_program",
        "catalog_course", "graduate_programs", "undergraduate_programs",
        "catalog_department",
    }
    for page in pages:
        if page.get("page_type") not in valid_types:
            continue
        for m in pattern.finditer(page.get("content", "")):
            code = "CSE " + re.sub(r"\s+", "", m.group(1))[3:].upper()
            if code not in courses:
                courses[code] = {
                    "code":    code,
                    "title":   m.group(2).strip().rstrip(",;:."),
                    "credits": m.group(3) or "3",
                    "url":     page.get("url", ""),
                }
    log.info(f"Extracted {len(courses)} courses")
    return list(courses.values())


def extract_programs() -> list[dict]:
    return [
        {"name": "MS in Computer Science and Engineering",  "level": "MS",          "abbreviation": "MS CSE"},
        {"name": "PhD in Computer Science and Engineering", "level": "PhD",         "abbreviation": "PhD CSE"},
        {"name": "BS in Computer Science",                  "level": "BS",          "abbreviation": "BS CS"},
        {"name": "BS in Computer Engineering",              "level": "BS",          "abbreviation": "BS CEN"},
        {"name": "BA in Computer Science",                  "level": "BA",          "abbreviation": "BA CS"},
        {"name": "Advanced Certificate in Cybersecurity",   "level": "Certificate", "abbreviation": "Cybersecurity Cert"},
        {"name": "Minor in Computer Science",               "level": "Minor",       "abbreviation": "CS Minor"},
        {"name": "Minor in Cybersecurity",                  "level": "Minor",       "abbreviation": "Cybersecurity Minor"},
    ]


def extract_relationships(
    pages: list[dict],
    faculty: list[dict],
    courses: list[dict],
) -> dict:
    """Extract entity relationships from crawled pages."""
    teaches   = []
    prereqs   = []
    member_of = []
    works_in  = []
    part_of   = []

    prof_map     = {p["name"].lower(): p["name"] for p in faculty}
    course_codes = {c["code"] for c in courses}
    code_re      = re.compile(r"\bCSE\s*(\d{3}[A-Z]?)\b")

    for page in pages:
        text  = page.get("content", "")
        ptype = page.get("page_type", "")
        title = page.get("title", "").lower()

        # TEACHES — faculty profile mentions course codes
        if ptype in ("faculty_profile", "faculty"):
            for m in code_re.finditer(text):
                code = f"CSE {m.group(1)}"
                if code not in course_codes:
                    continue
                for pname_lower, pname in prof_map.items():
                    if pname_lower.split()[-1] in title:
                        teaches.append((pname, code))

        # TEACHES — syllabi with explicit instructor line
        if ptype in ("syllabus", "courses"):
            inst = re.search(
                r"(?:instructor|professor)[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)",
                text, re.IGNORECASE
            )
            if inst:
                iname = inst.group(1)
                for m in code_re.finditer(text):
                    code = f"CSE {m.group(1)}"
                    if code in course_codes and iname.lower() in prof_map:
                        teaches.append((prof_map[iname.lower()], code))

        # PREREQS
        pm = re.search(r"pre-?requisites?[:\s]+([^\n]{5,200})", text, re.IGNORECASE)
        if pm:
            before_codes = code_re.findall(text[max(0, pm.start()-100):pm.start()])
            for main_num in before_codes:
                main = f"CSE {main_num}"
                for pre_num in code_re.findall(pm.group(1)):
                    pre = f"CSE {pre_num}"
                    if main in course_codes and pre in course_codes and main != pre:
                        prereqs.append((pre, main))

        # WORKS_IN — research topics from faculty profiles
        if ptype in ("faculty_profile", "faculty"):
            tm = re.search(r"Research Topics?[:\s]+([^\n]{10,300})", text, re.IGNORECASE)
            if tm:
                topics = [t.strip() for t in re.split(r"[;,]", tm.group(1)) if len(t.strip()) > 3]
                for pname_lower, pname in prof_map.items():
                    if pname_lower.split()[-1] in title:
                        for topic in topics[:5]:
                            works_in.append((pname, topic))

        # MEMBER_OF — lab mentions in faculty profiles
        if ptype in ("faculty_profile", "faculty"):
            for lab in re.findall(r"([A-Z][A-Za-z ]{3,40}(?:Lab|Laboratory|Center|Group))", text):
                for pname_lower, pname in prof_map.items():
                    if pname_lower.split()[-1] in title:
                        member_of.append((pname, lab.strip()))

    # PART_OF — hardcoded known program courses
    program_courses = {
        "MS in Computer Science and Engineering": [
            "CSE 521", "CSE 531", "CSE 565", "CSE 574", "CSE 676",
            "CSE 573", "CSE 562", "CSE 635", "CSE 586", "CSE 589",
        ],
        "PhD in Computer Science and Engineering": [
            "CSE 531", "CSE 565", "CSE 574", "CSE 596",
        ],
        "BS in Computer Science": [
            "CSE 115", "CSE 116", "CSE 191", "CSE 220", "CSE 250",
            "CSE 305", "CSE 331", "CSE 341", "CSE 396", "CSE 421",
        ],
    }
    for prog, codes in program_courses.items():
        for code in codes:
            if code in course_codes:
                part_of.append((code, prog))

    # Deduplicate all
    return {
        "teaches":   list(set(teaches)),
        "prereqs":   list(set(prereqs)),
        "member_of": list(set(member_of)),
        "works_in":  list(set(works_in)),
        "part_of":   list(set(part_of)),
    }


# ── Neo4j builder ─────────────────────────────────────────────────────────────

class GraphBuilder:

    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.driver.verify_connectivity()
        log.info("Neo4j connected")

    def close(self):
        self.driver.close()

    def setup(self):
        with self.driver.session() as s:
            for label, prop in [
                ("Professor", "name"), ("Course", "code"),
                ("Lab", "name"), ("Program", "name"), ("ResearchArea", "name"),
            ]:
                s.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE")
            s.run("MATCH (n) DETACH DELETE n")
        log.info("Constraints created, graph cleared")

    def load(self, faculty, courses, programs, rels):
        with self.driver.session() as s:
            # Professors
            s.run("""
                UNWIND $rows AS r
                MERGE (p:Professor {name: r.name})
                SET p.email = r.email, p.office = r.office,
                    p.phone = r.phone, p.rank = r.rank,
                    p.research_topics = r.research_topics, p.url = r.url
            """, rows=faculty)

            # Courses
            s.run("""
                UNWIND $rows AS r
                MERGE (c:Course {code: r.code})
                SET c.title = r.title, c.credits = r.credits, c.url = r.url
            """, rows=courses)

            # Programs
            s.run("""
                UNWIND $rows AS r
                MERGE (p:Program {name: r.name})
                SET p.level = r.level, p.abbreviation = r.abbreviation
                MERGE (d:Department {name: 'UB CSE'})
                MERGE (p)-[:OFFERED_BY]->(d)
            """, rows=programs)

            # Relationships
            for prof, code in rels["teaches"]:
                s.run("MATCH (p:Professor {name:$p}) MATCH (c:Course {code:$c}) MERGE (p)-[:TEACHES]->(c)", p=prof, c=code)
            for pre, main in rels["prereqs"]:
                s.run("MATCH (a:Course {code:$a}) MATCH (b:Course {code:$b}) MERGE (a)-[:PREREQ_FOR]->(b)", a=pre, b=main)
            for prof, lab in rels["member_of"]:
                s.run("MATCH (p:Professor {name:$p}) MERGE (l:Lab {name:$l}) MERGE (p)-[:MEMBER_OF]->(l)", p=prof, l=lab)
            for prof, area in rels["works_in"]:
                s.run("MATCH (p:Professor {name:$p}) MERGE (r:ResearchArea {name:$r}) MERGE (p)-[:WORKS_IN]->(r)", p=prof, r=area)
            for code, prog in rels["part_of"]:
                s.run("MATCH (c:Course {code:$c}) MATCH (p:Program {name:$p}) MERGE (c)-[:PART_OF]->(p)", c=code, p=prog)

        log.info("All data loaded")

    def stats(self):
        with self.driver.session() as s:
            for label in ["Professor", "Course", "Lab", "Program", "ResearchArea"]:
                n = s.run(f"MATCH (n:{label}) RETURN count(n) as c").single()["c"]
                print(f"  {label:<15} {n}")
            for r in s.run("MATCH ()-[r]->() RETURN type(r) as t, count(r) as c ORDER BY c DESC"):
                print(f"  [{r['t']}] {r['c']}")

    def verify(self):
        print("\nEmail verification:")
        with self.driver.session() as s:
            for name in ["Rohini Srihari", "Alina Vereshchaka", "David Doermann", "Junsong Yuan"]:
                r = s.run(
                    "MATCH (p:Professor) WHERE p.name CONTAINS $n "
                    "RETURN p.name, p.email, p.office LIMIT 1",
                    n=name.split()[-1]
                ).single()
                if r:
                    print(f"  {r['p.name']} → {r['p.email']} | {r['p.office']}")
                else:
                    print(f"  {name} NOT FOUND")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    faculty  = load_faculty()
    pages    = load_pages()
    courses  = extract_courses(pages)
    programs = extract_programs()
    rels     = extract_relationships(pages, faculty, courses)

    log.info(f"teaches={len(rels['teaches'])} works_in={len(rels['works_in'])} "
             f"member_of={len(rels['member_of'])} prereqs={len(rels['prereqs'])}")

    builder = GraphBuilder()
    builder.setup()
    builder.load(faculty, courses, programs, rels)

    print(f"\n{'='*40}")
    print("GRAPH STATS")
    print('='*40)
    builder.stats()
    builder.verify()
    builder.close()

    print("\n✅ Graph built! View at http://localhost:7474")
    print("Next: chainlit run app.py")


if __name__ == "__main__":
    main()
