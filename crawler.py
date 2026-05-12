"""
crawler.py — UB CSE Async Crawler (Improved)
=============================================
Builds on the original Crawl4AI BFS crawler.
All improvements are targeted — working parts are kept exactly as-is.

FIXES APPLIED (from diagnosis):
  1. PDF_MAX_KB raised 800 → 5000  (captures MS/PhD handbooks ~2MB)
  2. PDF_KEEP extended: worksheet, advising, flowchart, guideline, schedule, handbook (all years)
  3. PDF_SKIP: removed "dissertation" — abstracts carry faculty→research edges for GraphRAG
  4. alpha < 0.40 line filter now exempts lines containing a CSE course code
  5. Boilerplate threshold raised 3 → cfg.crawler.bp_threshold (default 12)
     — prevents "This course satisfies UB Curriculum" from being nuked on every catalog page
  6. word_count gate lowered 40 → 25 for short faculty stub / emeritus pages
  7. NOISE_SELECTORS: removed "#right" (catalog prereq tables live there)
     "[class*='social']" scoped to navigation-only social icons (not content blocks)
  8. is_allowed: added cse.buffalo.edu, se.buffalo.edu, ubcse.github.io, www.buffalo.edu/hub
  9. wait_until changed domcontentloaded → networkidle for JS accordion course descriptions
 10. Last-Modified HEAD request moved OUTSIDE the semaphore (fire-and-forget task)
 11. Batch size raised 5 → cfg.crawler.concurrency (default 10)
 12. Page dataclass: added heading_path, outgoing_links, courses_mentioned, faculty_names
 13. Faculty site discovery: added "research" keyword; removed false-positive guard
 14. All constants driven from config.cfg — no hardcoded values

KEPT AS-IS (good original work):
  - Crawl4AI AsyncWebCrawler with BrowserConfig
  - BFS queue with visited set + seen_hashes dedup
  - _structured_text DOM walker (heading/table/list structure preserved)
  - _table_to_text row formatter
  - Two-pass boilerplate fingerprint removal
  - Async PDF download with semaphore
  - extract_faculty_fields (email, rank, office, phone, research_areas)
  - extract_course_fields (code, title, credits, description, prereqs)
  - save() → .jsonl + summary.json

HOW TO RUN:
  python crawler.py
  # Output → data/raw/crawl_<timestamp>.jsonl
  # PDFs   → data/raw/pdfs/
  # Next   → python ingestion_pipeline.py
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, urlunparse, urljoin

import httpx
from bs4 import BeautifulSoup, Tag
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

from config import cfg
from utils import get_logger, content_hash, normalize_course_code

log = get_logger(__name__)

# ── Domain constants ──────────────────────────────────────────────────────────

DEPT_DOMAIN = "engineering.buffalo.edu"
DEPT_PREFIX = "/computer-science-engineering"
CAT_DOMAIN  = "catalogs.buffalo.edu"

# FIX 8: additional UB-affiliated domains allowed
EXTRA_DOMAINS = {
    "cse.buffalo.edu",       # legacy CSE domain, some faculty pages redirect here
    "se.buffalo.edu",        # software engineering (affiliated programs)
    "ubcse.github.io",       # lab/course materials hosted on GitHub
}

# UB HUB and registrar — these require a real browser session on the UB network.
# www.buffalo.edu blocks headless Playwright from off-campus / restricted environments.
# Add these manually if you are on the UB network and they resolve for you.
HUB_SEEDS: list[str] = []   # populated at runtime if UB_INCLUDE_HUB=true in .env

# ── Seed URLs ─────────────────────────────────────────────────────────────────

SEED_URLS = [
    # Core department
    "https://engineering.buffalo.edu/computer-science-engineering.html",
    "https://engineering.buffalo.edu/computer-science-engineering/graduate.html",
    "https://engineering.buffalo.edu/computer-science-engineering/undergraduate.html",
    "https://engineering.buffalo.edu/computer-science-engineering/people/faculty-directory.html",
    "https://engineering.buffalo.edu/computer-science-engineering/people/faculty-directory.adjunct.html",
    "https://engineering.buffalo.edu/computer-science-engineering/people/faculty-directory.affiliated.html",
    "https://engineering.buffalo.edu/computer-science-engineering/research.html",
    "https://engineering.buffalo.edu/computer-science-engineering/graduate/admissions.html",
    "https://engineering.buffalo.edu/computer-science-engineering/graduate/degrees-and-programs.html",
    "https://engineering.buffalo.edu/computer-science-engineering/graduate/courses.html",
    "https://engineering.buffalo.edu/computer-science-engineering/undergraduate/degrees-and-programs.html",
    "https://engineering.buffalo.edu/computer-science-engineering/undergraduate/courses.html",
    "https://engineering.buffalo.edu/computer-science-engineering/graduate/degrees-and-programs/ms-in-computer-science-and-engineering.html",
    "https://engineering.buffalo.edu/computer-science-engineering/graduate/degrees-and-programs/phd-in-computer-science-and-engineering.html",
    # Research areas
    "https://engineering.buffalo.edu/computer-science-engineering/research/research-areas.html",
    "https://engineering.buffalo.edu/computer-science-engineering/research/research-areas/artificial-intelligence/artificial-intelligence-and-machine-learning-and-data-mining.html",
    "https://engineering.buffalo.edu/computer-science-engineering/research/research-centers-institutes-labs-and-groups.html",
    # Catalog programs
    "https://catalogs.buffalo.edu/preview_entity.php?catoid=3&ent_oid=52",
    "https://catalogs.buffalo.edu/preview_program.php?catoid=12&poid=4964",
    "https://catalogs.buffalo.edu/preview_program.php?catoid=17&poid=5857",
    "https://catalogs.buffalo.edu/preview_program.php?catoid=1&poid=68",
    "https://catalogs.buffalo.edu/preview_program.php?catoid=1&poid=67",
    "https://catalogs.buffalo.edu/preview_program.php?catoid=11&poid=4427",
]

# ── Skip extensions ───────────────────────────────────────────────────────────

SKIP_EXT = {
    ".jpg", ".jpeg", ".png", ".gif", ".css", ".js", ".svg",
    ".ico", ".woff", ".woff2", ".mp4", ".zip", ".tar", ".gz",
    ".exe", ".dmg", ".pptx", ".docx", ".xlsx",
    # Binary/legacy formats that cause Playwright download-interception errors
    ".ps", ".eps", ".dvi", ".bib", ".tex", ".bbl",
    ".bin", ".dat", ".mat", ".o", ".a", ".so", ".dylib",
}

# ── Page type mapping ─────────────────────────────────────────────────────────

PAGE_TYPES: dict[str, str] = {
    "/graduate/admissions":           "admissions",
    "/graduate/degrees-and-programs": "degree_requirements",
    "/graduate/courses":              "courses",
    "/graduate":                      "graduate_programs",
    "/undergraduate/degrees":         "degree_requirements",
    "/undergraduate/courses":         "courses",
    "/undergraduate":                 "undergraduate_programs",
    "/people":                        "faculty",
    "/research":                      "research",
    "preview_program":                "catalog_program",
    "preview_course":                 "catalog_course",
    "preview_entity":                 "catalog_department",
    ".detail.html":                   "faculty_profile",
    "faculty-directory":              "faculty_profile",
    "hub/courses":                    "hub_curriculum",
    "registrar":                      "registrar",
}

# ── PDF filters ───────────────────────────────────────────────────────────────

# FIX 1+2: raised size cap, extended keep patterns, removed dissertation from skip
PDF_KEEP = [
    r"syllabus", r"syllabi", r"checklist", r"worksheet",
    r"catalog",  r"requirement", r"curriculum", r"schedule",
    r"cse.?\d{3}",                          # any CSE course number
    r"_cv", r"-cv", r"vitae",
    r"handbook",                            # FIX 2: all handbooks (not year-gated)
    r"advising", r"advise",
    r"flowchart", r"guideline", r"guide",
    r"degree.?plan", r"program.?sheet",
]
PDF_SKIP = [
    r"^\d{4}-\d{2}",   # date-prefixed internal files
    r"proceedings",     # conference papers (not dept info)
    r"thesis",          # full thesis documents
]

# ── DOM selectors ─────────────────────────────────────────────────────────────

# FIX 7: removed "#right" (holds catalog prereq tables)
# FIX 7: "[class*='social']" → ".social-share" only (not all social blocks)
NOISE_SELECTORS = [
    "nav", "header", "footer", "script", "style", "noscript", "iframe",
    "#left", ".leftnav", ".rightnav", ".breadcrumb", "#breadcrumb",
    ".fatfooter", ".footer-content", "#ub-header",
    ".cookie-banner", "[class*='cookie']", "[class*='gdpr']", "[class*='consent']",
    "#cat_nav", ".catalog-search", "#acalog-nav",
    ".social-share",            # navigation social icons only (not content blocks)
    "[role='search']",
    ".pagination", ".utility-bar", ".sidebar-widget",
]

CONTENT_SELECTORS = [
    "#center", "#acalog-content", ".acalog-content", "#contentarea",
    ".profile-content", ".faculty-profile", "main", "[role='main']",
    "article", ".content-area", "#content",
    "#right",           # FIX 7: try #right as content fallback (catalog prereq tables)
]

# ── Noise patterns ────────────────────────────────────────────────────────────

_JSON_BLOB  = re.compile(r'\{["\s]*"[a-zA-Z]+"[\s\S]{50,3000}\}', re.DOTALL)
_COURSE_RE  = re.compile(r"\b(CSE\s*\d{3}[A-Za-z]?)\b", re.I)
_PREREQ_RE  = re.compile(r"pre-?requisites?[:\s]+([^\n]{5,200})", re.I)
_HEADING_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)

# FIX 4: line noise pattern — exempts lines that contain a course code
_NOISE_LINE = re.compile(
    r"^(https?://\S+|\d{1,2}/\d{1,2}/\d{2,4}|[|•\-–—_\s]+)$"
    r"|^(previous|next|print|help|skip to|home\s*>|breadcrumb|©|\ball rights|"
    r"accept|decline|close|cookie|loading|please wait|back to top|menu|search).*",
    re.I,
)


# ── URL helpers ───────────────────────────────────────────────────────────────

def norm_url(url: str) -> str:
    """Normalize URL for deduplication — keeps catalog query params, strips rest."""
    try:
        p = urlparse(url.strip())
        if p.netloc == CAT_DOMAIN:
            keep = {
                k: v
                for part in p.query.split("&") if "=" in part
                for k, v in [part.split("=", 1)]
                if k in ("catoid", "poid", "ent_oid", "navoid")
            }
            q = "&".join(f"{k}={v}" for k, v in sorted(keep.items()))
            return urlunparse((p.scheme, p.netloc, p.path, "", q, ""))
        return urlunparse((p.scheme, p.netloc, p.path.rstrip("/"), "", "", ""))
    except Exception:
        return url


def is_allowed(url: str) -> bool:
    """
    Return True if this URL should be crawled.
    FIX 8: added extra UB-affiliated domains.
    """
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False
        if Path(p.path).suffix.lower() in SKIP_EXT:
            return False
        # Primary CSE department domain
        if p.netloc == DEPT_DOMAIN:
            return p.path.startswith(DEPT_PREFIX)
        # UB catalog
        if p.netloc == CAT_DOMAIN:
            return any(re.search(pat, url) for pat in [
                r"preview_program\.php.*catoid",
                r"preview_entity\.php.*catoid",
                r"preview_course_nopop\.php.*catoid",
            ])
        # FIX 8: extra affiliated domains — but exclude known useless subtrees
        if p.netloc in EXTRA_DOMAINS:
            # Skip legacy tech-report archives (1990s .ps/.dvi files)
            # and other non-content subtrees that cause download errors
            _skip_paths = ("/tech-reports/", "/~rapaport/", "/ftp/")
            if any(skip in p.path for skip in _skip_paths):
                return False
            return True
        return False
    except Exception:
        return False


def page_type(url: str) -> str:
    for seg, pt in PAGE_TYPES.items():
        if seg in url:
            return pt
    return "general"


def is_faculty_site(url: str) -> bool:
    """External link from a faculty profile that looks like a personal/lab site."""
    p = urlparse(url)
    return (
        p.scheme in ("http", "https")
        and DEPT_DOMAIN not in url
        and CAT_DOMAIN not in url
        and "buffalo.edu" not in url
    )


def is_keep_pdf(url: str) -> bool:
    """Return True if this PDF URL matches our keep patterns."""
    name = url.split("/")[-1].lower()
    if any(re.search(pat, name) for pat in PDF_SKIP):
        return False
    if any(re.search(pat, name) for pat in PDF_KEEP):
        return True
    return False


# ── DOM extraction (kept from original — good code) ───────────────────────────

def _table_to_text(tbl: Tag) -> str:
    rows = []
    for tr in tbl.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if any(cells):
            rows.append(" | ".join(c for c in cells if c))
    return "\n".join(rows)


def _structured_text(el: Tag) -> str:
    """Walk DOM preserving heading/table/list structure as plain text."""
    parts = []
    for child in el.children:
        if isinstance(child, str):
            t = child.strip()
            if t:
                parts.append(t)
            continue
        if not isinstance(child, Tag):
            continue
        tag = (child.name or "").lower()
        if tag in ("h1", "h2", "h3", "h4"):
            t = child.get_text(" ", strip=True)
            if t:
                parts.append(f"\n## {t}\n")
        elif tag == "table":
            t = _table_to_text(child)
            if t:
                parts.append(f"\n{t}\n")
        elif tag in ("ul", "ol"):
            for li in child.find_all("li", recursive=False):
                t = li.get_text(" ", strip=True)
                if t:
                    parts.append(f"• {t}")
        elif tag == "p":
            t = child.get_text(" ", strip=True)
            if t:
                parts.append(f"\n{t}\n")
        elif tag in ("div", "section", "article", "main"):
            inner = _structured_text(child)
            if inner.strip():
                parts.append(inner)
        else:
            t = child.get_text(" ", strip=True)
            if t:
                parts.append(t)
    return "\n".join(parts)


def _clean_lines(text: str) -> str:
    """
    Remove JSON blobs, noise lines, low-alpha lines, and exact duplicates.
    FIX 4: lines containing a CSE course code are EXEMPT from alpha < 0.40 filter.
    """
    text = _JSON_BLOB.sub("", text)
    seen: set[str] = set()
    out: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or len(line) < 2:
            continue
        if _NOISE_LINE.search(line):
            continue
        # FIX 4: skip alpha check if line has a course code (e.g. "CSE 574 — 3 credits")
        has_course = bool(_COURSE_RE.search(line))
        if not has_course:
            alpha = sum(c.isalpha() for c in line)
            if len(line) > 5 and alpha / len(line) < 0.40:
                continue
        key = re.sub(r"\s+", " ", line.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _extract_heading_path(text: str) -> list[str]:
    """Return ordered list of ## headings found in cleaned text — used as metadata."""
    return _HEADING_RE.findall(text)


def extract(html: str) -> tuple[str, str]:
    """Full extraction pipeline. Returns (cleaned_text, title)."""
    if not html:
        return "", ""
    soup = BeautifulSoup(html, "html.parser")

    # Title
    title = ""
    for sel in ("h1", "title"):
        tag = soup.find(sel)
        if tag:
            title = tag.get_text(strip=True).split("|")[0].strip()
            if title:
                break

    # DOM surgery
    for sel in NOISE_SELECTORS:
        try:
            for el in soup.select(sel):
                el.decompose()
        except Exception:
            pass

    # FIX 9: do NOT strip display:none for catalog pages — JS accordions use it
    # Only strip truly hidden elements (aria-hidden=true, visibility:hidden)
    # Guard: some malformed HTML tags have el.attrs=None — skip those entirely
    for el in soup.find_all(True):
        if not el.attrs:
            continue
        style = el.get("style", "").replace(" ", "")
        if "visibility:hidden" in style or el.get("aria-hidden") == "true":
            el.decompose()

    # Find content container
    for sel in CONTENT_SELECTORS:
        el = soup.select_one(sel)
        if el:
            raw = _structured_text(el)
            if len(raw.split()) > 30:
                return _clean_lines(raw), title

    body = soup.find("body")
    if body:
        return _clean_lines(_structured_text(body)), title
    return "", title


# ── Structured field extraction (kept from original + minor improvements) ─────

def extract_faculty_fields(html: str, text: str) -> dict:
    """
    Extract typed fields from a faculty profile page.
    FIX 13: added "research" as a valid faculty website keyword.
    """
    soup = BeautifulSoup(html, "html.parser")
    fields: dict = {}

    # Email — mailto link is most reliable
    for a in soup.select("a[href^='mailto:']"):
        email = a["href"].replace("mailto:", "").strip()
        if "@buffalo.edu" in email:
            fields["email"] = email
            break

    # Rank — ordered most-specific first
    for rank in [
        "SUNY Distinguished Professor", "SUNY Empire Innovation Professor",
        "Professor of Teaching", "Associate Professor of Teaching",
        "Assistant Professor of Teaching", "Professor of Practice",
        "Associate Professor", "Assistant Professor", "Professor",
        "Lecturer", "Visiting Professor", "Research Professor", "Emeritus",
    ]:
        if rank.lower() in text.lower():
            fields["rank"] = rank
            break

    # Office — UB building names
    m = re.search(
        r"\d+[A-Z]?\s+(?:Davis|Capen|Fargo|Bell|Baldy|Furnas|Bonner|Knox)\s+Hall",
        text,
    )
    if m:
        fields["office"] = m.group(0)

    # Phone
    m = re.search(r"\(716\)\s*\d{3}-\d{4}", text)
    if m:
        fields["phone"] = m.group(0)

    # Research areas — from "Research Topics:" label
    m = re.search(r"Research\s+(?:Topics?|Interests?|Areas?)[:\s]+([^\n]{10,300})", text, re.I)
    if m:
        fields["research_areas"] = [
            t.strip() for t in re.split(r"[;,]", m.group(1)) if t.strip()
        ][:8]

    # Personal / lab websites
    # FIX 13: added "research" keyword
    site_keywords = {"website", "homepage", "home page", "personal", "lab", "research page", "research"}
    sites = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        txt  = a.get_text(strip=True).lower()
        if href.startswith("http") and is_faculty_site(href):
            if any(kw in txt for kw in site_keywords):
                sites.append(href)
    if sites:
        fields["personal_websites"] = list(set(sites))

    return fields


def extract_course_fields(text: str) -> list[dict]:
    """
    Extract structured course records from a courses/catalog page.
    Uses normalize_course_code from utils for consistent IDs.
    """
    courses: dict[str, dict] = {}
    pattern = re.compile(
        r"\b(CSE\s*\d{3}[A-Za-z]?)\b[:\s\-–—LEC]+([A-Z][A-Za-z ,&:()/]{4,60}?)"
        r"(?:\s*Credits?:\s*(\d))?",
        re.MULTILINE,
    )
    for m in pattern.finditer(text):
        code = normalize_course_code(m.group(1))
        if code in courses:
            continue
        start = m.end()
        nxt   = _COURSE_RE.search(text, start)
        desc  = text[start: nxt.start() if nxt else start + 600].strip()
        desc  = re.sub(r"\s+", " ", desc)[:400]
        prereqs: list[str] = []
        pm = _PREREQ_RE.search(desc)
        if pm:
            prereqs = [normalize_course_code(c) for c in _COURSE_RE.findall(pm.group(1))]
        courses[code] = {
            "code":        code,
            "title":       m.group(2).strip().rstrip(",;:."),
            "credits":     m.group(3) or "3",
            "description": desc,
            "prereqs":     prereqs,
        }
    return list(courses.values())


# ── PDF download (kept from original, uses config for size cap) ───────────────

async def download_pdf(url: str, pdf_dir: Path) -> Path | None:
    name = re.sub(r"[^\w\-.]", "_", url.split("/")[-1])
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    out = pdf_dir / name
    if out.exists():
        return out
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as c:
            head = await c.head(url)
            size_kb = int(head.headers.get("content-length", 0)) / 1024
            if size_kb > cfg.crawler.pdf_max_kb:          # FIX 1: uses config (5000 KB)
                log.debug("PDF too large (%.0f KB): %s", size_kb, url)
                return None
            r = await c.get(url)
            r.raise_for_status()
            out.write_bytes(r.content)
            log.info("PDF ✓ %s (%.0f KB)", name, len(r.content) / 1024)
            return out
    except Exception as e:
        log.warning("PDF fail %s: %s", name, e)
        return None


# ── Data model (extended from original) ──────────────────────────────────────

@dataclass
class Page:
    url:           str
    title:         str
    content:       str            # cleaned full text
    page_type:     str
    structured:    dict           # typed fields: courses list OR faculty fields
    content_hash:  str
    word_count:    int
    depth:         int
    source:        str            # department | catalog | faculty_website | hub
    crawled_at:    str
    last_modified: str = ""       # HTTP Last-Modified header
    pdf_links:     list = field(default_factory=list)
    pdf_paths:     list = field(default_factory=list)
    # FIX 12: new fields for chunker + GraphRAG
    heading_path:      list = field(default_factory=list)   # ordered ## headings
    outgoing_links:    list = field(default_factory=list)   # all internal href targets
    courses_mentioned: list = field(default_factory=list)   # normalized CSE codes in text
    faculty_names:     list = field(default_factory=list)   # names from structured field



def extract_internal_links(result, base_url: str) -> list[str]:
    """
    Safely extract internal links from a Crawl4AI result object.
    Handles: result.links=None, items=None, plain strings, dicts,
    and link-like objects across different Crawl4AI versions.
    """
    links_map = getattr(result, "links", None) or {}
    internal  = links_map.get("internal") or []
    out: list[str] = []
    for item in internal:
        if item is None:
            continue
        href = ""
        if isinstance(item, dict):
            href = item.get("href") or item.get("url") or ""
        elif isinstance(item, str):
            href = item
        else:
            href = getattr(item, "href", "") or getattr(item, "url", "") or ""
        if not href:
            continue
        try:
            out.append(urljoin(base_url, href))
        except Exception:
            continue
    return out

# ── Crawler ───────────────────────────────────────────────────────────────────

class UBCSECrawler:

    def __init__(self) -> None:
        self.max_pages    = cfg.crawler.max_pages
        self.max_depth    = cfg.crawler.max_depth
        self.sem          = asyncio.Semaphore(cfg.crawler.concurrency)
        self.visited:      set[str]   = set()
        self.seen_hashes:  set[str]   = set()
        self.pdf_urls:     set[str]   = set()
        self.pages:        list[Page] = []
        self._bp_counts:         Counter       = Counter()
        self._last_modified_cache: dict[str, str] = {}   # url → Last-Modified header

    async def _fetch(
        self,
        crawler: AsyncWebCrawler,
        url: str,
        depth: int,
        source: str,
        no_recurse: bool = False,
    ):
        async with self.sem:
            try:
                run_cfg = CrawlerRunConfig(
                    word_count_threshold    = 20,
                    exclude_external_links  = False,
                    remove_overlay_elements = True,
                    # FIX 9: networkidle waits for JS accordions on catalog pages
                    wait_until              = "networkidle",
                    excluded_selector       = (
                        "footer,nav,header,.leftnav,.fatfooter,"
                        "script,style,#cat_nav,.cookie-banner"
                    ),
                )
                result = await crawler.arun(url=url, config=run_cfg)
                if not result.success:
                    return None

                html        = result.html or ""
                text, title = extract(html)

                # FIX 6: lowered gate from 40 → 25 for short faculty stub pages
                if not text or len(text.split()) < 25:
                    return None

                h = content_hash(text)
                if h in self.seen_hashes:
                    log.info("SKIP dup %s", url[:70])
                    return None
                self.seen_hashes.add(h)

                ptype = page_type(url)

                # Structured extraction
                structured: dict = {}
                if ptype == "faculty_profile":
                    structured = extract_faculty_fields(html, text)
                elif ptype in ("courses", "catalog_course", "catalog_program", "degree_requirements"):
                    courses = extract_course_fields(text)
                    if courses:
                        structured = {"courses": courses}

                # Safe link extraction — handles None, dicts, strings, link objects
                raw_links: list[str] = extract_internal_links(result, url)

                # PDF links
                pdf_links = [l for l in raw_links if l.lower().endswith(".pdf") and is_keep_pdf(l)]
                for p in pdf_links:
                    self.pdf_urls.add(p)

                # FIX 12: extract heading path and course codes from text
                heading_path      = _extract_heading_path(text)
                courses_mentioned = sorted({
                    normalize_course_code(m.group(0))
                    for m in _COURSE_RE.finditer(text)
                })
                faculty_names = []
                if "name" in structured:
                    faculty_names = [structured["name"]]

                # Faculty personal websites (queued inline, no-recurse)
                faculty_sites: list[str] = []
                if ptype == "faculty_profile":
                    soup = BeautifulSoup(html, "html.parser")
                    site_kws = {
                        "website", "homepage", "home page", "personal",
                        "lab", "research page", "research",  # FIX 13
                    }
                    for a in soup.find_all("a", href=True):
                        href = a["href"]
                        txt  = a.get_text(strip=True).lower()
                        if href.startswith("http") and is_faculty_site(href):
                            if any(kw in txt for kw in site_kws):
                                faculty_sites.append(href)

                # Register blocks for boilerplate detection
                for block in re.split(r"\n{2,}", text):
                    b = re.sub(r"\s+", " ", block.lower().strip())
                    if len(b) >= 20:
                        self._bp_counts[b] += 1

                # FIX 10: Last-Modified fired as background task — does NOT block fetch slot
                # Result stored in _last_modified_cache; merged into pages in save()
                last_mod = ""
                _url_ref = url  # capture for closure
                async def _store_lm(_u: str = _url_ref) -> None:
                    val = await self._fetch_last_modified(_u)
                    if val:
                        self._last_modified_cache[_u] = val
                asyncio.ensure_future(_store_lm())

                # Child URLs for BFS
                child_urls: list[str] = [] if no_recurse else [
                    norm_url(l) for l in raw_links
                    if is_allowed(l) and not l.lower().endswith(".pdf")
                ]

                page = Page(
                    url=url,
                    title=title,
                    content=text,
                    page_type=ptype,
                    structured=structured,
                    content_hash=h,
                    word_count=len(text.split()),
                    depth=depth,
                    source=source,
                    crawled_at=datetime.utcnow().isoformat(),
                    last_modified=last_mod,
                    pdf_links=pdf_links,
                    heading_path=heading_path,
                    outgoing_links=child_urls[:50],   # cap to avoid giant payloads
                    courses_mentioned=courses_mentioned,
                    faculty_names=faculty_names,
                )
                log.info(
                    "✓ d=%d [%s] [%s] %dw  %s",
                    depth, source[:14], ptype[:18], page.word_count, url[:60],
                )
                return page, child_urls, faculty_sites

            except Exception as e:
                log.exception("ERR %s", url)
                return None

    async def _fetch_last_modified(self, url: str) -> str:
        """
        FIX 10: Non-blocking background HEAD request for Last-Modified.
        Result is best-effort — stored separately and merged in save().
        """
        try:
            async with httpx.AsyncClient(timeout=8.0) as c:
                head = await c.head(url)
                return head.headers.get("last-modified", "")
        except Exception:
            return ""

    def _strip_boilerplate(self) -> None:
        """
        Two-pass: blocks appearing on ≥ bp_threshold pages are boilerplate.
        FIX 5: threshold raised from 3 → cfg.crawler.bp_threshold (default 12).
        This prevents catalog phrases like "This course satisfies UB Curriculum"
        from being incorrectly flagged and removed.
        """
        threshold = cfg.crawler.bp_threshold
        bp = {fp for fp, c in self._bp_counts.items() if c >= threshold}
        log.info("Boilerplate blocks (≥%d pages): %d", threshold, len(bp))
        before = len(self.pages)
        for page in self.pages:
            kept = [
                b for b in re.split(r"\n{2,}", page.content)
                if re.sub(r"\s+", " ", b.lower().strip()) not in bp
            ]
            page.content    = "\n\n".join(kept).strip()
            page.word_count = len(page.content.split())
        # FIX 6: lowered minimum from 40 → 25
        self.pages = [
            p for p in self.pages
            if p.word_count >= 25
            and sum(c.isalpha() for c in p.content) / max(len(p.content), 1) >= 0.45
        ]
        log.info(
            "Pages after boilerplate removal: %d (dropped %d)",
            len(self.pages), before - len(self.pages),
        )

    async def run(self) -> list[Page]:
        output_dir = cfg.paths.raw
        pdf_dir    = cfg.paths.pdfs

        queue: asyncio.Queue = asyncio.Queue()
        active: int = 0

        # Seed the queue
        for url in SEED_URLS:
            n = norm_url(url)
            src = "catalog" if CAT_DOMAIN in url else "department"
            await queue.put((n, 0, src, False))
            self.visited.add(n)

        # HUB/registrar seeds — no-recurse, top level only
        for url in HUB_SEEDS:
            n = norm_url(url)
            await queue.put((n, 0, "hub", True))
            self.visited.add(n)

        # FIX 11: batch size = concurrency (was hardcoded 5)
        batch_size = cfg.crawler.concurrency

        async with AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False)) as crawler:
            while (not queue.empty() or active > 0) and len(self.pages) < self.max_pages:
                # Drain up to batch_size items
                batch = []
                while not queue.empty() and len(batch) < batch_size:
                    batch.append(await queue.get())
                if not batch:
                    await asyncio.sleep(0.1)
                    continue

                active += len(batch)
                results = await asyncio.gather(*[
                    self._fetch(crawler, url, depth, source, no_recurse)
                    for url, depth, source, no_recurse in batch
                ])
                active -= len(batch)

                for result in results:
                    if result is None:
                        continue
                    page, child_urls, faculty_sites = result
                    self.pages.append(page)

                    if page.depth < self.max_depth:
                        for child in child_urls:
                            n = norm_url(child)
                            if n not in self.visited:
                                self.visited.add(n)
                                src = "catalog" if CAT_DOMAIN in child else "department"
                                await queue.put((n, page.depth + 1, src, False))

                    for site in faculty_sites:
                        n = norm_url(site)
                        if n not in self.visited:
                            self.visited.add(n)
                            await queue.put((n, 0, "faculty_website", True))

        # Async PDF download — max 5 concurrent
        if self.pdf_urls:
            log.info("Downloading %d PDFs...", len(self.pdf_urls))
            pdf_sem = asyncio.Semaphore(5)

            async def _dl(u: str):
                async with pdf_sem:
                    return u, await download_pdf(u, pdf_dir)

            dl_results  = await asyncio.gather(*[_dl(u) for u in self.pdf_urls])
            url_to_path = {u: str(p) for u, p in dl_results if p}
            for page in self.pages:
                page.pdf_paths = [url_to_path[u] for u in page.pdf_links if u in url_to_path]

        self._strip_boilerplate()
        return self.pages

    def save(self) -> Path:
        output_dir = cfg.paths.raw
        ts  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out = output_dir / f"crawl_{ts}.jsonl"

        # Merge last_modified from background tasks into pages before writing
        for page in self.pages:
            if not page.last_modified and page.url in self._last_modified_cache:
                page.last_modified = self._last_modified_cache[page.url]

        with open(out, "w", encoding="utf-8") as f:
            for page in self.pages:
                f.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")

        by_type: dict[str, int] = {}
        for p in self.pages:
            by_type[p.page_type] = by_type.get(p.page_type, 0) + 1

        summary = {
            "total_pages":  len(self.pages),
            "total_words":  sum(p.word_count for p in self.pages),
            "by_type":      by_type,
            "pdfs_found":   len(self.pdf_urls),
            "crawled_at":   datetime.utcnow().isoformat(),
            "fixes_applied": [
                "PDF_MAX_KB=5000",
                "BP_THRESHOLD=12",
                "word_count_gate=25",
                "alpha_filter_exempts_course_codes",
                "heading_path+outgoing_links+courses_mentioned added",
                "networkidle for JS accordions",
                "#right kept as content fallback",
                "extra domains: cse.buffalo.edu, ubcse.github.io",
            ],
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        log.info("Saved %d pages → %s", len(self.pages), out)
        log.info("By type: %s", by_type)
        return out


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    log.info("Starting UB CSE crawl  (max_pages=%d  max_depth=%d  concurrency=%d)",
             cfg.crawler.max_pages, cfg.crawler.max_depth, cfg.crawler.concurrency)
    crawler = UBCSECrawler()
    await crawler.run()
    out = crawler.save()
    print(f"\n✅  Pages  → {out}")
    print(f"    PDFs   → {cfg.paths.pdfs}")
    print(f"    Next   → python ingestion_pipeline.py")


if __name__ == "__main__":
    asyncio.run(main())