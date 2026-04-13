"""
crawler.py — UB CSE Complete Crawler
======================================
Data sources:
  1. engineering.buffalo.edu/computer-science-engineering — department site
  2. catalogs.buffalo.edu — CSE degree catalog pages only
  3. Faculty personal websites — top-level page only (no recursion)
  4. Faculty CV PDFs — downloaded
  5. Syllabi, checklists, recent handbooks (2024+)

Smart PDF filtering:
  - CVs: downloaded for all faculty
  - Handbooks: only 2024, 2025, 2026
  - Syllabi, checklists: always keep
  - Research papers, tech reports: always skip
"""

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, urlunparse, urljoin

import httpx
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("crawler")

# ── Seed URLs ─────────────────────────────────────────────────────────────────

SEED_URLS = [
    # Department site
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
    "https://engineering.buffalo.edu/computer-science-engineering/research/research-areas/artificial-intelligence/artificial-intelligence-and-machine-learning-and-data-mining.html",
    "https://engineering.buffalo.edu/computer-science-engineering/research/research-centers-institutes-labs-and-groups.html",

    # Catalog — CSE specific pages
    "https://catalogs.buffalo.edu/preview_entity.php?catoid=3&ent_oid=52",
    "https://catalogs.buffalo.edu/preview_program.php?catoid=12&poid=4964",
    "https://catalogs.buffalo.edu/preview_program.php?catoid=17&poid=5857",
    "https://catalogs.buffalo.edu/preview_program.php?catoid=1&ent_oid=52",
    "https://catalogs.buffalo.edu/preview_program.php?catoid=1&poid=68",
    "https://catalogs.buffalo.edu/preview_program.php?catoid=1&poid=67",
    "https://catalogs.buffalo.edu/preview_program.php?catoid=11&poid=4427",
]

# ── Domain rules ──────────────────────────────────────────────────────────────

DEPARTMENT_DOMAIN = "engineering.buffalo.edu"
DEPARTMENT_PREFIX = "/computer-science-engineering"
CATALOG_DOMAIN    = "catalogs.buffalo.edu"

CATALOG_FOLLOW_PATTERNS = [
    r"preview_program\.php.*catoid",
    r"preview_entity\.php.*catoid",
    r"preview_course_nopop\.php.*catoid",
]

SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".css", ".js",
    ".svg", ".ico", ".woff", ".woff2", ".mp4", ".zip",
}

PAGE_TYPE_MAP = {
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
    "faculty-directory":              "faculty_profile",
    ".detail.html":                   "faculty_profile",
}

# ── PDF filter ────────────────────────────────────────────────────────────────

# Always skip these — research noise
PDF_SKIP_PATTERNS = [
    r"^\d{4}-\d{2}",        # tech reports: 2024-15.pdf
    r"\bvita\b",             # some CVs named vita (but we want _CV files)
    r"dissertation",
    r"proceedings",
    r"siwei_lyu",            # known research CV
]

# Always keep these
PDF_KEEP_PATTERNS = [
    r"syllabus", r"syllabi",
    r"checklist",
    r"catalog",
    r"requirement",
    r"curriculum",
    r"cse.5\d\d", r"cse.6\d\d",
    r"schedule",
    r"bulletin",
    r"_cv", r"-cv",          # faculty CVs — KEEP these
]

PDF_MAX_SIZE_KB = 800


def is_relevant_pdf(url: str) -> bool:
    """
    Smart PDF filter:
    - CVs: always keep (faculty CVs have rich teaching/course info)
    - Handbooks: only 2024+ (older ones are 90% same content)
    - Syllabi/checklists: always keep
    - Research papers/tech reports: skip
    """
    filename = url.split("/")[-1].lower()

    # Always skip
    for p in PDF_SKIP_PATTERNS:
        if re.search(p, filename):
            return False

    # Always keep CVs and syllabi
    for p in PDF_KEEP_PATTERNS:
        if re.search(p, filename):
            return True

    # Handbooks — keep only recent
    if "handbook" in filename:
        m = re.search(r"20(\d{2})", filename)
        if m:
            year = 2000 + int(m.group(1))
            return year >= 2024
        return False

    # Default: skip unknown
    return False


# ── URL helpers ───────────────────────────────────────────────────────────────

def normalize_url(url: str) -> str:
    try:
        p = urlparse(url)
        if p.netloc == CATALOG_DOMAIN:
            params = {}
            for part in (p.query or "").split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    if k in ("catoid", "navoid", "poid", "ent_oid"):
                        params[k] = v
            query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            return urlunparse((p.scheme, p.netloc, p.path, "", query, ""))
        normalized = urlunparse((p.scheme, p.netloc, p.path, "", "", ""))
        return normalized.rstrip("/") if len(normalized) > 8 else normalized
    except Exception:
        return url


def is_allowed(url: str) -> bool:
    """Allow department site + catalog CSE pages."""
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False
        if Path(p.path).suffix.lower() in SKIP_EXTENSIONS:
            return False
        if p.netloc == DEPARTMENT_DOMAIN:
            return p.path.startswith(DEPARTMENT_PREFIX)
        if p.netloc == CATALOG_DOMAIN:
            for pattern in CATALOG_FOLLOW_PATTERNS:
                if re.search(pattern, url.lower()):
                    return True
            return False
        return False
    except Exception:
        return False


def detect_page_type(url: str) -> str:
    for segment, ptype in PAGE_TYPE_MAP.items():
        if segment in url:
            return ptype
    return "general"


def content_hash(text: str) -> str:
    return hashlib.md5(
        re.sub(r"\s+", " ", text.lower().strip()).encode()
    ).hexdigest()


# ── Content extraction ────────────────────────────────────────────────────────

CONTENT_SELECTORS = [
    "#center",
    "#acalog-content",
    ".acalog-content",
    "#contentarea",
    ".profile-content",
    ".faculty-profile",
    "main",
    "[role='main']",
    "article",
    ".content-area",
    "#content",
    "body",
]

NOISE_SELECTORS = [
    "nav", "header", "footer",
    "#left", "#right",
    ".leftnav", ".rightnav",
    ".breadcrumbs",
    ".pagination",
    "script", "style", "noscript",
    ".cookie-banner",
    ".fatfooter",
    "#cat_nav",
    ".catalog-search",
    "#acalog-nav",
]


def extract_content(html: str) -> tuple[str, str]:
    if not html:
        return "", ""
    soup = BeautifulSoup(html, "html.parser")
    for selector in NOISE_SELECTORS:
        for el in soup.select(selector):
            el.decompose()
    title = ""
    for sel in ["h1", "title"]:
        tag = soup.find(sel)
        if tag:
            title = tag.get_text(strip=True).split("|")[0].strip()
            if title:
                break
    for selector in CONTENT_SELECTORS:
        el = soup.select_one(selector)
        if el:
            text = el.get_text(separator="\n", strip=True)
            if len(text.split()) > 30:
                return clean_text(text), title
    return "", title


def extract_faculty_links(html: str, base_url: str) -> tuple[list[str], list[str]]:
    """
    Extract from faculty profile pages:
    - personal_websites: external professor websites
    - cv_pdfs: CV PDF links
    Returns (personal_websites, cv_pdfs)
    """
    if not html:
        return [], []

    soup    = BeautifulSoup(html, "html.parser")
    websites = []
    cv_pdfs  = []

    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        text = a.get_text(strip=True).lower()

        if not href:
            continue

        # CV PDFs
        if href.lower().endswith(".pdf") and any(
            kw in text or kw in href.lower()
            for kw in ["cv", "curriculum", "vita", "resume"]
        ):
            full_url = urljoin(base_url, href)
            cv_pdfs.append(full_url)

        # Personal websites — external links with "website" or "home" in text
        elif (
            href.startswith("http")
            and DEPARTMENT_DOMAIN not in href
            and CATALOG_DOMAIN not in href
            and "buffalo.edu" not in href
            and any(kw in text for kw in [
                "website", "home page", "homepage", "personal",
                "dr.", "prof.", "lab", "research page"
            ])
        ):
            websites.append(href)

    return list(set(websites)), list(set(cv_pdfs))


def clean_text(text: str) -> str:
    lines   = text.split("\n")
    cleaned = []
    seen    = set()
    for line in lines:
        line = line.strip()
        if len(line) < 3:
            continue
        if re.match(r"^https?://\S+$", line):
            continue
        if line in ("previous", "next", "Skip to Content",
                    ">", "»", "Print", "Help", "?", "Call", "Email"):
            continue
        if re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", line):
            continue
        key = re.sub(r"\s+", " ", line.lower())
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


# ── PDF downloader ────────────────────────────────────────────────────────────

async def download_pdf(url: str, pdf_dir: Path) -> Path | None:
    filename = re.sub(r"[^\w\-.]", "_", url.split("/")[-1])
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    out_path = pdf_dir / filename
    if out_path.exists():
        return out_path
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            head    = await client.head(url)
            size_kb = int(head.headers.get("content-length", 0)) / 1024
            if size_kb > PDF_MAX_SIZE_KB:
                log.info(f"PDF skip (too large {size_kb:.0f}KB): {filename}")
                return None
            resp = await client.get(url)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            log.info(f"✓ PDF: {filename} ({len(resp.content)/1024:.0f}KB)")
            return out_path
    except Exception as e:
        log.error(f"PDF failed {url}: {e}")
        return None


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Page:
    url:          str
    title:        str
    content:      str
    page_type:    str
    content_hash: str
    word_count:   int
    depth:        int
    source:       str        # department | catalog | faculty_website
    crawled_at:   str
    pdf_links:    list = field(default_factory=list)
    pdf_paths:    list = field(default_factory=list)


# ── Crawler ───────────────────────────────────────────────────────────────────

class UBCSECrawler:

    def __init__(self, max_pages=400, max_depth=5, concurrency=5):
        self.max_pages          = max_pages
        self.max_depth          = max_depth
        self.semaphore          = asyncio.Semaphore(concurrency)
        self.visited_urls       : set[str] = set()
        self.seen_hashes        : set[str] = set()
        self.all_pdf_urls       : set[str] = set()
        self.faculty_websites   : set[str] = set()
        self.pages              : list[Page] = []

    async def fetch(self, crawler: AsyncWebCrawler, url: str, depth: int,
                    source: str = "department"):
        async with self.semaphore:
            try:
                cfg = CrawlerRunConfig(
                    word_count_threshold    = 25,
                    exclude_external_links  = False,  # need external for faculty websites
                    remove_overlay_elements = True,
                    wait_until              = "domcontentloaded",
                    excluded_selector       = ",".join([
                        "footer", "nav", "header",
                        "#left", "#right", ".leftnav",
                        ".fatfooter", "script", "style",
                        "#cat_nav", ".catalog-search",
                    ]),
                )

                result = await crawler.arun(url=url, config=cfg)
                if not result.success:
                    return None

                html           = result.html or ""
                content, title = extract_content(html)

                if not content or len(content.split()) < 30:
                    return None

                chash = content_hash(content)
                if chash in self.seen_hashes:
                    log.info(f"SKIP (dup) {url[:70]}")
                    return None
                self.seen_hashes.add(chash)

                # Extract all internal links
                raw_links = [
                    urljoin(url, lnk.get("href", ""))
                    for lnk in (result.links.get("internal") or [])
                    if lnk.get("href")
                ]

                # For faculty profile pages — extract personal websites + CVs
                if "faculty" in detect_page_type(url) or ".detail.html" in url:
                    personal_sites, cv_pdfs = extract_faculty_links(html, url)
                    for site in personal_sites:
                        self.faculty_websites.add(site)
                    for cv in cv_pdfs:
                        self.all_pdf_urls.add(cv)
                        log.info(f"  📄 CV found: {cv.split('/')[-1]}")

                # Collect relevant PDFs
                pdf_links = [
                    l for l in raw_links
                    if l.lower().endswith(".pdf") and is_relevant_pdf(l)
                ]
                for pdf_url in pdf_links:
                    self.all_pdf_urls.add(pdf_url)

                # Child URLs
                child_urls = [
                    normalize_url(l) for l in raw_links
                    if is_allowed(l) and not l.lower().endswith(".pdf")
                ]

                page = Page(
                    url          = url,
                    title        = title,
                    content      = content,
                    page_type    = detect_page_type(url),
                    content_hash = chash,
                    word_count   = len(content.split()),
                    depth        = depth,
                    source       = source,
                    crawled_at   = datetime.utcnow().isoformat(),
                    pdf_links    = pdf_links,
                )

                log.info(
                    f"✓ d={depth} [{source:16s}] "
                    f"[{page.page_type:18s}] "
                    f"{page.word_count:5d}w  {url[:60]}"
                )
                return page, child_urls

            except Exception as e:
                log.error(f"ERROR {url}: {e}")
                return None

    async def fetch_faculty_websites(self, crawler: AsyncWebCrawler) -> None:
        """
        Fetch top-level page of each professor's personal website.
        No recursion — just the homepage for research interests, courses, contact.
        """
        if not self.faculty_websites:
            log.info("No faculty websites found")
            return

        log.info(f"Fetching {len(self.faculty_websites)} faculty personal websites...")

        for url in self.faculty_websites:
            if len(self.pages) >= self.max_pages:
                break
            norm = normalize_url(url)
            if norm in self.visited_urls:
                continue
            self.visited_urls.add(norm)

            result = await self.fetch(crawler, url, depth=0, source="faculty_website")
            if result:
                page, _ = result  # no recursion from personal sites
                page.page_type = "faculty_website"
                self.pages.append(page)

    async def download_all_pdfs(self, pdf_dir: Path) -> None:
        if not self.all_pdf_urls:
            log.info("No PDFs to download")
            return
        log.info(f"Downloading {len(self.all_pdf_urls)} PDFs...")
        sem = asyncio.Semaphore(3)

        async def bounded(url):
            async with sem:
                return url, await download_pdf(url, pdf_dir)

        results     = await asyncio.gather(*[bounded(u) for u in self.all_pdf_urls])
        url_to_path = {u: str(p) for u, p in results if p}
        for page in self.pages:
            page.pdf_paths = [url_to_path[u] for u in page.pdf_links if u in url_to_path]
        downloaded = sum(1 for _, p in results if p)
        log.info(f"PDFs downloaded: {downloaded}/{len(self.all_pdf_urls)}")

    async def run(self, output_dir: Path, pdf_dir: Path) -> list[Page]:
        queue: asyncio.Queue = asyncio.Queue()
        for url in SEED_URLS:
            norm = normalize_url(url)
            await queue.put((norm, 0))
            self.visited_urls.add(norm)

        async with AsyncWebCrawler(
            config=BrowserConfig(headless=True, verbose=False)
        ) as crawler:

            # ── Phase 1: Crawl department site + catalog ──────────────────
            log.info("Phase 1: Crawling department site + catalog...")
            while not queue.empty() and len(self.pages) < self.max_pages:
                batch = []
                while not queue.empty() and len(batch) < 5:
                    batch.append(await queue.get())

                results = await asyncio.gather(*[
                    self.fetch(crawler, url, depth,
                               source="catalog" if CATALOG_DOMAIN in url else "department")
                    for url, depth in batch
                ])

                for result in results:
                    if result is None:
                        continue
                    page, child_urls = result
                    self.pages.append(page)
                    if page.depth < self.max_depth:
                        for child in child_urls:
                            norm = normalize_url(child)
                            if norm not in self.visited_urls:
                                self.visited_urls.add(norm)
                                await queue.put((norm, page.depth + 1))

            log.info(f"Phase 1 done: {len(self.pages)} pages")
            log.info(f"Faculty websites found: {len(self.faculty_websites)}")

            # ── Phase 2: Fetch faculty personal websites ──────────────────
            log.info("Phase 2: Fetching faculty personal websites...")
            await self.fetch_faculty_websites(crawler)
            log.info(f"Phase 2 done: {len(self.pages)} total pages")

        # ── Phase 3: Download PDFs ────────────────────────────────────────
        log.info("Phase 3: Downloading PDFs...")
        await self.download_all_pdfs(pdf_dir)

        return self.pages

    def save(self, output_dir: Path) -> Path:
        ts  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out = output_dir / f"crawl_{ts}.jsonl"

        with open(out, "w") as f:
            for page in self.pages:
                f.write(json.dumps(asdict(page)) + "\n")

        by_type   = {}
        by_source = {}
        for p in self.pages:
            by_type[p.page_type]   = by_type.get(p.page_type, 0) + 1
            by_source[p.source]    = by_source.get(p.source, 0) + 1

        summary = {
            "total_pages":           len(self.pages),
            "total_words":           sum(p.word_count for p in self.pages),
            "duplicates_skipped":    len(self.visited_urls) - len(self.pages),
            "faculty_websites":      len(self.faculty_websites),
            "pdfs_found":            len(self.all_pdf_urls),
            "pdfs_downloaded":       len(list(Path("data/raw/pdfs").glob("*.pdf"))),
            "by_type":               by_type,
            "by_source":             by_source,
            "crawled_at":            datetime.utcnow().isoformat(),
        }

        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        log.info(f"Saved     → {out}")
        log.info(f"By type   : {by_type}")
        log.info(f"By source : {by_source}")
        log.info(f"Words     : {summary['total_words']:,}")
        log.info(f"Faculty websites crawled: {summary['faculty_websites']}")
        log.info(f"PDFs downloaded: {summary['pdfs_downloaded']}")
        return out


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    output_dir = Path("data/raw")
    pdf_dir    = Path("data/raw/pdfs")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("UB CSE Complete Crawler")
    log.info("Sources:")
    log.info("  1. Department site (faculty profiles, programs, courses)")
    log.info("  2. Degree catalog (CSE programs only)")
    log.info("  3. Faculty personal websites (top-level only)")
    log.info("  4. CV PDFs + syllabi + recent handbooks")
    log.info("=" * 60)

    crawler = UBCSECrawler(max_pages=400, max_depth=5, concurrency=5)
    await crawler.run(output_dir, pdf_dir)
    out = crawler.save(output_dir)

    print(f"\n✅ Done!")
    print(f"Pages → {out}")
    print(f"PDFs  → {pdf_dir}")
    print(f"Next  → python graph.py")


if __name__ == "__main__":
    asyncio.run(main())
