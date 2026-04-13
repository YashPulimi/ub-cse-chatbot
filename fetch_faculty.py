"""
fetch_faculty.py — Extract faculty emails from directory page
=============================================================
Uses BeautifulSoup to parse mailto: links directly from the
faculty directory page. Clean structure, no guessing.

HOW TO RUN:
  python fetch_faculty.py

Saves to: data/raw/faculty_emails.json
"""

import asyncio
import json
import re
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

DIRECTORY_URLS = [
    "https://engineering.buffalo.edu/computer-science-engineering/people/faculty-directory.html",
    "https://engineering.buffalo.edu/computer-science-engineering/people/faculty-directory.adjunct.html",
    "https://engineering.buffalo.edu/computer-science-engineering/people/faculty-directory.affiliated.html",
]

OUTPUT = Path("data/raw/faculty_emails.json")


async def fetch_page(url: str) -> str:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text


def parse_faculty(html: str) -> dict:
    """
    Parse faculty directory HTML.
    Each faculty block has:
      - Bold link with name: <strong>Name, PhD</strong>
      - mailto link: <a href="mailto:email@buffalo.edu">
      - Office text: "353 Davis Hall"
      - Phone text: "(716) 645-XXXX"

    We use li elements — each faculty is one list item.
    """
    soup    = BeautifulSoup(html, "html.parser")
    faculty = {}

    # Each faculty is in a <li> inside the main content
    content = soup.select_one("#center") or soup.select_one("main") or soup
    items   = content.select("li")

    for li in items:
        # Get name from bold link text
        bold = li.select_one("strong, b")
        if not bold:
            continue

        name = bold.get_text(strip=True)
        # Remove degree suffix: "Rohini Srihari, PhD" → "Rohini Srihari"
        name = re.sub(r",?\s*(PhD|MS|MD|EdD|JD|DPhil|MFA)\s*$", "", name).strip()

        if len(name.split()) < 2:
            continue

        # Get email from mailto link
        email = ""
        for a in li.select("a[href^='mailto:']"):
            href = a.get("href", "")
            candidate = href.replace("mailto:", "").strip()
            if "@buffalo.edu" in candidate:
                email = candidate
                break

        if not email:
            continue

        # Get office
        office = ""
        office_m = re.search(
            r"\d+[A-Z]?\s+(?:Davis|Capen|Fargo|Bell|Baldy|Furnas)\s+Hall",
            li.get_text()
        )
        if office_m:
            office = office_m.group(0)

        # Get phone
        phone = ""
        phone_m = re.search(r"\(716\)\s*\d{3}-\d{4}", li.get_text())
        if phone_m:
            phone = phone_m.group(0)

        # Get rank
        rank = "Faculty"
        text = li.get_text()
        for pattern, label in [
            ("SUNY Distinguished Professor",     "SUNY Distinguished Professor"),
            ("SUNY Empire Innovation Professor", "SUNY Empire Innovation Professor"),
            ("Professor of Teaching",            "Professor of Teaching"),
            ("Associate Professor of Teaching",  "Associate Professor of Teaching"),
            ("Assistant Professor of Teaching",  "Assistant Professor of Teaching"),
            ("Professor of Practice",            "Professor of Practice"),
            ("Associate Professor",              "Associate Professor"),
            ("Assistant Professor",              "Assistant Professor"),
            ("Professor",                        "Professor"),
        ]:
            if pattern in text:
                rank = label
                break

        # Get research topics
        topics = []
        topic_m = re.search(r"Research Topics?[:\s]+([^\n]{10,300})", text)
        if topic_m:
            topics = [t.strip() for t in re.split(r"[;,]", topic_m.group(1)) if t.strip()]

        # Get profile URL
        profile_url = ""
        for a in li.select("a[href*='detail.html']"):
            profile_url = a.get("href", "")
            break

        faculty[name] = {
            "name":            name,
            "email":           email,
            "office":          office,
            "phone":           phone,
            "rank":            rank,
            "research_topics": topics[:5],
            "profile_url":     profile_url,
        }

    return faculty


async def main():
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    all_faculty = {}
    for url in DIRECTORY_URLS:
        print(f"Fetching {url}...")
        try:
            html    = await fetch_page(url)
            faculty = parse_faculty(html)
            all_faculty.update(faculty)
            print(f"  Found {len(faculty)} faculty")
        except Exception as e:
            print(f"  Failed: {e}")

    OUTPUT.write_text(json.dumps(all_faculty, indent=2))
    print(f"\n✅ Saved {len(all_faculty)} faculty to {OUTPUT}")

    # Verify key professors
    print("\nVerification:")
    for name in ["Rohini Srihari", "Alina Vereshchaka", "David Doermann", "Junsong Yuan"]:
        if name in all_faculty:
            f = all_faculty[name]
            print(f"  {f['name']} → {f['email']} | {f['office']}")
        else:
            print(f"  {name} NOT FOUND")


if __name__ == "__main__":
    asyncio.run(main())
