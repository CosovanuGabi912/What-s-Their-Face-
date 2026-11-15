"""Download the Wikipedia article of every breed in data/breeds_map.json (issue #5).

Uses the MediaWiki TextExtracts API (prop=extracts, explaintext=1, redirects=1): plain text where
section headings are kept as "== Heading ==" lines, so no wikitext cleaning is needed. Each page is
saved with its revision ID, so the exact text behind every chunk ID can be traced and re-downloaded.

Output:
    data/raw/wikipedia/<breed-slug>.json   one file per breed (extract, title, pageid, revid, ...)
    data/raw/wikipedia/_report.json        breeds that are missing, disambiguation pages, suspicious pages

Candidate titles are tried in order (WIKIPEDIA_TITLES override, "<name> (dog breed)", "<name> (dog)",
"<name>"). A disambiguation page is never used; it is reported. Pages that do not look like a dog or
canid article (no dog-related category) are reported too, so a wrong article is caught before ingestion.

ingest.py reads only data/raw/, so re-running this script is the only way chunk IDs can change.

Usage:
    python scripts/download_wikipedia.py            # skips breeds that are already downloaded
    python scripts/download_wikipedia.py --refresh  # download everything again
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATA_DIR  # noqa: E402
from shared.breeds import load_breeds_map, slugify  # noqa: E402

RAW_DIR = DATA_DIR / "raw" / "wikipedia"
API = "https://en.wikipedia.org/w/api.php"
REQUEST_PAUSE = 3.0  # seconds between requests: full-text requests are throttled to a few per minute
USER_AGENT = "WhatsTheirFace-course-project/1.0 (student project; breed knowledge base)"

# display name -> exact Wikipedia title, where the automatic candidates pick the wrong page or none.
WIKIPEDIA_TITLES: dict[str, str] = {
    "Poodle (Toy)": "Poodle",
    "Poodle (Miniature)": "Poodle",
    "Poodle (Standard)": "Poodle",
    "African Wild Dog": "African wild dog",
    "Dhole": "Dhole",
    "Dingo": "Dingo",
    "Havanese": "Havanese dog",                       # "Havanese" alone is the city
    "Eskimo Dog": "Canadian Eskimo Dog",
    "Segugio Italiano": "Segugio Italiano a Pelo Raso",  # disambiguation page; two coat varieties, smooth-coated chosen
    "Bavarian Mountain Scent Hound": "Bavarian Mountain Hound",
    "Braque Francais Pyrenean": "Braque Français",   # one article covers both types
    "Cirneco dell’Etna": "Cirneco dell'Etna",
    "Manchester Terrier (Standard)": "Manchester Terrier",
    "Manchester Terrier (Toy)": "English Toy Terrier",
    "Slovakian Wirehaired Pointer": "Slovak Rough-haired Pointer",
    "Small Munsterlander Pointer": "Small Münsterländer",
    "Working Kelpie": "Australian Kelpie",
}

DOG_CATEGORY_WORDS = ("dog", "canid", "canis", "hound", "terrier", "spaniel", "retriever", "mastiff")


def candidate_titles(display: str) -> list[str]:
    if display in WIKIPEDIA_TITLES:
        return [WIKIPEDIA_TITLES[display]]
    return [f"{display} (dog breed)", f"{display} (dog)", display]


def api_get(session: requests.Session, params: dict) -> dict:
    """GET with polite pacing and back-off: Wikipedia answers 429 with Retry-After when it is hit too fast."""
    for attempt in range(8):
        response = session.get(API, params=params, timeout=60)
        if response.status_code == 429 or response.status_code >= 500:
            retry_after = response.headers.get("Retry-After", "")
            time.sleep(int(retry_after) + 1 if retry_after.isdigit() else 5 * (attempt + 1))
            continue
        response.raise_for_status()
        time.sleep(REQUEST_PAUSE)
        return response.json()
    raise RuntimeError(f"Wikipedia kept failing: {params.get('titles')!r}")


def resolve_titles(session: requests.Session, titles: list[str]) -> dict[str, dict]:
    """Cheap batched lookup (no text): requested title -> {title, missing, is_disambiguation, categories}."""
    found: dict[str, dict] = {}
    redirect_of: dict[str, str] = {}
    for start in range(0, len(titles), 40):
        batch = titles[start:start + 40]
        params = {"action": "query", "format": "json", "formatversion": 2, "redirects": 1,
                  "titles": "|".join(batch), "prop": "pageprops|categories", "ppprop": "disambiguation",
                  "cllimit": "max", "clshow": "!hidden"}
        extra: dict = {}
        while True:  # categories continue across several responses
            data = api_get(session, {**params, **extra})
            query = data["query"]
            for item in query.get("normalized", []) + query.get("redirects", []):
                redirect_of[item["from"]] = item["to"]
            for page in query["pages"]:
                entry = found.setdefault(page["title"], {"title": page["title"], "categories": [],
                                                         "missing": bool(page.get("missing")),
                                                         "is_disambiguation": False})
                entry["is_disambiguation"] |= "disambiguation" in page.get("pageprops", {})
                entry["categories"] += [c["title"] for c in page.get("categories", [])]
            if "continue" not in data:
                break
            extra = data["continue"]
    result = {}
    for title in titles:
        final = title
        for _ in range(5):  # follow normalisation / redirect chains
            if final in redirect_of:
                final = redirect_of[final]
        result[title] = found.get(final, {"title": final, "missing": True, "is_disambiguation": False, "categories": []})
    return result


def fetch_page(session: requests.Session, title: str) -> dict:
    """Plain-text extract (sections as '== Heading ==' lines) plus the revision ID of that exact text."""
    params = {"action": "query", "format": "json", "formatversion": 2, "redirects": 1, "titles": title,
              "prop": "extracts|revisions", "explaintext": 1, "exsectionformat": "wiki",
              "rvprop": "ids|timestamp"}
    page = api_get(session, params)["query"]["pages"][0]
    revision = page["revisions"][0]
    return {"title": page["title"], "pageid": page["pageid"], "revid": revision["revid"],
            "revision_timestamp": revision["timestamp"], "extract": page.get("extract", "")}


def looks_like_dog(categories: list[str]) -> bool:
    text = " ".join(categories).casefold()
    return any(word in text for word in DOG_CATEGORY_WORDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="download breeds that already have a file again")
    parser.add_argument("--only", nargs="*", help="display names to download (default: all)")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    breeds = [b for b in load_breeds_map() if not args.only or b in args.only]

    report = {"missing": [], "disambiguation": {}, "not_dog_like": {}}
    todo = [d for d in breeds if args.refresh or not (RAW_DIR / f"{slugify(d)}.json").exists()]
    skipped = len(breeds) - len(todo)
    all_titles = sorted({t for d in todo for t in candidate_titles(d)})
    print(f"Resolving {len(all_titles)} candidate titles for {len(todo)} breeds ...", flush=True)
    resolved = resolve_titles(session, all_titles)

    downloaded = 0
    for display in todo:
        chosen = None
        tried = []
        for title in candidate_titles(display):
            info = resolved[title]
            if info["missing"]:
                tried.append(f"{title} (missing)")
            elif info["is_disambiguation"]:
                tried.append(f"{title} (disambiguation)")
                report["disambiguation"].setdefault(display, []).append(info["title"])
            else:
                chosen = info
                break
        if chosen is None:
            report["missing"].append({"breed": display, "tried": tried})
            continue
        page = fetch_page(session, chosen["title"])
        if not looks_like_dog(chosen["categories"]):
            report["not_dog_like"][display] = page["title"]
        record = {"breed": display, "requested_titles": candidate_titles(display),
                  "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "is_disambiguation": False, "categories": chosen["categories"], **page}
        (RAW_DIR / f"{slugify(display)}.json").write_text(
            json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        downloaded += 1
        if downloaded % 25 == 0:
            print(f"  {downloaded}/{len(todo)}", flush=True)

    # Disambiguation pages that were skipped in favour of another candidate are informational;
    # only breeds with no usable page are errors.
    (RAW_DIR / "_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wikipedia: {downloaded} downloaded, {skipped} already present, {len(report['missing'])} missing")
    if report["disambiguation"]:
        print("Disambiguation pages met (skipped):")
        for breed, titles in report["disambiguation"].items():
            print(f"  {breed}: {titles}")
    if report["not_dog_like"]:
        print("Pages without a dog-related category (check these by hand):")
        for breed, title in report["not_dog_like"].items():
            print(f"  {breed} -> {title}")
    for item in report["missing"]:
        print(f"MISSING {item['breed']}: tried {item['tried']}")
    sys.exit(1 if report["missing"] else 0)


if __name__ == "__main__":
    main()
