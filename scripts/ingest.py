"""Single entry point that turns the raw data into the ChromaDB collections (issue #5).

Reads ONLY from data/raw/ (filled by scripts/download_wikipedia.py, download_thedogapi.py and
download_petfinder.py), so a chunk ID depends on the pinned raw files and nothing else.

Collections:
    knowledge      (CHROMA_KNOWLEDGE_DIR)  one chunk per section of a breed profile, cosine distance
    knowledge_90w  (CHROMA_KNOWLEDGE_DIR)  baseline for Experiment 1: 90-word windows, 15-word overlap
    listings       (CHROMA_LISTINGS_DIR)   one chunk per PetFinder listing

Chunk IDs: "kb:<breed-slug>:<section-slug>:<idx>" (listings: "pf:<PetID>"). Everything is upserted by ID, so
running the script twice adds and updates nothing the second time.

Usage:
    python scripts/ingest.py --collection knowledge
    python scripts/ingest.py --collection listings [--exclude-ids data/heldout_ids.txt]
    python scripts/ingest.py --all
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from shared.breeds import load_breeds_map, normalise, slugify, to_display  # noqa: E402

RAW_DIR = config.DATA_DIR / "raw"
WIKIPEDIA_DIR = RAW_DIR / "wikipedia"
THEDOGAPI_PATH = RAW_DIR / "thedogapi" / "breeds.json"
PETFINDER_DIR = RAW_DIR / "petfinder"

KNOWLEDGE = "knowledge"
KNOWLEDGE_90W = "knowledge_90w"
LISTINGS = "listings"

SECTION_MAX_WORDS = 250     # longer sections are re-split
SECTION_OVERLAP = 20        # words shared by consecutive chunks of one section
WINDOW_WORDS = 90           # Experiment 1 baseline
WINDOW_OVERLAP = 15
MIN_SECTION_WORDS = 15      # Wikipedia sections shorter than this carry no usable fact (TheDogAPI is exempt)
MAX_DESCRIPTION_WORDS = 300  # listing text is capped so one rambling description cannot overflow the embedder
EMBED_BATCH = 32
CHROMA_BATCH = 256

# --- Wikipedia heading -> config.KNOWLEDGE_SECTIONS ---------------------------------------------------
# Rules are tried in order and the first match wins, so more specific keywords come first
# ("coat care" is Grooming before "coat" is Overview). Each heading is matched on its own; only when
# it matches nothing does it inherit the section of its parent heading (see parse_sections).
HEADING_RULES: list[tuple[str, str]] = [
    ("Grooming", r"\b(coat care|grooming|shedding|brushing|bathing|hygiene|hypoallergenic)\b|^care$"),
    ("Exercise", r"\b(exercise|physical activity|activity level|energy level|working ability|activit(y|ies)|"
                 r"sports?|work and sport)\b"),
    ("Training", r"\b(training|trainab\w*|obedience|intelligence|socialis\w*|socializ\w*)\b"),
    ("Feeding", r"\b(diet|feeding|nutrition|food|prey)\b"),
    ("Health", r"\b(health|diseases?|disorders?|illness\w*|medical|lifespan|life span|longevity|mortality|"
               r"genetic (health|disorders?|conditions?)|inherited)\b"),
    ("Temperament", r"\b(temperament|behaviou?rs?|personality|character|disposition|social|barking|vocali[sz]ation)\b"),
    ("Living Environment", r"\b(living|environment|habitat|as (a )?pets?|companion\w*|housing|"
                           r"family life)\b"),
    ("History", r"\b(history|origins?|etymology|lineage|taxonomy|phylogeny|ancestry|evolution|prehistoric|"
                r"founding|recognition|classification)\b"),
    ("Overview", r"\b(description|appearance|characteristics|coat|colou?rs?|size|physical|conformation|"
                 r"anatomy|breed standard|head|eyes?|ears?|nose|tail|gait|height|weight|morphology|build|"
                 r"overview|general)\b"),
]
_COMPILED_RULES = [(section, re.compile(pattern, re.IGNORECASE)) for section, pattern in HEADING_RULES]
SKIPPED_HEADINGS = re.compile(
    r"^(see also|references?|external links?|notes?|footnotes?|further reading|bibliography|sources|"
    r"citations?|gallery|in popular culture|popular culture|notable [\w ]+)$", re.IGNORECASE)
HEADING_LINE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")

assert all(section in config.KNOWLEDGE_SECTIONS for section, _ in HEADING_RULES)


def map_heading(heading: str) -> str | None:
    """Section for one heading, or None when no rule matches (the caller then uses the parent's)."""
    for section, pattern in _COMPILED_RULES:
        if pattern.search(heading):
            return section
    return None


def parse_sections(extract: str) -> list[tuple[str, str]]:
    """Split a TextExtracts plain-text article into (section, body) pairs, in article order.

    The text before the first heading is "Overview". A heading is mapped by itself first; if no rule
    matches it takes the section of its nearest parent heading, and a top-level heading without a
    match becomes "Other". Headings such as "References" are dropped together with everything below them.
    Several headings that map to the same section are merged into one body, so a (breed, section)
    pair is chunked once and its chunk IDs stay unique.
    """
    pieces: list[tuple[str, str]] = []
    stack: list[tuple[int, str | None]] = []  # (level, section or None when skipped)
    current: str | None = "Overview"
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if current is not None and text:
            pieces.append((current, re.sub(r"\n{3,}", "\n\n", text)))
        buffer.clear()

    for line in extract.splitlines():
        match = HEADING_LINE.match(line.strip())
        if not match:
            buffer.append(line)
            continue
        flush()
        level, heading = len(match.group(1)) - 1, match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        parent = stack[-1][1] if stack else "Other"
        skipped_parent = bool(stack) and stack[-1][1] is None
        if skipped_parent or SKIPPED_HEADINGS.match(heading):
            section = None
        else:
            section = map_heading(heading) or parent
        stack.append((level, section))
        current = section
    flush()

    merged: dict[str, list[str]] = {}
    for section, text in pieces:
        merged.setdefault(section, []).append(text)
    return [(section, "\n\n".join(texts)) for section, texts in merged.items()]


# --- Chunking ---------------------------------------------------------------------------------------
def split_words(text: str, size: int, overlap: int) -> list[str]:
    """Windows of `size` words that share `overlap` words; text that fits in one window stays unchanged."""
    words = text.split()
    if len(words) <= size:
        return [text.strip()]
    windows, start, step = [], 0, size - overlap
    while True:
        windows.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            return windows
        start += step


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    breed: str
    section: str
    source: str
    document: str  # what Chroma stores: "{breed} - {section}\n{body}"

    @property
    def metadata(self) -> dict:
        return {"breed": self.breed, "section": self.section, "source": self.source, "chunk_id": self.chunk_id}


def chunk_id(breed: str, section: str, index: int) -> str:
    return f"kb:{slugify(breed)}:{slugify(section)}:{index}"


def _digest(text: str) -> str:
    return hashlib.sha1(" ".join(text.split()).casefold().encode("utf-8")).hexdigest()


# --- TheDogAPI fields -> sentences --------------------------------------------------------------------
def _clean(value) -> str:
    return " ".join(str(value).split()).strip() if value else ""


def thedogapi_sections(record: dict) -> dict[str, str]:
    """TheDogAPI fields as short sentences, grouped by section. They are exempt from the word minimum."""
    name = _clean(record.get("name"))
    overview, temperament, history = [], [], []
    if _clean(record.get("description")):
        overview.append(_clean(record["description"]).rstrip(".") + ".")
    if _clean(record.get("breed_group")):
        overview.append(f"The {name} belongs to the {_clean(record['breed_group'])} group.")
    if _clean(record.get("bred_for")):
        overview.append(f"The {name} was bred for: {_clean(record['bred_for'])}.")
    if _clean(record.get("perfect_for")):
        overview.append(f"The {name} is perfect for: {_clean(record['perfect_for'])}.")
    weight, height = record.get("weight") or {}, record.get("height") or {}
    if weight.get("metric"):
        overview.append(f"A typical {name} weighs {weight['metric']} kg ({weight.get('imperial', '?')} lb).")
    if height.get("metric"):
        overview.append(f"A typical {name} stands {height['metric']} cm ({height.get('imperial', '?')} in) tall.")
    if _clean(record.get("life_span")):
        span = _clean(record["life_span"])
        overview.append(f"The life span of the {name} is {span}{'' if 'year' in span.lower() else ' years'}.")
    if _clean(record.get("temperament")):
        temperament.append(f"The {name} is typically described as: {_clean(record['temperament'])}.")
    if _clean(record.get("history")):
        history.append(_clean(record["history"]).rstrip(".") + ".")
    if _clean(record.get("origin")):
        history.append(f"The {name} originates from {_clean(record['origin'])}.")
    sections = {"Overview": overview, "Temperament": temperament, "History": history}
    return {section: " ".join(parts) for section, parts in sections.items() if parts}


# --- Knowledge chunks ---------------------------------------------------------------------------------
def load_wikipedia(display: str) -> dict | None:
    path = WIKIPEDIA_DIR / f"{slugify(display)}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_thedogapi() -> dict[str, dict]:
    if not THEDOGAPI_PATH.exists():
        sys.exit(f"{THEDOGAPI_PATH} is missing. Run `python scripts/download_thedogapi.py` "
                 "(needs THEDOGAPI_KEY in .env), or pass --skip-thedogapi to ingest Wikipedia only.")
    return {_clean(item.get("name")): item for item in json.loads(THEDOGAPI_PATH.read_text(encoding="utf-8"))}


@dataclass
class KnowledgeBuild:
    chunks: list[KnowledgeChunk]
    windows: list[KnowledgeChunk]
    duplicates: int
    without_wikipedia: list[str]
    without_thedogapi: list[str]  # breeds that map to a TheDogAPI name but have no record in the file


def build_knowledge(skip_thedogapi: bool = False) -> KnowledgeBuild:
    """Section chunks (Wikipedia first, then TheDogAPI) and the 90-word baseline windows for every breed."""
    entries = load_breeds_map()
    dogapi = {} if skip_thedogapi else load_thedogapi()
    chunks: list[KnowledgeChunk] = []
    windows: list[KnowledgeChunk] = []
    duplicates = 0
    without_wikipedia: list[str] = []
    without_thedogapi: list[str] = []

    for display in sorted(entries, key=str.casefold):
        sections: list[tuple[str, str, str]] = []  # (section, body, source), Wikipedia before TheDogAPI
        page = load_wikipedia(display)
        if page is None:
            without_wikipedia.append(display)
        else:
            for section, body in parse_sections(page["extract"]):
                if len(body.split()) >= MIN_SECTION_WORDS:
                    sections.append((section, body, "wikipedia"))
        api_name = entries[display].get("thedogapi_name")
        if api_name and not skip_thedogapi:
            if api_name in dogapi:
                sections += [(s, b, "thedogapi") for s, b in thedogapi_sections(dogapi[api_name]).items()]
            else:
                without_thedogapi.append(display)

        next_index: dict[str, int] = {}
        seen: set[tuple[str, str]] = set()
        for section, body, source in sections:
            key = (section, _digest(body))
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            for part in split_words(body, SECTION_MAX_WORDS, SECTION_OVERLAP):
                index = next_index.get(section, 0)
                next_index[section] = index + 1
                chunks.append(KnowledgeChunk(chunk_id(display, section, index), display, section, source,
                                             f"{display} - {section}\n{part}"))

        # Baseline: the same cleaned text of the breed, with no section structure, in fixed windows.
        # The breed name heads every window (as it does in section chunks) so the experiment isolates the chunking.
        # The section is "Other" because a window can span several sections.
        flat = " ".join(body for _, body, _ in sorted(sections, key=lambda item: item[2] != "wikipedia"))
        if flat.strip():
            for index, part in enumerate(split_words(flat, WINDOW_WORDS, WINDOW_OVERLAP)):
                cid = f"kb:{slugify(display)}:window:{index}"
                windows.append(KnowledgeChunk(cid, display, "Other", "wikipedia" if page else "thedogapi",
                                              f"{display}\n{part}"))
    return KnowledgeBuild(chunks, windows, duplicates, without_wikipedia, without_thedogapi)


# --- Listings ---------------------------------------------------------------------------------------
PLACEHOLDER_NAME = re.compile(
    r"""^(
        (no\s*name|noname|unnamed|un-named|no\s*name\s*yet|none|null|nil|n/?a|unknown|tbc|tba|nameless)(\s*\d+)?
      | (puppy|puppies|pup|pups|kitten|kittens|kitty|cat|cats|dog|dogs|male|female|stray|adult|
         lil|little)\s*(no\.?|\#)?\s*\d*
      | \d+\s*(puppy|puppies|pups?|kittens?|kitties|cats?|dogs?|babies)
      | .*\blitter\b.*
      | .*\b(puppies|kittens)\b.*
    )$""", re.IGNORECASE | re.VERBOSE)

SIZES = {0: "unknown", 1: "small", 2: "medium", 3: "large", 4: "large"}  # 4 = extra large -> "large" (describe_pet vocabulary)


def is_placeholder_name(name: str | None) -> bool:
    cleaned = " ".join((name or "").split())
    if len(cleaned) < 2 or not any(ch.isalpha() for ch in cleaned):
        return True
    return bool(PLACEHOLDER_NAME.match(cleaned.strip(" .-_!")))


def age_text(months: int) -> str:
    if months < 12:
        return f"{months} month{'s' if months != 1 else ''}"
    years = months // 12
    return f"{years} year{'s' if years != 1 else ''}"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_exclude_ids(path: Path | None) -> set[str]:
    """Listing IDs to keep out (the Week 9 held-out set): a JSON list or one ID per line, "pf:<PetID>" or bare PetID."""
    if path is None:
        return set()
    text = path.read_text(encoding="utf-8").strip()
    items = json.loads(text) if text.startswith("[") else text.splitlines()
    return {("pf:" + item.strip()) if not item.strip().startswith("pf:") else item.strip()
            for item in items if item.strip() and not item.strip().startswith("#")}


@dataclass(frozen=True)
class Listing:
    listing_id: str
    document: str
    metadata: dict


@dataclass
class ListingsBuild:
    listings: list[Listing]
    skipped: dict[str, int]
    excluded: int


def build_listings(exclude_ids: set[str] | None = None) -> ListingsBuild:
    exclude_ids = exclude_ids or set()
    breed_names = {int(r["BreedID"]): r["BreedName"].strip() for r in read_csv(PETFINDER_DIR / "breed_labels.csv")}
    colours = {int(r["ColorID"]): r["ColorName"].strip().lower() for r in read_csv(PETFINDER_DIR / "color_labels.csv")}
    skipped = {"placeholder name": 0, "litter (Quantity > 1)": 0, "colour not in PET_COLOURS": 0, "duplicate PetID": 0}
    listings: list[Listing] = []
    seen: set[str] = set()
    excluded = 0

    for row in read_csv(PETFINDER_DIR / "train.csv"):
        listing_id = f"pf:{row['PetID'].strip()}"
        if listing_id in exclude_ids:
            excluded += 1
            continue
        if listing_id in seen:
            skipped["duplicate PetID"] += 1
            continue
        seen.add(listing_id)
        if int(row["Quantity"] or 1) > 1:
            skipped["litter (Quantity > 1)"] += 1
            continue
        if is_placeholder_name(row["Name"]):
            skipped["placeholder name"] += 1
            continue
        colour = colours.get(int(row["Color1"] or 0))
        if colour not in config.PET_COLOURS:
            skipped["colour not in PET_COLOURS"] += 1
            continue
        species = "dog" if row["Type"] == "1" else "cat"
        raw_breed = breed_names.get(int(row["Breed1"] or 0), "Mixed Breed") if int(row["Breed1"] or 0) else "Mixed Breed"
        breed = (to_display(raw_breed) if species == "dog" else None) or raw_breed
        size = SIZES.get(int(row["MaturitySize"] or 0), "unknown")
        description = " ".join(words) if (words := (row["Description"] or "").split()[:MAX_DESCRIPTION_WORDS]) else ""
        document = (f"Species: {species}. Breed: {breed}. Colour: {colour}. Size: {size}. "
                    f"Age: {age_text(int(row['Age'] or 0))}. {description}").strip()
        listings.append(Listing(listing_id, document, {
            "name": " ".join(row["Name"].split()), "species": species, "breed": breed,
            "colour": colour, "size": size, "listing_id": listing_id}))
    return ListingsBuild(listings, skipped, excluded)


# --- Embedding and Chroma -------------------------------------------------------------------------------
class Embedder:
    """LM Studio's OpenAI-compatible /embeddings endpoint with the nomic document prefix."""

    def __init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=config.LM_STUDIO_BASE_URL, api_key=config.LM_STUDIO_API_KEY, timeout=300)

    def check(self) -> None:
        try:
            self.embed(["connection check"])
        except Exception as error:  # noqa: BLE001 - any failure here means ingestion cannot run
            sys.exit(f"Cannot embed with model {config.EMBED_MODEL_ID!r} at {config.LM_STUDIO_BASE_URL}: {error}\n"
                     "Start the LM Studio server and load the embedding model (EMBEDDING_MODEL in .env).")

    def embed(self, documents: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=config.EMBED_MODEL_ID, input=[config.EMBED_DOC_PREFIX + doc for doc in documents])
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


def batches(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def upsert(collection, rows: list[tuple[str, str, dict]], embedder: Embedder | None) -> dict[str, int]:
    """Upsert (id, document, metadata) rows. Unchanged rows are not embedded again."""
    counts = {"added": 0, "updated": 0, "skipped": 0}
    for batch in batches(rows, CHROMA_BATCH):
        existing = collection.get(ids=[row[0] for row in batch], include=["documents", "metadatas"])
        known = {i: (d, m) for i, d, m in zip(existing["ids"], existing["documents"], existing["metadatas"])}
        todo = []
        for row_id, document, metadata in batch:
            if row_id not in known:
                counts["added"] += 1
            elif known[row_id] != (document, metadata):
                counts["updated"] += 1
            else:
                counts["skipped"] += 1
                continue
            todo.append((row_id, document, metadata))
        if embedder is None:  # --dry-run
            continue
        for part in batches(todo, EMBED_BATCH):
            collection.upsert(ids=[r[0] for r in part], documents=[r[1] for r in part],
                              metadatas=[r[2] for r in part],
                              embeddings=embedder.embed([r[1] for r in part]))
    return counts


def get_collection(directory: Path, name: str):
    import chromadb

    directory.mkdir(parents=True, exist_ok=True)  # NOTE: callers skip this in --dry-run (see _collection_for)
    client = chromadb.PersistentClient(path=str(directory))
    return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


def prune(collection, rows: list[tuple[str, str, dict]], name: str) -> int:
    """Delete chunks whose ID the raw data no longer produces (e.g. after a heading rule changed),
    so a collection always mirrors data/raw/ exactly."""
    keep = {row[0] for row in rows}
    stale = [i for i in collection.get(include=[])["ids"] if i not in keep]
    for part in batches(stale, CHROMA_BATCH):
        collection.delete(ids=part)
    if stale:
        print(f"  {name}: removed {len(stale)} stale chunks no longer produced by data/raw/", file=sys.stderr)
    return len(stale)


class _EmptyCollection:
    """Stand-in for a collection in --dry-run, so a dry run creates no files on disk."""

    def get(self, ids=None, include=None):
        return {"ids": [], "documents": [], "metadatas": []}


def _collection_for(directory: Path, name: str, dry_run: bool):
    if dry_run and not (directory / "chroma.sqlite3").exists():
        return _EmptyCollection()
    return get_collection(directory, name)


def ingest_knowledge(embedder: Embedder | None, skip_thedogapi: bool = False) -> None:
    build = build_knowledge(skip_thedogapi)
    if build.without_wikipedia:
        print(f"WARNING: no Wikipedia file for {len(build.without_wikipedia)} breeds "
              f"(run scripts/download_wikipedia.py): {build.without_wikipedia[:10]}", file=sys.stderr)
    if build.without_thedogapi:
        print(f"WARNING: TheDogAPI name not in breeds.json for: {build.without_thedogapi}", file=sys.stderr)
    for name, chunks in ((KNOWLEDGE, build.chunks), (KNOWLEDGE_90W, build.windows)):
        collection = _collection_for(config.CHROMA_KNOWLEDGE_DIR, name, embedder is None)
        rows = [(c.chunk_id, c.document, c.metadata) for c in chunks]
        counts = upsert(collection, rows, embedder)
        if embedder is not None:
            prune(collection, rows, name)
        skipped = counts["skipped"] + (build.duplicates if name == KNOWLEDGE else 0)
        label = "Knowledge collection:" if name == KNOWLEDGE else "Knowledge 90w collection:"
        print(f"{label} {counts['added']} added, {counts['updated']} updated, {skipped} skipped (duplicates)")


def ingest_listings(embedder: Embedder | None, exclude_path: Path | None) -> None:
    exclude_ids = load_exclude_ids(exclude_path)
    build = build_listings(exclude_ids)
    collection = _collection_for(config.CHROMA_LISTINGS_DIR, LISTINGS, embedder is None)
    removed = 0
    if exclude_ids and embedder is not None:  # a held-out listing must not stay in a collection built earlier
        present = collection.get(ids=sorted(exclude_ids))["ids"]
        if present:
            collection.delete(ids=present)
            removed = len(present)
    metadata_rows = [(item.listing_id, item.document, item.metadata) for item in build.listings]
    counts = upsert(collection, metadata_rows, embedder)
    if embedder is not None:
        prune(collection, metadata_rows, LISTINGS)
    placeholder = build.skipped["placeholder name"] + build.skipped["litter (Quantity > 1)"]
    print(f"Listings collection:  {counts['added']} added, {counts['updated']} updated, "
          f"{placeholder} skipped (placeholder names)")
    detail = ", ".join(f"{count} {reason}" for reason, count in build.skipped.items() if count)
    print(f"  skipped before ingestion: {detail or 'none'}; {counts['skipped']} unchanged; "
          f"{build.excluded} held out via --exclude-ids ({removed} removed from the collection)", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--collection", choices=["knowledge", "listings"])
    which.add_argument("--all", action="store_true", help="knowledge and listings")
    parser.add_argument("--exclude-ids", type=Path,
                        help="listings only: file with listing IDs to leave out (held-out set), one per line or a JSON list")
    parser.add_argument("--skip-thedogapi", action="store_true",
                        help="knowledge only: ingest Wikipedia without data/raw/thedogapi/breeds.json "
                             "(otherwise a missing file is an error)")
    parser.add_argument("--dry-run", action="store_true", help="count what would be added or updated without calling the embedder")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    embedder = None if args.dry_run else Embedder()
    if embedder:
        embedder.check()
    if args.all or args.collection == "knowledge":
        ingest_knowledge(embedder, args.skip_thedogapi)
    if args.all or args.collection == "listings":
        ingest_listings(embedder, args.exclude_ids)


if __name__ == "__main__":
    main()
