"""Print breed and section coverage of the ingested collections (issue #5).

Checks that every one of the 120 Stanford breeds has at least one chunk in the knowledge collection,
that every `breed` metadata value is a display name from breeds_map.json, and that chunk IDs are unique.
Exit code 1 when one of these fails, so it can run in CI or before a PR.

Usage:
    python scripts/validate_ingestion.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from shared.breeds import load_breeds_map  # noqa: E402


def collection_rows(directory: Path, name: str) -> dict:
    import chromadb

    client = chromadb.PersistentClient(path=str(directory))
    try:
        collection = client.get_collection(name)
    except Exception:  # noqa: BLE001 - chromadb raises different errors by version
        sys.exit(f"Collection {name!r} not found in {directory}. Run: python scripts/ingest.py --all")
    return collection.get(include=["metadatas"])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    entries = load_breeds_map()
    problems: list[str] = []

    knowledge = collection_rows(config.CHROMA_KNOWLEDGE_DIR, "knowledge")
    metadatas = knowledge["metadatas"]
    per_breed = Counter(m["breed"] for m in metadatas)
    per_section = Counter(m["section"] for m in metadatas)
    per_source = Counter(m["source"] for m in metadatas)
    print(f"Knowledge collection: {len(metadatas)} chunks, {len(per_breed)} breeds")
    print("  by source:  " + ", ".join(f"{k} {v}" for k, v in sorted(per_source.items())))
    print("  by section: " + ", ".join(f"{s} {per_section.get(s, 0)}" for s in config.KNOWLEDGE_SECTIONS))

    ids = knowledge["ids"]
    if len(ids) != len(set(ids)) or len({m["chunk_id"] for m in metadatas}) != len(metadatas):
        problems.append("duplicate chunk IDs in the knowledge collection")
    unknown = sorted(set(per_breed) - set(entries))
    if unknown:
        problems.append(f"breed metadata not in breeds_map.json: {unknown[:10]}")
    bad_sections = sorted(set(per_section) - set(config.KNOWLEDGE_SECTIONS))
    if bad_sections:
        problems.append(f"sections outside config.KNOWLEDGE_SECTIONS: {bad_sections}")

    stanford = sorted(name for name, entry in entries.items() if entry["stanford_label"])
    missing_stanford = [name for name in stanford if per_breed.get(name, 0) == 0]
    print(f"  Stanford breeds with at least one chunk: {len(stanford) - len(missing_stanford)}/{len(stanford)}")
    if missing_stanford:
        problems.append(f"Stanford breeds without chunks: {missing_stanford}")
    no_chunks = [name for name in entries if per_breed.get(name, 0) == 0]
    print(f"  Breeds in breeds_map.json without chunks: {len(no_chunks)}" + (f" {no_chunks}" if no_chunks else ""))
    thin = sorted(name for name, count in per_breed.items() if count <= 2)
    print(f"  Breeds with only 1-2 chunks: {len(thin)}")
    sections_per_breed = Counter(len({m['section'] for m in metadatas if m['breed'] == b}) for b in per_breed) \
        if len(per_breed) < 400 else Counter()
    if sections_per_breed:
        print("  Distinct sections per breed: " + ", ".join(f"{n} sections: {c} breeds"
                                                          for n, c in sorted(sections_per_breed.items())))

    baseline = collection_rows(config.CHROMA_KNOWLEDGE_DIR, "knowledge_90w")
    print(f"Knowledge 90w collection: {len(baseline['ids'])} windows, "
          f"{len({m['breed'] for m in baseline['metadatas']})} breeds")

    listings = collection_rows(config.CHROMA_LISTINGS_DIR, "listings")
    metas = listings["metadatas"]
    print(f"Listings collection: {len(metas)} listings")
    print("  by species: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(m['species'] for m in metas).items())))
    print("  by colour:  " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(m['colour'] for m in metas).items())))
    if set(m["colour"] for m in metas) - set(config.PET_COLOURS):
        problems.append("listing colours outside config.PET_COLOURS")

    for problem in problems:
        print(f"PROBLEM: {problem}")
    print("OK" if not problems else f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
