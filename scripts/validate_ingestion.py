"""Print breed and section coverage of the ingested collections (issue #5).

Checks that every one of the 120 Stanford breeds has at least one chunk in the knowledge collection,
that every `breed` metadata value is a display name from breeds_map.json, and that chunk IDs are unique.
It also compares this laptop with data/ingestion_manifest.json: checksums of the raw files, number of chunks
per collection and the embedding model (the quantisation LM Studio reports, and whether re-embedding stored
chunks gives the same vectors). Any mismatch is reported as MISMATCH and fails the run.
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
import scripts.data_manifest as manifest_tools  # noqa: E402
from shared.breeds import load_breeds_map  # noqa: E402

VECTOR_MATCH = 0.999  # cosine between a stored vector and a fresh one; the same model file gives ~1.0


def collection_rows(directory: Path, name: str) -> dict:
    import chromadb

    client = chromadb.PersistentClient(path=str(directory))
    try:
        collection = client.get_collection(name)
    except Exception:  # noqa: BLE001 - chromadb raises different errors by version
        sys.exit(f"Collection {name!r} not found in {directory}. Run: python scripts/ingest.py --all")
    return collection.get(include=["metadatas"])


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5))


def lm_studio_model_info() -> dict | None:
    """What LM Studio reports for the embedding model (its native API names the quantisation), or None."""
    import json
    import urllib.request

    base = config.LM_STUDIO_BASE_URL.removesuffix("/v1").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/v0/models", timeout=10) as response:
            models = json.load(response)["data"]
    except Exception:  # noqa: BLE001 - server down, or not LM Studio
        return None
    return next((m for m in models if m.get("id") == config.EMBED_MODEL_ID), {})


def check_manifest(problems: list[str], warnings: list[str]) -> None:
    """Compare raw files, collection sizes and the embedding model with data/ingestion_manifest.json."""
    manifest = manifest_tools.load_manifest(required=False)
    print("Manifest check (data/ingestion_manifest.json):")
    if manifest is None:
        problems.append("MISMATCH data/ingestion_manifest.json is missing: run `git checkout -- data/ingestion_manifest.json`")
        return
    source = manifest["source_commit"].get("commit") or "unknown"
    print(f"  built from commit {source[:10]} on {manifest['generated_at']}")

    raw_problems = manifest_tools.verify_raw(manifest, ["wikipedia", "thedogapi", "petfinder"], all_petfinder=True)
    print(f"  raw files: {'all checksums match' if not raw_problems else str(len(raw_problems)) + ' problem(s)'}")
    problems += [f"MISMATCH raw files {p}" for p in raw_problems]

    import chromadb

    for directory, name in ((config.CHROMA_KNOWLEDGE_DIR, "knowledge"), (config.CHROMA_KNOWLEDGE_DIR, "knowledge_90w"),
                            (config.CHROMA_LISTINGS_DIR, "listings")):
        expected = manifest["chunk_counts"].get(name)
        try:
            actual = chromadb.PersistentClient(path=str(directory)).get_collection(name).count()
        except Exception:  # noqa: BLE001
            actual = None
        status = "ok" if actual == expected else "MISMATCH"
        print(f"  {name}: {actual} chunks, manifest says {expected} -> {status}")
        if actual != expected:
            problems.append(f"MISMATCH {name} has {actual} chunks, the manifest has {expected} "
                            f"(different data, or built with --skip-thedogapi)")

    wanted = manifest["embedding_model"]
    info = lm_studio_model_info()
    if info is None:
        warnings.append(f"could not reach LM Studio at {config.LM_STUDIO_BASE_URL}: embedding model not checked. "
                        f"Start the server with {wanted['model']} {wanted['quantization']} loaded and run this again.")
        return
    if not info:
        problems.append(f"MISMATCH LM Studio does not list the model {config.EMBED_MODEL_ID!r}: load "
                        f"{wanted['model']} {wanted['quantization']} (or set EMBEDDING_MODEL in .env)")
        return
    quantization = info.get("quantization")
    print(f"  embedding model in LM Studio: {info.get('publisher')}/{config.EMBED_MODEL_ID} {quantization}, "
          f"manifest wants {wanted['model']} {wanted['quantization']}")
    if quantization != wanted["quantization"]:
        problems.append(f"MISMATCH embedding model is {quantization}, the data was built with {wanted['quantization']}: "
                        "search quality drops silently when the model files differ. Load the Q4_K_M file.")

    from scripts.ingest import Embedder

    collection = chromadb.PersistentClient(path=str(config.CHROMA_KNOWLEDGE_DIR)).get_collection("knowledge")
    sample = collection.get(limit=5, include=["documents", "embeddings"])
    fresh = Embedder().embed(list(sample["documents"]))
    scores = [cosine(list(old), new) for old, new in zip(sample["embeddings"], fresh)]
    worst = min(scores)
    print(f"  re-embedding {len(scores)} stored chunks: lowest cosine to the stored vector {worst:.5f} "
          f"(needs >= {VECTOR_MATCH}) -> {'ok' if worst >= VECTOR_MATCH else 'MISMATCH'}")
    if worst < VECTOR_MATCH:
        problems.append("MISMATCH the loaded embedding model gives different vectors than the stored ones: it is not the "
                        "model file the data was built with")


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

    warnings: list[str] = []
    check_manifest(problems, warnings)

    for warning in warnings:
        print(f"WARNING: {warning}")
    for problem in problems:
        print(f"PROBLEM: {problem}")
    print("OK" if not problems else f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
