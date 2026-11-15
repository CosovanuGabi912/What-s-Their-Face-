"""Fingerprints of the ingested data (issue #5): data/ingestion_manifest.json.

The repository is public, so TheDogAPI data, PetFinder data and the Chroma databases cannot be committed.
They are shared privately as a zip. The manifest is safe to commit because it holds no data, only
checksums, chunk counts and the embedding model, and it lets every laptop prove it has the same data:

    ingest.py            refuses to run when a raw input file is missing or differs from the manifest
    validate_ingestion.py  compares raw files, collection sizes and the embedding model with the manifest

Only Part A rewrites it: `python scripts/ingest.py --all --update-manifest`.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

MANIFEST_PATH = config.DATA_DIR / "ingestion_manifest.json"
RAW_DIR = config.DATA_DIR / "raw"

# What each raw group contains. "needed_by" lists the collections whose ingestion reads it.
WIKIPEDIA_DIR = RAW_DIR / "wikipedia"
THEDOGAPI_FILES = [RAW_DIR / "thedogapi" / "breeds.json"]
PETFINDER_FILES = [RAW_DIR / "petfinder" / name
                   for name in ("train.csv", "breed_labels.csv", "color_labels.csv", "state_labels.csv")]
PETFINDER_REQUIRED = {"train.csv", "breed_labels.csv", "color_labels.csv"}  # state_labels.csv is not read by ingest.py

# Every laptop must embed with the same model file: a different quantisation gives slightly different
# vectors, and search quality drops without any error message.
EMBEDDING_MODEL = {
    "model": "nomic-ai/nomic-embed-text-v1.5",
    "quantization": "Q4_K_M",
    "lm_studio_id": "text-embedding-nomic-embed-text-v1.5",
    "document_prefix": config.EMBED_DOC_PREFIX,
    "query_prefix": config.EMBED_QUERY_PREFIX,
}

HOW_TO_GET = {
    "wikipedia": "it is committed in git (data/raw/wikipedia/): run `git checkout -- data/raw/wikipedia`",
    "thedogapi": "get data/raw/thedogapi/breeds.json from Part A (shared privately, it cannot be on the public "
                 "GitHub repo), or run `python scripts/download_thedogapi.py` with your own THEDOGAPI_KEY "
                 "(the checksum then still has to match), or pass --skip-thedogapi to ingest Wikipedia only",
    "petfinder": "run `python scripts/download_petfinder.py` (accept the competition rules on Kaggle and save a "
                 "Kaggle token first)",
}


def rel(path: Path) -> str:
    return path.relative_to(config.DATA_DIR).as_posix()


def sha256_file(path: Path) -> str:
    """SHA-256 of the file with CRLF turned into LF first, so the checksum is the same on Windows and Linux.

    git can convert line endings on checkout (core.autocrlf) and Windows Python writes CRLF; every raw
    input is a text file (JSON or CSV), and its content does not depend on the line ending.
    """
    cr_lf, lf = bytes([13, 10]), bytes([10])
    return hashlib.sha256(path.read_bytes().replace(cr_lf, lf)).hexdigest()


def raw_groups() -> dict[str, list[Path]]:
    """The raw input files that exist now, per group."""
    return {
        "wikipedia": sorted(WIKIPEDIA_DIR.glob("*.json")) if WIKIPEDIA_DIR.exists() else [],
        "thedogapi": [p for p in THEDOGAPI_FILES if p.exists()],
        "petfinder": [p for p in PETFINDER_FILES if p.exists()],
    }


def tree_digest(files: dict[str, str]) -> str:
    """One checksum for a folder: the SHA-256 of the sorted 'name sha256' lines."""
    text = "".join(f"{name} {sha}\n" for name, sha in sorted(files.items()))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD") or None,
            "uncommitted_changes": bool(git("status", "--porcelain", "--untracked-files=no", "--", "scripts",
                                            "config.py", "shared", "data/raw/wikipedia", "data/breeds_map.json"))}


def build_manifest(counts: dict[str, int]) -> dict:
    groups = raw_groups()
    raw: dict[str, dict] = {}
    for group, paths in groups.items():
        files = {rel(path): sha256_file(path) for path in paths}
        raw[group] = {"files": files, "digest": tree_digest(files)}
    return {
        "description": "Fingerprints of the ingested data. No data inside: see scripts/data_manifest.py.",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_commit": git_state(),
        "embedding_model": EMBEDDING_MODEL,
        "chunk_counts": counts,
        "raw_files": raw,
    }


def load_manifest(required: bool = True) -> dict | None:
    if not MANIFEST_PATH.exists():
        if required:
            sys.exit(f"{MANIFEST_PATH} is missing (it is committed in git). "
                     f"Run `git checkout -- data/ingestion_manifest.json`.")
        return None
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def verify_raw(manifest: dict, groups: list[str], *, skip_thedogapi: bool = False,
               all_petfinder: bool = False) -> list[str]:
    """Problems between the raw files on disk and the manifest (empty list = all good)."""
    problems: list[str] = []
    for group in groups:
        if group == "thedogapi" and skip_thedogapi:
            continue
        expected = manifest["raw_files"].get(group, {}).get("files", {})
        if not expected:
            problems.append(f"[{group}] the manifest lists no files")
            continue
        for name, sha in sorted(expected.items()):
            path = config.DATA_DIR / name
            optional = group == "petfinder" and not all_petfinder and Path(name).name not in PETFINDER_REQUIRED
            if not path.exists():
                if not optional:
                    problems.append(f"[{group}] missing {name}: {HOW_TO_GET[group]}")
            elif sha256_file(path) != sha:
                problems.append(f"[{group}] checksum mismatch for {name}: this is not the file the manifest was built "
                                f"from (a different download, or edited). Get the original: {HOW_TO_GET[group]}")
        if group == "wikipedia" and WIKIPEDIA_DIR.exists():
            extra = sorted(rel(p) for p in WIKIPEDIA_DIR.glob("*.json") if rel(p) not in expected)
            if extra:
                problems.append(f"[wikipedia] {len(extra)} file(s) not in the manifest, e.g. {extra[:3]}: chunk IDs would "
                                f"differ from everyone else's")
    return problems
