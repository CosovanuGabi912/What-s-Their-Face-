"""Tests for scripts/ingest.py (issue #5).

Three layers:
    - pure functions (heading mapping, chunking, placeholder names): always run;
    - the chunks and listings built from data/raw/: skipped until the download scripts have run;
    - the Chroma collections written by `python scripts/ingest.py --all`: skipped until they exist.

The "breed is in breeds_map.json" rule applies to the KNOWLEDGE collection only. Listings keep the PetFinder
breed label when it has no breeds_map equivalent (cat breeds, crossbreeds such as "Mixed Breed").
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

import config
from scripts import ingest
from shared.breeds import load_breeds_map
from shared.schemas import CHUNK_ID_PATTERN, LISTING_ID_PATTERN


# --- Pure functions ----------------------------------------------------------------------------------
@pytest.mark.parametrize("heading, expected", [
    ("Exercise", "Exercise"),
    ("Coat care", "Grooming"),      # more specific than "coat"
    ("Coat", "Overview"),
    ("Health and lifespan", "Health"),
    ("Prey and diet", "Feeding"),
    ("Temperament", "Temperament"),
    ("Behavior", "Temperament"),
    ("As pets", "Living Environment"),
    ("Origination of name and split from Chukotka Sled Dogs", None),
    ("Human use", None),
    ("Activities", "Exercise"),
    ("Sport", "Exercise"),
    ("Sports", "Exercise"),
    ("Dog sports", "Exercise"),
    ("Work and sport", "Exercise"),
    ("Care", "Grooming"),
    ("Hypoallergenic coats", "Grooming"),   # before "coats" is Overview
    ("Socialization", "Training"),
    ("Socialisation", "Training"),
    ("Barking", "Temperament"),
    ("Vocalisation", "Temperament"),
    ("Kennel club classification", "History"),
    ("Kennel club recognition", "History"),
    ("Health care", "Health"),              # only the heading "Care" is Grooming
])
def test_heading_mapping(heading, expected):
    assert ingest.map_heading(heading) == expected


def test_subsection_is_mapped_by_its_own_heading_before_the_parent():
    text = "Intro words.\n== Description ==\nA\n=== Behavior ===\nB\n=== Tail ===\nC\n=== Odd name ===\nD\n== Human use ==\nE"
    sections = dict(ingest.parse_sections(text))
    assert "B" in sections["Temperament"]         # own heading wins over parent "Description"
    assert "C" in sections["Overview"] and "A" in sections["Overview"]
    assert "D" in sections["Overview"]            # no match -> parent's section
    assert "E" in sections["Other"]               # top-level without a match -> Other


def test_skipped_headings_drop_their_subsections():
    text = "Lead.\n== History ==\nOld times.\n== References ==\nA reference.\n=== Books ===\nMore references."
    sections = dict(ingest.parse_sections(text))
    assert set(sections) == {"Overview", "History"}


def test_every_rule_targets_a_known_section():
    assert {section for section, _ in ingest.HEADING_RULES} <= set(config.KNOWLEDGE_SECTIONS)


def test_split_words_overlap_and_chunk_ids():
    text = " ".join(str(i) for i in range(600))
    parts = ingest.split_words(text, 250, 20)
    assert len(parts) == 3
    assert parts[0].split()[-20:] == parts[1].split()[:20]
    assert parts[-1].split()[-1] == "599"
    assert ingest.split_words("short text", 250, 20) == ["short text"]
    assert ingest.chunk_id("Siberian Husky", "Exercise", 0) == "kb:siberian-husky:exercise:0"
    assert ingest.chunk_id("Poodle (Toy)", "Living Environment", 1) == "kb:poodle-toy:living-environment:1"


@pytest.mark.parametrize("name", ["No Name", "no name yet", "Puppy 1", "Kitten 3", "", None, "4 Puppies", "Litter of 5", "A", "123"])
def test_placeholder_names(name):
    assert ingest.is_placeholder_name(name)


@pytest.mark.parametrize("name", ["Max", "Luna", "Mimi", "Baby", "Coco 2", "Bella"])
def test_real_names_survive(name):
    assert not ingest.is_placeholder_name(name)


def test_missing_thedogapi_file_is_an_error_unless_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "THEDOGAPI_PATH", tmp_path / "breeds.json")
    with pytest.raises(SystemExit) as error:
        ingest.load_thedogapi()
    assert "download_thedogapi.py" in str(error.value)
    if ingest.WIKIPEDIA_DIR.exists() and any(ingest.WIKIPEDIA_DIR.glob("*.json")):
        build = ingest.build_knowledge(skip_thedogapi=True)
        assert build.chunks and {c.source for c in build.chunks} == {"wikipedia"}


def test_thedogapi_fields_become_sentences_not_dropped():
    record = {"name": "Affenpinscher", "temperament": "Confident, alert", "life_span": "12-15", "origin": "Germany",
              "breed_group": "Toy", "weight": {"metric": "3.2-4.5", "imperial": "7-10"}}
    sections = ingest.thedogapi_sections(record)
    assert "Confident, alert" in sections["Temperament"]
    assert "12-15 years" in sections["Overview"] and "Toy group" in sections["Overview"]
    assert "Germany" in sections["History"]


# --- Chunks built from data/raw ------------------------------------------------------------------------
@pytest.fixture(scope="module")
def knowledge_build():
    if not ingest.WIKIPEDIA_DIR.exists() or not any(ingest.WIKIPEDIA_DIR.glob("*.json")):
        pytest.skip("data/raw/wikipedia is empty: run scripts/download_wikipedia.py")
    return ingest.build_knowledge()


@pytest.fixture(scope="module")
def listings_build():
    if not (ingest.PETFINDER_DIR / "train.csv").exists():
        pytest.skip("data/raw/petfinder/train.csv is missing: run scripts/download_petfinder.py")
    return ingest.build_listings()


def test_knowledge_chunks(knowledge_build):
    chunks = knowledge_build.chunks
    assert len(chunks) > 100
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), [i for i, n in Counter(ids).items() if n > 1][:5]
    assert all(re.match(CHUNK_ID_PATTERN, i) for i in ids)
    breeds = load_breeds_map()
    assert {c.breed for c in chunks} <= set(breeds)
    assert {c.section for c in chunks} <= set(config.KNOWLEDGE_SECTIONS)
    assert {c.source for c in chunks} <= {"wikipedia", "thedogapi"}
    assert all(c.document.startswith(f"{c.breed} - {c.section}\n") for c in chunks)


def test_every_stanford_breed_has_a_chunk(knowledge_build):
    covered = {c.breed for c in knowledge_build.chunks}
    missing = [name for name, entry in load_breeds_map().items() if entry["stanford_label"] and name not in covered]
    assert not missing, missing


def test_baseline_windows(knowledge_build):
    ids = [c.chunk_id for c in knowledge_build.windows]
    assert len(ids) == len(set(ids)) and all(re.match(CHUNK_ID_PATTERN, i) for i in ids)
    assert max(len(c.document.split()) for c in knowledge_build.windows) <= ingest.WINDOW_WORDS + 8


def test_listings(listings_build):
    listings = listings_build.listings
    assert len(listings) > 10_000
    ids = [item.listing_id for item in listings]
    assert len(ids) == len(set(ids)) and all(re.match(LISTING_ID_PATTERN, i) for i in ids)
    for item in listings[:2000]:
        meta = item.metadata
        assert meta["colour"] in config.PET_COLOURS and meta["species"] in ("dog", "cat")
        assert meta["name"] not in item.document.split(". ")[0]   # the name is metadata only
        assert item.document.startswith(f"Species: {meta['species']}. Breed: {meta['breed']}. Colour: {meta['colour']}.")
        assert not ingest.is_placeholder_name(meta["name"])


def test_exclude_ids_keeps_held_out_listings_out(listings_build):
    held_out = {item.listing_id for item in listings_build.listings[:50]}
    rebuilt = ingest.build_listings(held_out)
    assert rebuilt.excluded == 50
    assert held_out.isdisjoint(item.listing_id for item in rebuilt.listings)
    assert len(rebuilt.listings) == len(listings_build.listings) - 50


# --- The collections on disk ---------------------------------------------------------------------------
def _collection(directory, name):
    chromadb = pytest.importorskip("chromadb")
    if not directory.exists():
        pytest.skip(f"{directory} does not exist: run scripts/ingest.py --all")
    try:
        return chromadb.PersistentClient(path=str(directory)).get_collection(name)
    except Exception:  # noqa: BLE001
        pytest.skip(f"collection {name!r} not ingested yet")


def test_knowledge_collection_is_canonical():
    collection = _collection(config.CHROMA_KNOWLEDGE_DIR, "knowledge")
    data = collection.get(include=["metadatas"])
    assert len(data["ids"]) > 100
    chunk_ids = [m["chunk_id"] for m in data["metadatas"]]
    assert len(chunk_ids) == len(set(chunk_ids)) and chunk_ids == data["ids"]
    assert {m["breed"] for m in data["metadatas"]} <= set(load_breeds_map())
    assert {m["section"] for m in data["metadatas"]} <= set(config.KNOWLEDGE_SECTIONS)


def test_listings_collection():
    collection = _collection(config.CHROMA_LISTINGS_DIR, "listings")
    assert collection.count() > 10_000


def test_prune_removes_chunks_the_raw_data_no_longer_produces(tmp_path):
    chromadb = pytest.importorskip("chromadb")
    collection = chromadb.PersistentClient(path=str(tmp_path)).get_or_create_collection("prune_test")
    collection.upsert(ids=["a", "b"], documents=["x", "y"], embeddings=[[1.0, 0.0], [0.0, 1.0]])
    assert ingest.prune(collection, [("a", "x", {})], "prune_test") == 1
    assert collection.get()["ids"] == ["a"]


# --- Manifest checks -----------------------------------------------------------------------------------
@pytest.fixture
def fake_raw(tmp_path, monkeypatch):
    """A tiny data/ folder with one TheDogAPI file and a manifest that describes it."""
    from scripts import data_manifest

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    (tmp_path / "raw" / "thedogapi").mkdir(parents=True)
    file = tmp_path / "raw" / "thedogapi" / "breeds.json"
    file.write_text("[]", encoding="utf-8")
    manifest = {"raw_files": {"thedogapi": {"files": {"raw/thedogapi/breeds.json": data_manifest.sha256_file(file)}}}}
    return data_manifest, manifest, file


def test_verify_raw_accepts_matching_files(fake_raw):
    data_manifest, manifest, _ = fake_raw
    assert data_manifest.verify_raw(manifest, ["thedogapi"]) == []


def test_verify_raw_reports_a_changed_file(fake_raw):
    data_manifest, manifest, file = fake_raw
    file.write_text('[{"name": "other"}]', encoding="utf-8")
    problems = data_manifest.verify_raw(manifest, ["thedogapi"])
    assert len(problems) == 1 and "checksum mismatch" in problems[0] and "breeds.json" in problems[0]


def test_verify_raw_reports_a_missing_file_and_skip_thedogapi_allows_it(fake_raw):
    data_manifest, manifest, file = fake_raw
    file.unlink()
    problems = data_manifest.verify_raw(manifest, ["thedogapi"])
    assert len(problems) == 1 and "missing" in problems[0] and "Part A" in problems[0]
    assert data_manifest.verify_raw(manifest, ["thedogapi"], skip_thedogapi=True) == []


def test_ingest_stops_before_doing_anything_when_a_checksum_differs(fake_raw, monkeypatch):
    data_manifest, manifest, file = fake_raw
    file.write_text('[{"name": "other"}]', encoding="utf-8")
    monkeypatch.setattr(data_manifest, "load_manifest", lambda required=True: manifest)
    monkeypatch.setattr("sys.argv", ["ingest.py", "--collection", "knowledge", "--dry-run"])
    with pytest.raises(SystemExit) as error:
        ingest.main()
    assert "does not match data/ingestion_manifest.json" in str(error.value)


def test_committed_wikipedia_files_match_the_committed_manifest():
    """Fails when a checkout changed line endings or a file differs from what the manifest was built from."""
    from scripts import data_manifest

    manifest = data_manifest.load_manifest(required=False)
    if manifest is None:
        pytest.skip("data/ingestion_manifest.json does not exist yet")
    assert data_manifest.verify_raw(manifest, ["wikipedia"]) == []


def test_checksum_ignores_line_endings(tmp_path):
    from scripts import data_manifest

    lf, crlf = tmp_path / "lf.json", tmp_path / "crlf.json"
    lf.write_bytes(bytes([123, 10, 125, 10]))                  # "{\n}\n"
    crlf.write_bytes(bytes([123, 13, 10, 125, 13, 10]))        # "{\r\n}\r\n"
    assert data_manifest.sha256_file(lf) == data_manifest.sha256_file(crlf)
