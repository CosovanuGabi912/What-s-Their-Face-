"""Schema and integrity tests for eval/golden.jsonl (issue #8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import config
from shared.schemas import CHUNK_ID_PATTERN, GoldenItem, load_breeds_map


@pytest.fixture(scope="module")
def golden_items():
    assert config.GOLDEN_PATH.exists(), f"Golden dataset missing at {config.GOLDEN_PATH}"
    items = []
    with config.GOLDEN_PATH.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                pytest.fail(f"Line {line_num} in {config.GOLDEN_PATH} is not valid JSON: {exc}")
            item = GoldenItem.model_validate(data)
            items.append(item)
    return items


@pytest.fixture(scope="module")
def val_files():
    val_path = config.DATA_DIR / "splits" / "stanford_dogs_val.json"
    if not val_path.exists():
        return set()
    data = json.loads(val_path.read_text(encoding="utf-8"))
    return {Path(f).name for f in data.get("val_files", [])}


@pytest.fixture(scope="module")
def breeds_map():
    return load_breeds_map()


def test_golden_items_counts(golden_items):
    part_a = [i for i in golden_items if i.part == "A"]
    part_b = [i for i in golden_items if i.part == "B"]
    part_c = [i for i in golden_items if i.part == "C"]

    assert len(part_a) >= 100, f"Expected >= 100 Part A items, got {len(part_a)}"
    assert len(part_b) >= 60, f"Expected >= 60 Part B items, got {len(part_b)}"
    assert len(part_c) >= 15, f"Expected >= 15 Part C items, got {len(part_c)}"
    assert len(golden_items) >= 175, f"Expected >= 175 total items, got {len(golden_items)}"


def test_all_ids_are_unique(golden_items):
    ids = [i.id for i in golden_items]
    assert len(ids) == len(set(ids)), f"Duplicate IDs found in golden items: {[k for k, v in collections.Counter(ids).items() if v > 1]}"


def test_part_a_source_and_images(golden_items, val_files, breeds_map):
    part_a = [i for i in golden_items if i.part == "A"]
    sources = set()

    for item in part_a:
        assert len(item.source_ids) >= 1, f"Part A item {item.id} has no source_ids"
        source = item.source_ids[0]
        assert source.startswith("stanford_dogs/test/"), f"{item.id} source {source!r} must start with 'stanford_dogs/test/'"
        assert source not in sources, f"Duplicate Part A source path {source!r}"
        sources.add(source)

        filename = Path(source).name
        assert filename not in val_files, f"{item.id} source {source!r} is in the validation split"

        # Check image exists in eval/images/
        image_id = item.input.get("image_id")
        assert image_id, f"{item.id} missing input.image_id"
        image_path = config.EVAL_IMAGES_DIR / image_id
        assert image_path.exists(), f"Image file {image_path} missing on disk"

        # Check expected breed is a canonical display name
        expected_breed = item.expected.get("breed")
        assert expected_breed in breeds_map, f"{item.id} expected breed {expected_breed!r} not in breeds_map.json"


def test_part_b_chunks_and_gold_spans(golden_items):
    part_b = [i for i in golden_items if i.part == "B"]
    breeds = set()
    sections = set()

    for item in part_b:
        query = item.input.get("query")
        assert query and isinstance(query, str), f"{item.id} missing valid query string"

        source_chunk_ids = item.expected.get("source_chunk_ids", [])
        assert source_chunk_ids, f"{item.id} missing expected.source_chunk_ids"
        assert item.source_ids == source_chunk_ids, f"{item.id} source_ids must match expected.source_chunk_ids"

        for cid in source_chunk_ids:
            import re
            assert re.match(CHUNK_ID_PATTERN, cid), f"{item.id} chunk ID {cid!r} does not match CHUNK_ID_PATTERN"
            parts = cid.split(":")
            breeds.add(parts[1])
            sections.add(parts[2])

        gold_span = item.expected.get("gold_span")
        assert gold_span and isinstance(gold_span, str), f"{item.id} missing non-empty gold_span"

        answer_contains = item.expected.get("answer_contains")
        assert isinstance(answer_contains, list) and len(answer_contains) >= 1, f"{item.id} missing expected.answer_contains list"

    assert len(breeds) >= 10, f"Part B must cover >= 10 breeds, got {len(breeds)}"
    assert len(sections) >= 5, f"Part B must cover >= 5 sections, got {len(sections)}"


def test_part_c_response_types(golden_items):
    part_c = [i for i in golden_items if i.part == "C"]
    allowed_types = {"vet_referral", "i_dont_know", "out_of_scope"}

    type_counts = {}
    for item in part_c:
        msg = item.input.get("message")
        assert msg and isinstance(msg, str), f"{item.id} missing valid message"

        resp_type = item.expected.get("response_type")
        assert resp_type in allowed_types, f"{item.id} invalid response_type {resp_type!r}"
        type_counts[resp_type] = type_counts.get(resp_type, 0) + 1

    assert type_counts.get("vet_referral", 0) >= 6, f"Expected >= 6 vet_referral, got {type_counts.get('vet_referral')}"
    assert type_counts.get("i_dont_know", 0) >= 5, f"Expected >= 5 i_dont_know, got {type_counts.get('i_dont_know')}"
    assert type_counts.get("out_of_scope", 0) >= 4, f"Expected >= 4 out_of_scope, got {type_counts.get('out_of_scope')}"
