"""Schema, integrity, and holdout tests for Golden Parts E and F (issue #17)."""

from __future__ import annotations

import json
from pathlib import Path
import re
import pytest

import config
from shared.schemas import GoldenItem, LISTING_ID_PATTERN


@pytest.fixture(scope="module")
def golden_items() -> list[GoldenItem]:
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
def holdout_ids() -> list[str]:
    path = config.EVAL_DIR / "holdout_listing_ids.txt"
    assert path.exists(), f"Holdout IDs file missing at {path}"
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines


def test_golden_ef_counts(golden_items: list[GoldenItem]):
    part_e = [it for it in golden_items if it.part == "E"]
    part_f = [it for it in golden_items if it.part == "F"]

    assert len(part_e) >= 50, f"Expected >= 50 Part E items, got {len(part_e)}"
    assert len(part_f) >= 10, f"Expected >= 10 Part F items, got {len(part_f)}"


def test_holdout_listing_ids_file(holdout_ids: list[str], golden_items: list[GoldenItem]):
    assert len(holdout_ids) >= 50, f"Expected >= 50 holdout listing IDs, got {len(holdout_ids)}"
    assert len(holdout_ids) == len(set(holdout_ids)), "Duplicate listing IDs found in holdout_listing_ids.txt"

    for lid in holdout_ids:
        assert re.match(LISTING_ID_PATTERN, lid), f"Listing ID {lid!r} does not match {LISTING_ID_PATTERN}"

    part_e = [it for it in golden_items if it.part == "E"]
    part_e_source_ids = [it.source_ids[0] for it in part_e]
    assert set(holdout_ids) == set(part_e_source_ids), "holdout_listing_ids.txt IDs must match Part E source_ids"


def test_part_e_schema_and_images(golden_items: list[GoldenItem]):
    part_e = [it for it in golden_items if it.part == "E"]
    seen_ids = set()

    for it in part_e:
        assert it.id.startswith("E_"), f"Item id {it.id!r} should start with 'E_'"
        assert it.id not in seen_ids, f"Duplicate id {it.id}"
        seen_ids.add(it.id)

        # Input fields
        image_id = it.input.get("image_id")
        assert image_id and isinstance(image_id, str), f"{it.id} missing input.image_id"
        image_path = config.EVAL_IMAGES_DIR / image_id
        assert image_path.exists(), f"Image {image_path} does not exist on disk"

        desc = it.input.get("description")
        assert desc and isinstance(desc, str), f"{it.id} missing input.description"

        # Expected fields
        real_name = it.expected.get("real_name")
        assert real_name and isinstance(real_name, str), f"{it.id} missing expected.real_name"

        # Source IDs
        assert len(it.source_ids) == 1, f"{it.id} should have exactly 1 source_id"
        assert re.match(LISTING_ID_PATTERN, it.source_ids[0]), f"{it.id} invalid source_id {it.source_ids[0]}"

        # Name anonymity check: real_name must not be in description
        pattern = rf"\b{re.escape(real_name)}\b"
        assert not re.search(pattern, desc, re.IGNORECASE), f"{it.id} leaks real_name {real_name!r} into description: {desc!r}"


def test_part_f_schema_and_journeys(golden_items: list[GoldenItem]):
    part_f = [it for it in golden_items if it.part == "F"]
    seen_ids = set()
    rejection_loop_count = 0

    allowed_tools = {"identify_breed", "search_knowledge", "get_breed_traits", "recommend_breeds", "suggest_names"}

    for it in part_f:
        assert it.id.startswith("F_"), f"Item id {it.id!r} should start with 'F_'"
        assert it.id not in seen_ids, f"Duplicate id {it.id}"
        seen_ids.add(it.id)

        turns = it.input.get("scripted_turns")
        assert isinstance(turns, list) and len(turns) >= 3, f"{it.id} scripted_turns must be a list of >= 3 turns"

        for turn_idx, turn in enumerate(turns):
            assert turn.get("role") == "user", f"{it.id} turn {turn_idx} role must be 'user'"
            content = turn.get("content")
            assert content and isinstance(content, str), f"{it.id} turn {turn_idx} missing content"
            if "image_id" in turn:
                img_path = config.EVAL_IMAGES_DIR / turn["image_id"]
                assert img_path.exists(), f"{it.id} turn {turn_idx} image {img_path} does not exist"

        expected = it.expected
        seq = expected.get("tool_sequence")
        assert isinstance(seq, list) and len(seq) >= 3, f"{it.id} missing or short tool_sequence"
        for tool_name in seq:
            assert tool_name in allowed_tools, f"{it.id} unknown tool {tool_name!r} in expected tool_sequence"

        outcome = expected.get("final_outcome")
        assert outcome == "names_suggested", f"{it.id} expected final_outcome 'names_suggested', got {outcome!r}"

        # Count journeys with rejection loops
        suggest_count = seq.count("suggest_names")
        if suggest_count >= 2:
            rejection_loop_count += 1

    assert rejection_loop_count >= 2, f"Expected >= 2 journeys with rejection loop, got {rejection_loop_count}"
