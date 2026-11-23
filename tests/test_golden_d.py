"""Golden Part D (issue #12): items in eval/golden.jsonl and the independent rules that derive them."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

import config
from scripts import validate_golden_d as gd

FLAT_9H_ALLERGIES = {"home": "flat", "hours_alone": 9, "daily_walks_min": 20,
                     "kids_at_home": False, "has_allergies": True, "experience": "none"}


@pytest.fixture(scope="module")
def akc():
    return gd.load_akc()


@pytest.fixture(scope="module")
def items():
    return gd.read_golden(config.GOLDEN_PATH)


def test_part_d_items_pass_the_validator(items, akc):
    assert gd.validate(items, akc) == []


def test_items_match_the_rules_exactly(items, akc):
    for item in (i for i in items if i["part"] == "D"):
        assert item["expected"] == gd.expected_breeds(item["input"], akc), item["id"]


def test_milestone_example_breeds(akc):
    expected = gd.expected_breeds(FLAT_9H_ALLERGIES, akc)
    for breed in ("German Shepherd Dog", "Labrador Retriever", "Border Collie"):
        assert breed in expected["forbidden_breeds"]
    for breed in ("Cavalier King Charles Spaniel", "Bichon Frise"):
        assert breed in expected["acceptable_breeds"]


def test_rule_boundaries():
    akc = {
        "Exactly 25 kg": {"max_weight_kg": 25.0, "energy": 3, "shedding": 2},
        "Just over 25 kg": {"max_weight_kg": 25.1, "energy": 3, "shedding": 2},
        "Energy 5": {"max_weight_kg": 10.0, "energy": 5, "shedding": 2},
        "Shedding 4": {"max_weight_kg": 10.0, "energy": 3, "shedding": 4},
        "Unknown shedding": {"max_weight_kg": 10.0, "energy": 3, "shedding": None},
    }
    flat_8h_allergies = {**FLAT_9H_ALLERGIES, "hours_alone": 8}
    expected = gd.expected_breeds(flat_8h_allergies, akc)
    assert expected["forbidden_breeds"] == ["Energy 5", "Just over 25 kg", "Shedding 4"]
    assert expected["acceptable_breeds"] == ["Exactly 25 kg"]  # "over 25 kg"; unknown shedding is in neither list

    seven_hours_in_a_house = {**FLAT_9H_ALLERGIES, "home": "house", "hours_alone": 7, "has_allergies": False}
    expected = gd.expected_breeds(seven_hours_in_a_house, akc)
    assert expected["forbidden_breeds"] == [] and len(expected["acceptable_breeds"]) == len(akc)


def test_validator_catches_a_wrong_item(items, akc):
    broken = copy.deepcopy([i for i in items if i["part"] == "D"])
    first = broken[0]["expected"]
    first["acceptable_breeds"].append(first["forbidden_breeds"].pop())
    assert any("does not pass every hard filter" in error for error in gd.validate(broken, akc))


def test_score_top3_follows_the_milestone_metric():
    expected = {"forbidden_breeds": ["Bad"], "acceptable_breeds": ["Good", "Fine"]}
    assert gd.score_top3(["Good", "Fine", "Unknown"], expected)["pass"] is True
    assert gd.score_top3(["Good", "Bad", "Fine"], expected) == {
        "forbidden_in_top3": ["Bad"], "acceptable_in_top3": ["Fine", "Good"], "pass": False}
    assert gd.score_top3(["Unknown"], expected)["pass"] is False     # no acceptable breed
    assert gd.score_top3([], expected)["pass"] is False


def test_recommend_breeds_passes_every_part_d_item(items):
    from mcp_server.tools.traits import recommend_breeds
    for item in (i for i in items if i["part"] == "D"):
        top3 = [candidate.breed for candidate in recommend_breeds(**item["input"]).top3]
        assert gd.score_top3(top3, item["expected"])["pass"], (item["id"], top3)


def test_rules_are_written_independently_of_the_tool():
    source = Path(gd.__file__).read_text(encoding="utf-8")
    assert "tools.traits" not in source and "tools import traits" not in source
