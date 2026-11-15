"""Contract tests (issue #3): config, breeds_map, schemas and the 6 tools, in-process.

Tools run as stubs (WTF_STUB_TOOLS=all) so these tests need no model weights or Chroma; each
part tests its real implementation in its own test file. get_breed_traits has no stub, so the
real AKC lookup is tested here. The stdio handshake with a real MCP client is in tests/test_smoke.py.
"""

from __future__ import annotations

import json
import re

import pytest
from pydantic import ValidationError

import config
from mcp_server import server
from scripts.validate_breeds_map import validate
from shared import breeds, images
from shared.schemas import (CHUNK_ID_RE, TOOL_SCHEMAS, BreedCandidate, DescribePetOutput,
                            GetBreedTraitsOutput, GoldenItem, IdentifyBreedOutput, KnowledgeChunk,
                            ToolCallLog, breed_decision)

WILD = ["Dingo", "Dhole", "African Wild Dog"]


@pytest.fixture(autouse=True)
def stub_tools(monkeypatch):
    monkeypatch.setattr(config, "STUB_TOOLS", {"all"})

SAMPLE_CALLS = {
    "identify_breed": {"image_id": "abc123.jpg"},
    "search_knowledge": {"query": "how much exercise", "breed": "Siberian Husky", "section": "Exercise"},
    "get_breed_traits": {"breed": "Border Collie"},
    "recommend_breeds": {"home": "flat", "hours_alone": 8, "daily_walks_min": 30, "kids_at_home": False,
                         "has_allergies": True, "experience": "none"},
    "describe_pet": {"image_id": "abc123.jpg"},
    "suggest_names": {"species": "dog", "colour": "golden", "description": "A fluffy golden puppy",
                      "exclude_names": ["Max"]},
}


# --- config / device --------------------------------------------------------------------
def test_config_paths_are_absolute_and_inside_the_repo():
    for path in (config.CHROMA_KNOWLEDGE_DIR, config.CHROMA_LISTINGS_DIR, config.AKC_TABLE_PATH,
                 config.BREEDS_MAP_PATH, config.MODELS_DIR, config.EVAL_DIR):
        assert path.is_absolute()
        assert path.is_relative_to(config.ROOT)
    assert config.AKC_TABLE_PATH.is_file() and config.BREEDS_MAP_PATH.is_file()


def test_config_thresholds_and_limits():
    assert 0 < config.CONFIDENCE_LOW < config.CONFIDENCE_HIGH < 1
    assert config.MAX_TOOLS_PER_TURN == 5
    assert config.MAX_SEARCHES_PER_QUESTION == 2


def test_get_device():
    torch = pytest.importorskip("torch")
    from utils.device import get_device
    device = get_device()
    assert isinstance(device, torch.device) and device.type in {"cuda", "mps", "cpu"}


# --- breeds_map ----------------------------------------------------------------------------
def test_breeds_map_passes_the_validator():
    import csv
    with config.AKC_TABLE_PATH.open(encoding="utf-8", newline="") as file:
        akc = {row[0] for row in list(csv.reader(file))[1:] if row}
    assert validate(breeds.load_breeds_map(), akc, None) == []


@pytest.mark.parametrize("name", WILD)
def test_wild_canids_have_no_akc_name(name):
    entry = breeds.load_breeds_map()[name]
    assert entry["kind"] == "wild_canid" and entry["akc_name"] is None


@pytest.mark.parametrize("alias, display", [
    ("German Shepherd", "German Shepherd Dog"),          # design-doc spelling
    ("n02106662-German_shepherd", "German Shepherd Dog"),  # Stanford label
    ("n02115641", "Dingo"),                                # WordNet ID (train_classifier.py, #6)
    ("Japanese Spaniel", "Japanese Chin"),                 # readable Stanford class name (#6)
    ("German Short-haired Pointer", "German Shorthaired Pointer"),
    ("Blenheim Spaniel", "Cavalier King Charles Spaniel"),
    ("Mexican Hairless", "Xoloitzcuintli"),
    ("Toy Poodle", "Poodle (Toy)"),
    ("shih-tzu", "Shih Tzu"),
    ("lowchen", "Löwchen"),
    ("African Hunting Dog", "African Wild Dog"),
])
def test_to_display_resolves_every_known_spelling(alias, display):
    assert breeds.to_display(alias) == display


def test_unknown_breed_resolves_to_none():
    assert breeds.to_display("Dragon Hound") is None
    assert len(breeds.closest_breeds("Golden Retreiver")) == 3


# --- the 6 tools (stub implementations through the server's contract check) -------------------
@pytest.mark.parametrize("tool", list(SAMPLE_CALLS))
def test_every_tool_returns_schema_valid_output(tool):
    _, output_model = TOOL_SCHEMAS[tool]
    result = getattr(server, tool)(**SAMPLE_CALLS[tool])
    assert isinstance(result, output_model)
    assert output_model.model_validate(json.loads(result.model_dump_json())) == result


def test_every_sample_call_matches_its_input_schema():
    for tool, args in SAMPLE_CALLS.items():
        TOOL_SCHEMAS[tool][0].model_validate(args)


def test_stub_selection(monkeypatch):
    from mcp_server.tools import breed_id, stubs, traits
    monkeypatch.setattr(config, "STUB_TOOLS", set())
    assert server._impl(breed_id, "identify_breed") is breed_id.identify_breed
    monkeypatch.setattr(config, "STUB_TOOLS", {"identify_breed"})
    assert server._impl(breed_id, "identify_breed") is stubs.identify_breed
    monkeypatch.setattr(config, "STUB_TOOLS", {"all"})
    assert server._impl(traits, "get_breed_traits") is traits.get_breed_traits  # no stub exists


@pytest.mark.parametrize("image_id, decision, kind", [
    ("abc123.jpg", "confident", "breed"),
    ("unsure_1.jpg", "top2", "breed"),
    ("blurry.png", "retry", "breed"),
    ("dingo.jpg", "confident", "wild_canid"),
])
def test_identify_breed_stub_scenarios(image_id, decision, kind):
    result = server.identify_breed(image_id)
    assert (result.decision, result.kind) == (decision, kind)
    assert result.is_confident == (decision == "confident")


@pytest.mark.parametrize("name", WILD + ["Eskimo Dog", "dingo", "n02115913-dhole"])
def test_get_breed_traits_handles_breeds_without_akc_data(name):
    result = server.get_breed_traits(name)
    assert result.status == "no_akc_data" and result.traits is None and result.message


def test_get_breed_traits_dingo():
    result = server.get_breed_traits("Dingo")
    assert result.status == "no_akc_data"
    assert "Dingo" in result.message


def test_get_breed_traits_unknown_breed_suggests_three():
    result = server.get_breed_traits("Golden Retreiver")
    assert result.status == "not_found"
    assert result.closest_matches[0] == "Golden Retriever" and len(result.closest_matches) == 3


def test_get_breed_traits_reads_every_akc_breed():
    for name, entry in breeds.load_breeds_map().items():
        if entry["akc_name"]:
            result = server.get_breed_traits(name)
            assert result.status == "found" and result.traits.breed == name


def test_get_breed_traits_keeps_missing_scores_as_none():
    traits = server.get_breed_traits("Bedlington Terrier").traits
    assert traits.shedding is None


def test_search_knowledge_resolves_breed_and_section_and_returns_valid_ids():
    result = server.search_knowledge("exercise", breed="German Shepherd", section="exercise")
    assert [c.chunk_id for c in result.chunks] == ["kb:german-shepherd-dog:exercise:0"]
    assert result.chunks[0].breed == "German Shepherd Dog" and result.chunks[0].section == "Exercise"


@pytest.mark.parametrize("call", [
    lambda: server.identify_breed("../secret.jpg"),
    lambda: server.identify_breed("sub/dir.jpg"),
    lambda: server.describe_pet(""),
    lambda: server.search_knowledge("x", breed="Dragon Hound"),     # Chroma filter must never fail silently
    lambda: server.search_knowledge("x", section="Diet"),
    lambda: server.recommend_breeds("flat", 30, 30, False, False, "none"),
    lambda: server.suggest_names("hamster", None, "x"),
])
def test_invalid_inputs_raise(call):
    with pytest.raises((ValueError, ValidationError)):
        call()


def test_suggest_names_respects_exclude_names():
    names = [n.name for n in server.suggest_names("dog", None, "x", exclude_names=["max", "Bella"]).names]
    assert "Max" not in names and "Bella" not in names and names


# --- breeds://list -------------------------------------------------------------------------
def test_breeds_list_resource():
    names = json.loads(server.breeds_list())
    assert names == breeds.display_names()
    assert {"Dingo", "German Shepherd Dog"} <= set(names)
    assert len(names) == len(breeds.load_breeds_map())


# --- schema strictness ---------------------------------------------------------------------
def _identify(**overrides):
    data = {"top3": [{"breed": "German Shepherd Dog", "confidence": 0.91}],
            "decision": "confident", "is_confident": True, "kind": "breed"}
    return IdentifyBreedOutput.model_validate({**data, **overrides})


@pytest.mark.parametrize("overrides", [
    {"top3": [{"breed": "German shepherd", "confidence": 0.91}]},   # not a display name
    {"top3": [{"breed": "German Shepherd Dog", "confidence": 1.2}]},
    {"top3": []},
    {"is_confident": False},
    {"kind": "wild_canid"},
    {"decision": "uncertain"},
    {"extra_field": 1},
])
def test_identify_breed_output_rejects_contract_violations(overrides):
    with pytest.raises(ValidationError):
        _identify(**overrides)


def test_breed_decision_uses_config_thresholds():
    assert breed_decision(config.CONFIDENCE_HIGH) == "confident"
    assert breed_decision(config.CONFIDENCE_LOW) == "top2"
    assert breed_decision(config.CONFIDENCE_LOW - 0.01) == "retry"


def test_other_schema_rules():
    with pytest.raises(ValidationError):
        KnowledgeChunk(chunk_id="kb:Siberian Husky:exercise:0", breed="Siberian Husky", section="Exercise", text="x")
    with pytest.raises(ValidationError):
        GetBreedTraitsOutput(status="found")
    with pytest.raises(ValidationError):
        DescribePetOutput(species="dog", main_colour=None, size="large", description="word " * 21)
    with pytest.raises(ValidationError):
        BreedCandidate(breed="Not A Breed", confidence=0.5)
    assert re.findall(CHUNK_ID_RE, "Needs 2 h a day [kb:siberian-husky:exercise:0].") == ["kb:siberian-husky:exercise:0"]


def test_golden_item_and_tool_call_log():
    GoldenItem(id="A_001", part="A", input={"image_id": "A_001.jpg"},
               expected={"breed": "Siberian Husky"}, source_ids=[])
    ToolCallLog(turn=1, tool="search_knowledge", arguments={"query": "x"},
                result_ids=["kb:siberian-husky:exercise:0"], duration_ms=42)
    with pytest.raises(ValidationError):
        GoldenItem(id="X", part="G", input={}, expected={}, source_ids=[])


# --- image_id convention ---------------------------------------------------------------------
def test_image_upload_round_trip(tmp_path):
    image_id = images.save_upload(b"\xff\xd8fake", "Dog Photo.PNG", upload_dir=tmp_path)
    assert image_id.endswith(".png") and "/" not in image_id
    assert images.image_path(image_id, upload_dir=tmp_path) == tmp_path / image_id
    with pytest.raises(FileNotFoundError):
        images.image_path("missing.jpg", upload_dir=tmp_path)
    for bad in ("../x.jpg", "a/b.jpg", "", "x.exe"):
        with pytest.raises(ValueError):
            images.image_path(bad, upload_dir=tmp_path)
