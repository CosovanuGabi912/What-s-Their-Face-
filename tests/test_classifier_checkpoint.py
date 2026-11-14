"""Checks the breed classifier checkpoint produced by scripts/train_classifier.py (issue #6)."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torchvision import models  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "breed_classifier"
CHECKPOINT = MODEL_DIR / "convnext_tiny_stanford120.pt"
BREEDS_MAP = ROOT / "data" / "breeds_map.json"

pytestmark = pytest.mark.skipif(
    not CHECKPOINT.exists(), reason="No checkpoint yet: run scripts/train_classifier.py"
)


@pytest.fixture(scope="module")
def checkpoint():
    return torch.load(CHECKPOINT, map_location="cpu")


def test_checkpoint_has_contract_keys(checkpoint):
    assert {"state_dict", "class_names", "num_classes"} <= checkpoint.keys()
    assert checkpoint["num_classes"] == 120


def test_class_names_are_120_unique_names(checkpoint):
    names = checkpoint["class_names"]
    assert len(names) == 120
    assert len(set(names)) == 120
    assert all(isinstance(name, str) and name for name in names)


def test_class_names_json_matches_checkpoint(checkpoint):
    names = json.loads((MODEL_DIR / "class_names.json").read_text(encoding="utf-8"))
    assert names == checkpoint["class_names"]


def test_state_dict_loads_into_convnext_tiny(checkpoint):
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(768, 120)
    model.load_state_dict(checkpoint["state_dict"])


def test_metrics_meet_week7_targets():
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["top1_accuracy"] >= 0.70
    assert metrics["top3_accuracy"] >= 0.85


def test_class_names_are_canonical_breeds_map_names(checkpoint):
    if not BREEDS_MAP.exists():
        pytest.skip("data/breeds_map.json is not in the repository yet (issue #3)")
    breeds_map = json.loads(BREEDS_MAP.read_text(encoding="utf-8"))
    display_names = {entry["display"] for entry in breeds_map.values()}
    assert set(checkpoint["class_names"]) <= display_names
