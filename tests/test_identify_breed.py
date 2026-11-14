"""Tests for the identify_breed tool (issue #10)."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
from mcp_server.tools import breed_id  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"

pytestmark = pytest.mark.skipif(
    not breed_id.CHECKPOINT_PATH.exists(), reason="No checkpoint yet: run scripts/train_classifier.py"
)


@pytest.fixture(autouse=True)
def uploads_in_fixtures(monkeypatch):
    monkeypatch.setattr(breed_id, "IMAGE_UPLOAD_DIR", FIXTURES)


def test_husky_photo_returns_valid_output():
    result = breed_id.identify_breed(image_id="test_husky.jpg")
    class_names = json.loads((breed_id.MODEL_DIR / "class_names.json").read_text(encoding="utf-8"))

    assert set(result) == {"top3", "is_confident"}
    assert isinstance(result["is_confident"], bool)
    assert len(result["top3"]) == 3
    assert all(item["breed"] in class_names for item in result["top3"])
    confidences = [item["confidence"] for item in result["top3"]]
    assert confidences == sorted(confidences, reverse=True)
    assert all(0.0 <= c <= 1.0 for c in confidences)
    assert result["top3"][0]["breed"] == "Siberian Husky"


def test_calibration_files_are_valid_json():
    temperature = json.loads(breed_id.TEMPERATURE_PATH.read_text(encoding="utf-8"))["temperature"]
    threshold = json.loads(breed_id.THRESHOLD_PATH.read_text(encoding="utf-8"))["confidence_threshold"]
    assert temperature > 0
    assert 0 < threshold < 1


@pytest.mark.parametrize("image_id", ["../secret.jpg", "sub/dir.jpg", ""])
def test_rejects_paths_outside_the_upload_folder(image_id):
    with pytest.raises(ValueError):
        breed_id.identify_breed(image_id=image_id)


def test_missing_image_raises_clear_error():
    with pytest.raises(FileNotFoundError):
        breed_id.identify_breed(image_id="does_not_exist.jpg")
