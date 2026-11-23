"""Tests for the identify_breed tool (issue #10)."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from mcp_server.tools import breed_id  # noqa: E402
from shared.schemas import IdentifyBreedOutput  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
EXIF_ORIENTATION = 0x0112
ROTATE_90_CLOCKWISE_TO_VIEW = 6  # what a phone held upright writes

# Only these tests need the model; the image_id and calibration-file tests run everywhere
needs_model = pytest.mark.skipif(
    not breed_id.CHECKPOINT_PATH.exists(), reason="No checkpoint yet: run scripts/download_classifier.py"
)


@pytest.fixture(autouse=True)
def uploads_in_fixtures(monkeypatch):
    monkeypatch.setattr(breed_id, "IMAGE_UPLOAD_DIR", FIXTURES)


@needs_model
def test_husky_photo_returns_valid_output():
    result = breed_id.identify_breed(image_id="test_husky.jpg")
    class_names = json.loads((breed_id.MODEL_DIR / "class_names.json").read_text(encoding="utf-8"))

    assert isinstance(result, IdentifyBreedOutput)
    assert isinstance(result.is_confident, bool)
    assert len(result.top3) == 3
    assert all(item.breed in class_names for item in result.top3)
    confidences = [item.confidence for item in result.top3]
    assert confidences == sorted(confidences, reverse=True)
    assert all(0.0 <= c <= 1.0 for c in confidences)
    assert result.top3[0].breed == "Siberian Husky"
    assert result.decision == "confident" and result.kind == "breed"


@needs_model
def test_phone_photo_is_turned_upright_by_its_exif_tag(tmp_path, monkeypatch):
    """A phone stores the photo sideways plus an EXIF tag saying how to turn it."""
    with Image.open(FIXTURES / "test_husky.jpg") as image:
        sideways = image.rotate(90, expand=True)  # pixels as the sensor wrote them
    exif = Image.Exif()
    exif[EXIF_ORIENTATION] = ROTATE_90_CLOCKWISE_TO_VIEW
    sideways.save(tmp_path / "phone.jpg", exif=exif, quality=95)

    upright = breed_id.identify_breed(image_id="test_husky.jpg")
    monkeypatch.setattr(breed_id, "IMAGE_UPLOAD_DIR", tmp_path)
    phone = breed_id.identify_breed(image_id="phone.jpg")

    assert phone.top3[0].breed == "Siberian Husky" and phone.decision == "confident"
    assert abs(phone.top3[0].confidence - upright.top3[0].confidence) < 0.05


def test_calibration_files_are_valid_json():
    temperature = json.loads(breed_id.TEMPERATURE_PATH.read_text(encoding="utf-8"))["temperature"]
    thresholds = json.loads(breed_id.THRESHOLD_PATH.read_text(encoding="utf-8"))
    assert temperature > 0
    assert 0 < thresholds["top2_threshold"] < thresholds["confidence_threshold"] < 1
    assert thresholds["tuned_on"] == "validation split"
    assert thresholds["method"] == "trial and error"


@pytest.mark.parametrize("image_id", ["../secret.jpg", "sub/dir.jpg", ""])
def test_rejects_paths_outside_the_upload_folder(image_id, monkeypatch):
    monkeypatch.setattr(breed_id, "_classifier", lambda: pytest.fail("model loaded before the image_id check"))
    with pytest.raises(ValueError):
        breed_id.identify_breed(image_id=image_id)


def test_missing_image_raises_clear_error(monkeypatch):
    monkeypatch.setattr(breed_id, "_classifier", lambda: pytest.fail("model loaded before the image_id check"))
    with pytest.raises(FileNotFoundError):
        breed_id.identify_breed(image_id="does_not_exist.jpg")
