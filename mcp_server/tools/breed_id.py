"""identify_breed (issue #10) and describe_pet (Week 9, still the stub).

identify_breed uses the fine-tuned ConvNeXt-Tiny from issue #6 with temperature scaling.
Files read from models/breed_classifier/:
    convnext_tiny_stanford120.pt   weights + class names (issue #6, not in git: scripts/download_classifier.py)
    temperature.json               {"temperature": T}, from scripts/calibrate_classifier.py
    threshold.json                 confidence thresholds, read by config.py (CONFIDENCE_HIGH / LOW)

Calibrated probabilities are softmax(logits / T). shared.schemas.breed_decision() turns them into
"confident" (name the breed), "top2" (show the two most likely) or "retry" (ask for a new photo).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms

import config
from mcp_server.tools.stubs import describe_pet  # noqa: F401  (stub until Week 9)
from shared import images
from shared.breeds import load_breeds_map, to_display
from shared.schemas import IdentifyBreedOutput, breed_decision
from utils.device import get_device

MODEL_DIR = config.BREED_CLASSIFIER_DIR
CHECKPOINT_PATH = MODEL_DIR / "convnext_tiny_stanford120.pt"
TEMPERATURE_PATH = MODEL_DIR / "temperature.json"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"
IMAGE_UPLOAD_DIR = config.IMAGE_UPLOAD_DIR  # module-level so tests can point it elsewhere

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_classifier(device: torch.device) -> tuple[nn.Module, list[str]]:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"{CHECKPOINT_PATH} is missing: run python scripts/download_classifier.py")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(768, checkpoint["num_classes"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint["class_names"]


def read_temperature() -> float:
    if not TEMPERATURE_PATH.exists():
        return 1.0
    return float(json.loads(TEMPERATURE_PATH.read_text(encoding="utf-8"))["temperature"])


@lru_cache(maxsize=1)
def _classifier() -> tuple[nn.Module, list[str], torch.device, float]:
    """Load the model once per process; the MCP server calls identify_breed many times."""
    device = get_device()
    model, class_names = load_classifier(device)
    return model, class_names, device, read_temperature()


def image_path(image_id: str) -> Path:
    """Same image_id rules as the rest of the project (shared/images.py)."""
    return images.image_path(image_id, IMAGE_UPLOAD_DIR)


@torch.no_grad()
def identify_breed(image_id: str) -> IdentifyBreedOutput:
    """Identify the dog breed in an uploaded photo.

    Returns the 3 most likely breeds with calibrated confidence, the decision
    (confident / top2 / retry) and whether the top breed is a real breed or a wild canid.
    Example: identify_breed(image_id="abc123.jpg")
    """
    path = image_path(image_id)  # reject a bad image_id before the slow model load
    model, class_names, device, temperature = _classifier()
    with Image.open(path) as image:
        # Phone photos store their rotation as an EXIF tag; without this they reach the model sideways
        upright = ImageOps.exif_transpose(image)
        batch = PREPROCESS(upright.convert("RGB")).unsqueeze(0).to(device)
    probabilities = torch.softmax(model(batch).float() / temperature, dim=1)[0]
    values, indices = probabilities.topk(3)
    top3 = [{"breed": to_display(class_names[i]), "confidence": round(v, 4)}
            for v, i in zip(values.tolist(), indices.tolist())]
    decision = breed_decision(top3[0]["confidence"])
    return IdentifyBreedOutput.model_validate({
        "top3": top3,
        "decision": decision,
        "is_confident": decision == "confident",
        "kind": load_breeds_map()[top3[0]["breed"]]["kind"],
    })
