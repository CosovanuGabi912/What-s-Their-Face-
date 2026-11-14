"""identify_breed tool (issue #10): fine-tuned ConvNeXt-Tiny with temperature scaling.

Files read from models/breed_classifier/:
    convnext_tiny_stanford120.pt   weights + class names (issue #6)
    temperature.json               {"temperature": T}, from scripts/calibrate_classifier.py
    threshold.json                 {"confidence_threshold": t}

Calibrated probabilities are softmax(logits / T). The answer is "confident" when the top-1
probability is at least the threshold; below it the agent shows the top 2 or asks for a new photo.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "breed_classifier"
CHECKPOINT_PATH = MODEL_DIR / "convnext_tiny_stanford120.pt"
TEMPERATURE_PATH = MODEL_DIR / "temperature.json"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"

try:
    from config import IMAGE_UPLOAD_DIR  # shared config (issue #3)
except ImportError:
    IMAGE_UPLOAD_DIR = ROOT / "data" / "uploads"

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_device() -> torch.device:
    try:
        from utils.device import get_device as shared_get_device  # shared helper (issue #3)
        return shared_get_device()
    except ImportError:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


def load_classifier(device: torch.device) -> tuple[nn.Module, list[str]]:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(768, checkpoint["num_classes"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint["class_names"]


def read_json_value(path: Path, key: str, default: float) -> float:
    return float(json.loads(path.read_text(encoding="utf-8"))[key]) if path.exists() else default


@lru_cache(maxsize=1)
def _classifier() -> tuple[nn.Module, list[str], torch.device, float, float]:
    """Load the model once per process; the MCP server calls identify_breed many times."""
    device = get_device()
    model, class_names = load_classifier(device)
    temperature = read_json_value(TEMPERATURE_PATH, "temperature", 1.0)
    threshold = read_json_value(THRESHOLD_PATH, "confidence_threshold", 0.5)
    return model, class_names, device, temperature, threshold


def image_path(image_id: str) -> Path:
    """Only plain file names inside the upload folder are allowed (no '../' paths)."""
    if not image_id or Path(image_id).name != image_id:
        raise ValueError(f"Invalid image_id: {image_id!r}")
    path = IMAGE_UPLOAD_DIR / image_id
    if not path.is_file():
        raise FileNotFoundError(f"No uploaded image named {image_id!r}")
    return path


@torch.no_grad()
def identify_breed(image_id: str) -> dict:
    """Identify the dog breed in an uploaded photo.

    Returns the 3 most likely breeds with calibrated confidence, and is_confident=True when
    the top-1 confidence is at least the tuned threshold.
    Example: identify_breed(image_id="abc123.jpg")
    """
    model, class_names, device, temperature, threshold = _classifier()
    with Image.open(image_path(image_id)) as image:
        batch = PREPROCESS(image.convert("RGB")).unsqueeze(0).to(device)
    probabilities = torch.softmax(model(batch).float() / temperature, dim=1)[0]
    values, indices = probabilities.topk(3)
    top3 = [{"breed": class_names[i], "confidence": round(v, 4)} for v, i in zip(values.tolist(), indices.tolist())]
    result = {"top3": top3, "is_confident": top3[0]["confidence"] >= threshold}

    try:
        from shared.schemas import IdentifyBreedOutput  # contract from issue #3
        return IdentifyBreedOutput.model_validate(result).model_dump()
    except ImportError:
        return result
