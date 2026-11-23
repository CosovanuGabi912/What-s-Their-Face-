"""identify_breed (issue #10) and describe_pet (issue #15).

identify_breed uses the fine-tuned ConvNeXt-Tiny from issue #6 with temperature scaling.
Files read from models/breed_classifier/:
    convnext_tiny_stanford120.pt   weights + class names (issue #6, not in git: scripts/download_classifier.py)
    temperature.json               {"temperature": T}, from scripts/calibrate_classifier.py
    threshold.json                 confidence thresholds, read by config.py (CONFIDENCE_HIGH / LOW)

Calibrated probabilities are softmax(logits / T). shared.schemas.breed_decision() turns them into
"confident" (name the breed), "top2" (show the two most likely) or "retry" (ask for a new photo).

describe_pet sends the photo to Qwen2.5-VL-3B in LM Studio (docs/setup.md) and expects one JSON object.
If the model is unavailable or its reply is unusable it returns FALLBACK_DESCRIPTION and never raises;
a bad image_id or a missing file still raises, as in identify_breed.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from functools import lru_cache
from pathlib import Path

import torch
from openai import OpenAI, OpenAIError
from PIL import Image, ImageOps
from pydantic import ValidationError
from torch import nn
from torchvision import models, transforms

import config
from shared import images
from shared.breeds import load_breeds_map, to_display
from shared.schemas import DescribePetOutput, IdentifyBreedOutput, breed_decision
from utils.device import get_device

logger = logging.getLogger(__name__)  # stderr only: stdout carries the MCP protocol

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


# --- describe_pet (issue #15) --------------------------------------------------------------
_COLOUR_CHOICES = ", ".join(f'"{colour}"' for colour in config.PET_COLOURS)
VISION_PROMPT = (
    "Describe this pet in JSON with exactly these fields:\n"
    f'{{"species": "dog" or "cat", "main_colour": one of {_COLOUR_CHOICES}, '
    '"size": "small" or "medium" or "large", "description": "one sentence max 20 words"}\n'
    "Reply with only the JSON object."
)
FALLBACK_DESCRIPTION = DescribePetOutput(species="dog", main_colour=None, size="medium",
                                         description="Unable to describe pet from photo.")
VISION_MAX_SIDE = 768             # pixels; enough for colour and size, and much faster on a laptop
VISION_MAX_TOKENS = 120
VISION_TIMEOUT_SECONDS = 120      # LM Studio may have to load the model on the first call
MAX_DESCRIPTION_WORDS = 20
_COLOUR_SPELLINGS = {"grey": "gray", "gold": "golden"}
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_SENTENCE_END = re.compile(r"(?<=[.!?])(?<!St\.)\s")   # "St. Bernard" is not a sentence end


@lru_cache(maxsize=1)
def _vision_client() -> OpenAI:
    return OpenAI(base_url=config.LM_STUDIO_BASE_URL, api_key=config.LM_STUDIO_API_KEY,
                  timeout=VISION_TIMEOUT_SECONDS, max_retries=0)


def _photo_as_base64(path: Path) -> str:
    """Upright RGB JPEG, longest side at most VISION_MAX_SIDE, as base64."""
    with Image.open(path) as image:
        photo = ImageOps.exif_transpose(image).convert("RGB")
    photo.thumbnail((VISION_MAX_SIDE, VISION_MAX_SIDE))
    buffer = io.BytesIO()
    photo.save(buffer, "JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _one_short_sentence(text: object) -> str:
    words = " ".join(str(text or "").split())
    return " ".join(_SENTENCE_END.split(words, maxsplit=1)[0].split()[:MAX_DESCRIPTION_WORDS])


def parse_description(reply: str | None) -> DescribePetOutput | None:
    """The VLM's reply as DescribePetOutput, or None when it is not usable.

    Tolerates code fences and text around the JSON. A colour outside config.PET_COLOURS becomes None,
    so the agent asks the user; a wrong species or size makes the whole reply unusable.
    """
    match = _JSON_OBJECT.search(_THINK_BLOCK.sub("", reply or ""))
    if match is None:
        return None
    try:
        data = json.loads(match.group())
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    colour = str(data.get("main_colour") or data.get("main_color") or "").strip().casefold()
    colour = _COLOUR_SPELLINGS.get(colour, colour)
    try:
        return DescribePetOutput(
            species=str(data.get("species", "")).strip().casefold(),
            main_colour=colour if colour in config.PET_COLOURS else None,
            size=str(data.get("size", "")).strip().casefold(),
            description=_one_short_sentence(data.get("description")),
        )
    except ValidationError:
        return None


def describe_pet(image_id: str) -> DescribePetOutput:
    """Describe the pet in an uploaded photo: species, main colour, size and one short sentence.

    Example: describe_pet(image_id="abc123.jpg")
    """
    path = image_path(image_id)
    try:
        photo = _photo_as_base64(path)
        response = _vision_client().chat.completions.create(
            model=config.VISION_MODEL_ID,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{photo}"}},
                {"type": "text", "text": VISION_PROMPT},
            ]}],
            temperature=0,
            max_tokens=VISION_MAX_TOKENS,
        )
        reply = response.choices[0].message.content
    except (OpenAIError, OSError, ValueError, IndexError, AttributeError) as error:
        logger.warning("describe_pet: vision model failed for %s: %s", image_id, error)
        return FALLBACK_DESCRIPTION
    described = parse_description(reply)
    if described is None:
        logger.warning("describe_pet: unusable vision reply for %s: %r", image_id, (reply or "")[:200])
        return FALLBACK_DESCRIPTION
    return described
