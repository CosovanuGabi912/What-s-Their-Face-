"""Temperature scaling and a confidence threshold for the breed classifier (issue #10).

- Fits one temperature T on the validation split from issue #6 (data/splits/stanford_dogs_val.json),
  so that softmax(logits / T) gives honest confidences. Accuracy does not change.
- Picks the confidence threshold on the same validation split: the lowest threshold for which
  at least --min-precision of the "confident" answers are correct. This keeps as many confident
  answers as possible while limiting wrong confident answers.

Outputs in models/breed_classifier/: temperature.json and threshold.json.

Usage:
    python scripts/calibrate_classifier.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcp_server.tools.breed_id import (  # noqa: E402
    CHECKPOINT_PATH, MODEL_DIR, PREPROCESS, TEMPERATURE_PATH, THRESHOLD_PATH, get_device, load_classifier,
)

SPLIT_FILE = ROOT / "data" / "splits" / "stanford_dogs_val.json"


@torch.no_grad()
def validation_logits(data_dir: Path, batch_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    device = get_device()
    model, _ = load_classifier(device)
    stanford_labels = torch.load(CHECKPOINT_PATH, map_location="cpu")["stanford_labels"]
    label_index = {label: i for i, label in enumerate(stanford_labels)}
    files = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))["val_files"]

    logits, labels = [], []
    for start in range(0, len(files), batch_size):
        batch_files = files[start:start + batch_size]
        images = []
        for name in batch_files:
            with Image.open(data_dir / "train" / name) as image:
                images.append(PREPROCESS(image.convert("RGB")))
        logits.append(model(torch.stack(images).to(device)).float().cpu())
        labels.extend(label_index[name.split("/")[0]] for name in batch_files)
    return torch.cat(logits), torch.tensor(labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Minimise the validation negative log-likelihood over T (optimised as log T so T stays > 0)."""
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_t.exp().item()


def calibration_error(confidence: torch.Tensor, correct: torch.Tensor, bins: int = 15) -> float:
    """Expected calibration error: average gap between confidence and accuracy, over 15 bins."""
    edges = torch.linspace(0, 1, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            error += in_bin.float().mean().item() * abs(confidence[in_bin].mean().item() - correct[in_bin].float().mean().item())
    return error


def choose_threshold(confidence: torch.Tensor, correct: torch.Tensor, min_precision: float) -> dict:
    for step in range(5, 100):
        threshold = step / 100
        confident = confidence >= threshold
        if confident.any() and correct[confident].float().mean().item() >= min_precision:
            return {
                "confidence_threshold": threshold,
                "precision_when_confident": round(correct[confident].float().mean().item(), 4),
                "share_confident": round(confident.float().mean().item(), 4),
            }
    sys.exit(f"No threshold reaches precision {min_precision}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "stanford_dogs")
    parser.add_argument("--min-precision", type=float, default=0.95)
    args = parser.parse_args()

    logits, labels = validation_logits(args.data_dir)
    temperature = fit_temperature(logits, labels)

    before = torch.softmax(logits, dim=1)
    after = torch.softmax(logits / temperature, dim=1)
    correct = before.argmax(dim=1) == labels
    confidence = after.max(dim=1).values
    nll = torch.nn.functional.cross_entropy

    TEMPERATURE_PATH.write_text(json.dumps({
        "temperature": round(temperature, 4),
        "fitted_on": f"validation split ({len(labels)} images, {SPLIT_FILE.name})",
        "nll_before": round(nll(logits, labels).item(), 4),
        "nll_after": round(nll(logits / temperature, labels).item(), 4),
        "ece_before": round(calibration_error(before.max(dim=1).values, correct), 4),
        "ece_after": round(calibration_error(confidence, correct), 4),
        "top1_accuracy": round(correct.float().mean().item(), 4),
    }, indent=1), encoding="utf-8")

    threshold = choose_threshold(confidence, correct, args.min_precision)
    THRESHOLD_PATH.write_text(json.dumps({
        **threshold,
        "rule": f"lowest threshold where at least {args.min_precision:.0%} of confident answers are correct",
        "tuned_on": "validation split",
    }, indent=1), encoding="utf-8")

    print(json.loads(TEMPERATURE_PATH.read_text(encoding="utf-8")))
    print(json.loads(THRESHOLD_PATH.read_text(encoding="utf-8")))
    print(f"Saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
