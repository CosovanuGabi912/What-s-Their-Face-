"""Temperature scaling and confidence thresholds for the breed classifier (issue #10).

Everything runs on the validation split from issue #6 (data/splits/stanford_dogs_val.json):
2,400 train images the model never trained on. Golden Part A is never used here.

1. --fit-temperature (or when temperature.json is missing): fit one temperature T so that
   softmax(logits / T) gives honest confidences (accuracy does not change) and write
   temperature.json. Otherwise the saved T is reused, so --sweep never changes a committed file.
2. --sweep: print, for a grid of candidate thresholds, the numbers the team uses to choose them
   by trial and error (no F1):
     confident share, accuracy of confident answers, share of wrong guesses flagged;
     for "top2" answers, how often the true breed is one of the two shown.
   The "top2" table is printed for both rules: top-1 >= threshold, and top-1 + top-2 >= threshold.
3. --confidence-threshold X --top2-threshold Y --reason "...": write the chosen values to
   threshold.json (read by config.py as CONFIDENCE_HIGH / CONFIDENCE_LOW).

Usage:
    python scripts/calibrate_classifier.py --fit-temperature
    python scripts/calibrate_classifier.py --sweep
    python scripts/calibrate_classifier.py --confidence-threshold 0.82 --top2-threshold 0.5 --reason "..."
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
    read_temperature,
)

SPLIT_FILE = ROOT / "data" / "splits" / "stanford_dogs_val.json"


@torch.no_grad()
def validation_logits(data_dir: Path, batch_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    device = get_device()
    model, _ = load_classifier(device)
    stanford_labels = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)["stanford_labels"]
    label_index = {label: i for i, label in enumerate(stanford_labels)}
    files = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))["val_files"]

    logits, labels = [], []
    for start in range(0, len(files), batch_size):
        batch_files = files[start:start + batch_size]
        batch = []
        for name in batch_files:
            with Image.open(data_dir / "train" / name) as image:
                batch.append(PREPROCESS(image.convert("RGB")))
        logits.append(model(torch.stack(batch).to(device)).float().cpu())
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


def share(mask: torch.Tensor) -> float:
    return mask.float().mean().item() if mask.numel() else float("nan")


def print_sweep(probs: torch.Tensor, labels: torch.Tensor, confidence_threshold: float) -> None:
    top2 = probs.topk(2, dim=1)
    p1, p2 = top2.values[:, 0], top2.values[:, 1]
    correct = top2.indices[:, 0] == labels
    in_top2 = (top2.indices == labels[:, None]).any(dim=1)

    print("\nConfident threshold (top-1 >= t)")
    print("    t | confident | correct when confident | wrong guesses flagged")
    for step in range(10, 20):
        t = step / 20
        confident = p1 >= t
        print(f" {t:.2f} | {share(confident):9.1%} | {share(correct[confident]):22.1%} | {share(~confident[~correct]):21.1%}")

    not_confident = p1 < confidence_threshold
    print(f"\nTop-2 threshold, for answers under the confident threshold {confidence_threshold:.2f} "
          f"({share(not_confident):.1%} of all answers)")
    for rule, score, grid in (("top-1 >= t", p1, range(4, 17)), ("top-1 + top-2 >= t", p1 + p2, range(8, 19))):
        print(f"  Rule: {rule}")
        print("      t | top2 shown | true breed in the 2 shown | retry | true breed was in top 2 on retry")
        for step in grid:
            t = step / 20
            show_top2 = not_confident & (score >= t)
            retry = not_confident & (score < t)
            print(f"   {t:.2f} | {share(show_top2):10.1%} | {share(in_top2[show_top2]):25.1%} | "
                  f"{share(retry):5.1%} | {share(in_top2[retry]):32.1%}")


def save_temperature(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> None:
    before = torch.softmax(logits, dim=1)
    after = torch.softmax(logits / temperature, dim=1)
    correct = before.argmax(dim=1) == labels
    nll = torch.nn.functional.cross_entropy
    TEMPERATURE_PATH.write_text(json.dumps({
        "temperature": round(temperature, 4),
        "fitted_on": f"validation split ({len(labels)} images, {SPLIT_FILE.name})",
        "nll_before": round(nll(logits, labels).item(), 4),
        "nll_after": round(nll(logits / temperature, labels).item(), 4),
        "ece_before": round(calibration_error(before.max(dim=1).values, correct), 4),
        "ece_after": round(calibration_error(after.max(dim=1).values, correct), 4),
        "top1_accuracy": round(correct.float().mean().item(), 4),
    }, indent=1), encoding="utf-8")
    print(json.loads(TEMPERATURE_PATH.read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "stanford_dogs")
    parser.add_argument("--fit-temperature", action="store_true", help="refit T and overwrite temperature.json")
    parser.add_argument("--sweep", action="store_true", help="print the tables used to choose the thresholds")
    parser.add_argument("--confidence-threshold", type=float, help="value to save in threshold.json")
    parser.add_argument("--top2-threshold", type=float, help="value to save in threshold.json")
    parser.add_argument("--reason", help="one line: why these values")
    args = parser.parse_args()

    logits, labels = validation_logits(args.data_dir)
    if args.fit_temperature or not TEMPERATURE_PATH.exists():
        save_temperature(logits, labels, fit_temperature(logits, labels))
    temperature = read_temperature()
    print(f"Temperature T = {temperature} ({TEMPERATURE_PATH.name})")
    after = torch.softmax(logits / temperature, dim=1)

    if args.sweep:
        current = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8")) if THRESHOLD_PATH.exists() else {}
        print_sweep(after, labels, args.confidence_threshold or current.get("confidence_threshold", 0.82))

    if args.confidence_threshold is not None or args.top2_threshold is not None:
        if args.confidence_threshold is None or args.top2_threshold is None or not args.reason:
            sys.exit("To save thresholds pass --confidence-threshold, --top2-threshold and --reason together")
        THRESHOLD_PATH.write_text(json.dumps({
            "confidence_threshold": args.confidence_threshold,
            "top2_threshold": args.top2_threshold,
            "top2_rule": "top-1 >= top2_threshold (shared.schemas.breed_decision)",
            "tuned_on": "validation split",
            "method": "trial and error",
            "reason": args.reason,
        }, indent=1), encoding="utf-8")
        print(f"Saved {THRESHOLD_PATH.relative_to(ROOT)}")
    print(f"Files in {MODEL_DIR}")


if __name__ == "__main__":
    main()
