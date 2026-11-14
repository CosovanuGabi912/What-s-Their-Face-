"""Fine-tune ConvNeXt-Tiny on Stanford Dogs (120 breeds). Issue #6.

- Train on 80% of the official train split. The other 20% is kept as a validation split
  (saved to data/splits/stanford_dogs_val.json) and is reused for temperature scaling (#10).
- The best epoch on validation is kept, then evaluated once on the official test split.

Outputs in --output-dir:
    convnext_tiny_stanford120.pt   {'state_dict', 'class_names', 'num_classes', ...}
    class_names.json               120 canonical display names, in model output order
    metrics.json                   top-1 / top-3 accuracy on the test split + training details

Usage:
    python scripts/train_classifier.py --model convnext_tiny --data-dir data/raw/stanford_dogs \
        --epochs 20 --batch-size 32 --output-dir models/breed_classifier
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NUM_CLASSES = 120
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SPLIT_FILE = ROOT / "data" / "splits" / "stanford_dogs_val.json"
BREEDS_MAP_PATH = ROOT / "data" / "breeds_map.json"

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
# Same preprocessing as identify_breed (#10).
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def get_device() -> torch.device:
    try:
        from utils.device import get_device as shared_get_device  # from the shared skeleton (#3)
        return shared_get_device()
    except ImportError:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


def display_names(stanford_labels: list[str]) -> tuple[list[str], str]:
    """Canonical breed names from breeds_map.json (#3); readable folder names until it exists."""
    if BREEDS_MAP_PATH.exists():
        breeds_map = json.loads(BREEDS_MAP_PATH.read_text(encoding="utf-8"))
        return [breeds_map[label.split("-", 1)[0]]["display"] for label in stanford_labels], "breeds_map"
    names = []
    for label in stanford_labels:
        words = label.split("-", 1)[1].replace("_", " ").split()
        names.append(" ".join(word[0].upper() + word[1:] for word in words))
    return names, "stanford_folder_names"


def stratified_split(dataset: datasets.ImageFolder, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Same validation images on every run: reuse the saved split if it exists."""
    relative = [Path(path).relative_to(dataset.root).as_posix() for path, _ in dataset.samples]
    if SPLIT_FILE.exists():
        val_files = set(json.loads(SPLIT_FILE.read_text(encoding="utf-8"))["val_files"])
        val_idx = [i for i, name in enumerate(relative) if name in val_files]
        if len(val_idx) != len(val_files):
            sys.exit(f"{SPLIT_FILE} does not match the images in {dataset.root}")
    else:
        by_class: dict[int, list[int]] = defaultdict(list)
        for index, (_, target) in enumerate(dataset.samples):
            by_class[target].append(index)
        rng = random.Random(seed)
        val_idx = []
        for indices in by_class.values():
            rng.shuffle(indices)
            val_idx.extend(indices[: round(len(indices) * val_fraction)])
        val_idx.sort()
        SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        SPLIT_FILE.write_text(json.dumps({
            "description": "Validation images held out from the Stanford Dogs train split (paths relative to train/).",
            "seed": seed,
            "val_fraction": val_fraction,
            "val_files": [relative[i] for i in val_idx],
        }, indent=1), encoding="utf-8")
    val_set = set(val_idx)
    train_idx = [i for i in range(len(dataset)) if i not in val_set]
    return train_idx, val_idx


def build_model() -> nn.Module:
    model = models.convnext_tiny(weights="DEFAULT")
    model.classifier[2] = nn.Linear(768, NUM_CLASSES)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, autocast) -> dict[str, float]:
    model.eval()
    top1 = top3 = total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        with autocast():
            logits = model(images)
        best3 = logits.topk(3, dim=1).indices
        top1 += (best3[:, 0] == targets).sum().item()
        top3 += (best3 == targets[:, None]).any(dim=1).sum().item()
        total += targets.numel()
    return {"top1": top1 / total, "top3": top3 / total, "images": total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="convnext_tiny", choices=["convnext_tiny"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "stanford_dogs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "breed_classifier")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16

    def autocast():
        return torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda)

    train_root, test_root = args.data_dir / "train", args.data_dir / "test"
    if not train_root.exists():
        sys.exit(f"{train_root} not found. Run: python scripts/download_stanford_dogs.py")

    train_full = datasets.ImageFolder(train_root, transform=TRAIN_TRANSFORM)
    eval_full = datasets.ImageFolder(train_root, transform=EVAL_TRANSFORM)
    test_set = datasets.ImageFolder(test_root, transform=EVAL_TRANSFORM)
    if train_full.classes != test_set.classes or len(train_full.classes) != NUM_CLASSES:
        sys.exit("Train and test folders must contain the same 120 breed folders")

    train_idx, val_idx = stratified_split(eval_full, args.val_fraction, args.seed)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                       pin_memory=use_cuda, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(Subset(train_full, train_idx), shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(Subset(eval_full, val_idx), shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)
    print(f"Device: {device} | train {len(train_idx)} | val {len(val_idx)} | test {len(test_set)}", flush=True)

    model = build_model().to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = len(train_loader)  # one epoch of linear warm-up, then cosine decay

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and amp_dtype == torch.float16)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    history, best_state, best_val, best_epoch = [], None, -1.0, 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start, running_loss, seen = time.time(), 0.0, 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                loss = loss_fn(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item() * targets.size(0)
            seen += targets.size(0)

        val = evaluate(model, val_loader, device, autocast)
        history.append({"epoch": epoch, "train_loss": round(running_loss / seen, 4),
                        "val_top1": round(val["top1"], 4), "val_top3": round(val["top3"], 4)})
        print(f"Epoch {epoch:2d}/{args.epochs} | loss {running_loss / seen:.4f} | "
              f"val top-1 {val['top1']:.4f} top-3 {val['top3']:.4f} | {time.time() - epoch_start:.0f}s", flush=True)
        if val["top1"] > best_val:
            best_val, best_epoch = val["top1"], epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    val = evaluate(model, val_loader, device, autocast)
    test = evaluate(model, test_loader, device, autocast)
    print(f"Best epoch {best_epoch} | test top-1 {test['top1']:.4f} top-3 {test['top3']:.4f}", flush=True)

    class_names, names_source = display_names(train_full.classes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "class_names": class_names,
        "num_classes": NUM_CLASSES,
        "stanford_labels": train_full.classes,
        "class_names_source": names_source,
        "model": args.model,
        "image_size": 224,
    }, args.output_dir / "convnext_tiny_stanford120.pt")
    (args.output_dir / "class_names.json").write_text(json.dumps(class_names, indent=1), encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "top1_accuracy": round(test["top1"], 4),
        "top3_accuracy": round(test["top3"], 4),
        "evaluated_on": f"Stanford Dogs official test split ({test['images']} images)",
        "val_top1_accuracy": round(val["top1"], 4),
        "val_top3_accuracy": round(val["top3"], 4),
        "best_epoch": best_epoch,
        "class_names_source": names_source,
        "config": {"model": args.model, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                   "weight_decay": args.weight_decay, "label_smoothing": 0.1, "seed": args.seed,
                   "train_images": len(train_idx), "val_images": len(val_idx)},
        "device": torch.cuda.get_device_name(0) if use_cuda else str(device),
        "torch_version": torch.__version__,
        "train_minutes": round((time.time() - started) / 60, 1),
        "history": history,
    }, indent=1), encoding="utf-8")
    print(f"Saved checkpoint, class_names.json and metrics.json to {args.output_dir}")


if __name__ == "__main__":
    main()
