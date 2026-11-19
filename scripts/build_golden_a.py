"""Build golden dataset Part A from Stanford Dogs test split (issue #8).

Selection is scripted and reproducible:
    python scripts/build_golden_a.py --seed 42 --n 100

Requirements:
- Samples >= 50 breeds, <= 2 photos per breed.
- Always includes the 3 wild canids: Dingo, Dhole, African Wild Dog.
- Always includes the confusable sets:
    - Siberian Husky / Eskimo Dog / Alaskan Malamute
    - Poodle (Toy) / Poodle (Miniature) / Poodle (Standard)
- Source is data/raw/stanford_dogs/test/
- Copies selected images to eval/images/ as A_001.jpg ... A_100.jpg
- Records canonical display name from data/breeds_map.json
- Verifies none of the selected files are in data/splits/stanford_dogs_val.json
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from shared.schemas import GoldenItem  # noqa: E402

MANDATORY_BREEDS = [
    # Wild canids
    "African Wild Dog",
    "Dingo",
    "Dhole",
    # Confusable group 1
    "Siberian Husky",
    "Eskimo Dog",
    "Alaskan Malamute",
    # Confusable group 2
    "Poodle (Toy)",
    "Poodle (Miniature)",
    "Poodle (Standard)",
]


def load_val_filenames(val_split_path: Path) -> Set[str]:
    """Load validation image filenames that must never appear in test / golden."""
    if not val_split_path.exists():
        return set()
    data = json.loads(val_split_path.read_text(encoding="utf-8"))
    val_files = data.get("val_files", [])
    return {Path(f).name for f in val_files}


def build_golden_a(
    test_dir: Path,
    breeds_map_path: Path,
    val_split_path: Path,
    images_out_dir: Path,
    seed: int = 42,
    n: int = 100,
) -> List[dict]:
    rng = random.Random(seed)
    val_filenames = load_val_filenames(val_split_path)

    breeds_map = json.loads(breeds_map_path.read_text(encoding="utf-8"))
    # Map stanford_label to canonical display breed
    stanford_to_display: Dict[str, str] = {}
    display_to_label: Dict[str, str] = {}
    for display, meta in breeds_map.items():
        label = meta.get("stanford_label")
        if label:
            stanford_to_display[label] = display
            display_to_label[display] = label

    # Discover available test images per label
    available: Dict[str, List[Path]] = {}
    for label_dir in sorted(test_dir.iterdir()):
        if label_dir.is_dir() and label_dir.name in stanford_to_display:
            photos = sorted(
                p for p in label_dir.glob("*.jpg")
                if p.name not in val_filenames
            )
            if photos:
                available[label_dir.name] = photos

    # Verify mandatory breeds exist
    for breed in MANDATORY_BREEDS:
        label = display_to_label.get(breed)
        if not label or label not in available or not available[label]:
            sys.exit(f"Mandatory breed {breed!r} (label {label}) not found in {test_dir}")

    # Build breed selection
    # We want n items with >= 50 breeds and <= 2 photos per breed.
    # 50 breeds with 2 photos each gives exactly 100 photos.
    num_breeds = max(50, n // 2)
    mandatory_labels = [display_to_label[b] for b in MANDATORY_BREEDS]
    other_labels = sorted(set(available.keys()) - set(mandatory_labels))
    rng.shuffle(other_labels)

    chosen_labels = list(mandatory_labels)
    needed = num_breeds - len(chosen_labels)
    chosen_labels.extend(other_labels[:needed])
    chosen_labels.sort()

    # Collect 2 images per chosen breed (or 1 if n is reached)
    selected_photos: List[tuple[str, Path]] = []  # (display_name, photo_path)
    for label in chosen_labels:
        photos = list(available[label])
        rng.shuffle(photos)
        display = stanford_to_display[label]
        for p in photos[:2]:
            selected_photos.append((display, p))
            if len(selected_photos) == n:
                break
        if len(selected_photos) == n:
            break

    if len(selected_photos) < n:
        sys.exit(f"Could not sample {n} photos (got {len(selected_photos)})")

    # Copy to eval/images/ and build golden items
    images_out_dir.mkdir(parents=True, exist_ok=True)
    golden_items: List[dict] = []

    for idx, (display, photo_path) in enumerate(selected_photos, start=1):
        item_id = f"A_{idx:03d}"
        dest_filename = f"{item_id}.jpg"
        dest_path = images_out_dir / dest_filename
        shutil.copy2(photo_path, dest_path)

        label_name = photo_path.parent.name
        source_id = f"stanford_dogs/test/{label_name}/{photo_path.name}"

        item = {
            "id": item_id,
            "part": "A",
            "input": {"image_id": dest_filename},
            "expected": {"breed": display},
            "source_ids": [source_id],
        }
        # Validate against GoldenItem schema
        GoldenItem.model_validate(item)
        golden_items.append(item)

    return golden_items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--test-dir", type=Path, default=config.DATA_DIR / "raw" / "stanford_dogs" / "test")
    parser.add_argument("--breeds-map", type=Path, default=config.BREEDS_MAP_PATH)
    parser.add_argument("--val-split", type=Path, default=config.DATA_DIR / "splits" / "stanford_dogs_val.json")
    parser.add_argument("--images-out", type=Path, default=config.EVAL_IMAGES_DIR)
    parser.add_argument("--output", type=Path, default=config.GOLDEN_PATH)
    args = parser.parse_args()

    items = build_golden_a(
        test_dir=args.test_dir,
        breeds_map_path=args.breeds_map,
        val_split_path=args.val_split,
        images_out_dir=args.images_out,
        seed=args.seed,
        n=args.n,
    )
    print(f"Built {len(items)} Part A items.")


if __name__ == "__main__":
    main()
