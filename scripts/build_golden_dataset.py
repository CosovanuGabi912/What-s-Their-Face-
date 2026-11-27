"""Assemble the full golden dataset (Parts A, B, C, E, F) into eval/golden.jsonl (issues #8, #17)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from scripts.build_golden_a import build_golden_a
from scripts.generate_part_b import generate_items as generate_part_b_items
from scripts.build_golden_b_c import PART_C_ITEMS
from shared.schemas import GoldenItem


def build_all_golden() -> Path:
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    config.EVAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("Building Part A items (100 photos)...")
    part_a = build_golden_a(
        test_dir=config.DATA_DIR / "raw" / "stanford_dogs" / "test",
        breeds_map_path=config.BREEDS_MAP_PATH,
        val_split_path=config.DATA_DIR / "splits" / "stanford_dogs_val.json",
        images_out_dir=config.EVAL_IMAGES_DIR,
        seed=42,
        n=100,
    )

    print("Building Part B items (60 breed Q&A items)...")
    part_b = generate_part_b_items()

    print("Building Part C items (15 vet & scope items)...")
    part_c = list(PART_C_ITEMS)

    try:
        from scripts.build_golden_e_f import build_part_e, build_part_f
        print("Building Part E items (50 name items)...")
        part_e, holdout_ids = build_part_e()
        holdout_file = config.EVAL_DIR / "holdout_listing_ids.txt"
        with holdout_file.open("w", encoding="utf-8") as f:
            for hid in holdout_ids:
                f.write(f"{hid}\n")
        print("Building Part F items (10 journey items)...")
        part_f = build_part_f()
    except Exception as exc:
        print(f"Warning: could not build Part E/F: {exc}")
        part_e, part_f = [], []

    all_items = part_a + part_b + part_c + part_e + part_f
    print(f"Total items: {len(all_items)} (A: {len(part_a)}, B: {len(part_b)}, C: {len(part_c)}, E: {len(part_e)}, F: {len(part_f)})")

    # Validate all items
    for it in all_items:
        GoldenItem.model_validate(it)

    golden_path = config.GOLDEN_PATH
    with golden_path.open("w", encoding="utf-8") as f:
        for it in all_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    sha256 = hashlib.sha256(golden_path.read_bytes()).hexdigest()
    print(f"Written {len(all_items)} lines to {golden_path}")
    print(f"SHA-256: {sha256}")
    return golden_path


if __name__ == "__main__":
    build_all_golden()
