"""Check data/breeds_map.json against the rules in issue #3. Exits 1 and lists every problem.

Rules:
- keys equal `display`; display names are unique (also after normalising case/punctuation)
- `kind` is "breed" or "wild_canid"
- exactly 120 entries have a stanford_label, all different; if data/splits/stanford_dogs_val.json
  (#6) is present they must be exactly its 120 folders
- Dingo, Dhole and African Wild Dog are wild_canid with akc_name null
- akc_name may be null only for wild_canid entries and for Eskimo Dog
- display == akc_name whenever akc_name is set
- every akc_name is in data/akc_breeds.csv, and every AKC breed has an entry

Usage:
    python scripts/validate_breeds_map.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import AKC_TABLE_PATH, BREEDS_MAP_PATH  # noqa: E402
from shared.breeds import normalise  # noqa: E402

STANFORD_SPLIT_FILE = ROOT / "data" / "splits" / "stanford_dogs_val.json"
FIELDS = {"display", "kind", "stanford_label", "akc_name", "petfinder_name", "thedogapi_name"}
WILD_CANIDS = {"Dingo": "dingo", "Dhole": "dhole", "African Wild Dog": "African_hunting_dog"}
NULL_AKC_ALLOWED_BREEDS = {"Eskimo Dog"}
NUM_STANFORD = 120


def validate(breeds_map: dict[str, dict], akc_names: set[str], stanford_folders: set[str] | None) -> list[str]:
    errors: list[str] = []
    seen_norm: dict[str, str] = {}
    labels: list[str] = []

    for key, entry in breeds_map.items():
        missing = FIELDS - entry.keys()
        if missing:
            errors.append(f"{key!r}: missing fields {sorted(missing)}")
            continue
        display, kind, akc = entry["display"], entry["kind"], entry["akc_name"]
        if key != display:
            errors.append(f"{key!r}: key differs from display {display!r}")
        if normalise(display) in seen_norm:
            errors.append(f"{display!r}: duplicate of {seen_norm[normalise(display)]!r}")
        seen_norm[normalise(display)] = display
        if kind not in ("breed", "wild_canid"):
            errors.append(f"{display!r}: kind must be 'breed' or 'wild_canid', not {kind!r}")
        if akc is None:
            if kind != "wild_canid" and display not in NULL_AKC_ALLOWED_BREEDS:
                errors.append(f"{display!r}: akc_name is null but it is not a wild canid or Eskimo Dog")
        else:
            if display != akc:
                errors.append(f"{display!r}: display must equal akc_name {akc!r}")
            if akc not in akc_names:
                errors.append(f"{display!r}: akc_name {akc!r} is not in {AKC_TABLE_PATH.name}")
        if kind == "wild_canid" and akc is not None:
            errors.append(f"{display!r}: wild canids have no AKC entry")
        if entry["stanford_label"]:
            labels.append(entry["stanford_label"])

    if len(labels) != NUM_STANFORD:
        errors.append(f"{len(labels)} entries have a stanford_label, expected {NUM_STANFORD}")
    if len(set(labels)) != len(labels):
        errors.append("a stanford_label is used by more than one entry")
    if stanford_folders is not None and set(labels) != stanford_folders:
        errors.append(f"stanford_labels differ from {STANFORD_SPLIT_FILE.name}: "
                      f"missing {sorted(stanford_folders - set(labels))}, extra {sorted(set(labels) - stanford_folders)}")

    for display, folder in WILD_CANIDS.items():
        entry = breeds_map.get(display)
        if (not entry or entry["kind"] != "wild_canid" or entry["akc_name"] is not None
                or not (entry["stanford_label"] or "").endswith("-" + folder)):
            errors.append(f"{display!r} must be a wild_canid with akc_name null and Stanford folder {folder!r}")

    mapped_akc = {e["akc_name"] for e in breeds_map.values() if e.get("akc_name")}
    for name in sorted(akc_names - mapped_akc):
        errors.append(f"AKC breed {name!r} has no entry")
    return errors


def main() -> int:
    breeds_map = json.loads(BREEDS_MAP_PATH.read_text(encoding="utf-8"))
    with AKC_TABLE_PATH.open(encoding="utf-8", newline="") as file:
        akc_names = {row[0] for row in list(csv.reader(file))[1:] if row and row[0]}
    stanford_folders = None
    if STANFORD_SPLIT_FILE.exists():
        files = json.loads(STANFORD_SPLIT_FILE.read_text(encoding="utf-8"))["val_files"]
        stanford_folders = {Path(p).parts[0] for p in files}

    errors = validate(breeds_map, akc_names, stanford_folders)
    for error in errors:
        print("ERROR:", error)
    if errors:
        print(f"{len(errors)} problem(s) in {BREEDS_MAP_PATH}")
        return 1
    print(f"OK: {len(breeds_map)} breeds, {NUM_STANFORD} Stanford labels, {len(akc_names)} AKC breeds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
