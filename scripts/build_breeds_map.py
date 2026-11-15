"""Build data/breeds_map.json (issue #3): one canonical display name per breed.

Sources:
    data/akc_breeds.csv                      277 AKC breeds (scripts/download_akc.py)
    data/splits/stanford_dogs_val.json       the 120 Stanford Dogs folder names (#6), or --stanford-labels
    data/raw/petfinder/breed_labels.csv      optional: PetFinder breed names (#5, needs Kaggle login)
    data/raw/thedogapi_breeds.json           optional: TheDogAPI /v1/breeds response (needs an API key)

The Stanford folders whose name is not the AKC name are mapped by hand in STANFORD_TO_DISPLAY.
Re-running is safe: the output only depends on the inputs. Check the result with
    python scripts/validate_breeds_map.py

Usage:
    python scripts/build_breeds_map.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import AKC_TABLE_PATH, BREEDS_MAP_PATH  # noqa: E402
from shared.breeds import normalise, stanford_readable_name  # noqa: E402

STANFORD_SPLIT_FILE = ROOT / "data" / "splits" / "stanford_dogs_val.json"
PETFINDER_PATH = ROOT / "data" / "raw" / "petfinder" / "breed_labels.csv"
THEDOGAPI_PATH = ROOT / "data" / "raw" / "thedogapi_breeds.json"

# Stanford folder name (after the WordNet ID) -> display name, where they differ.
STANFORD_TO_DISPLAY = {
    "Japanese_spaniel": "Japanese Chin",
    "Maltese_dog": "Maltese",
    "Pekinese": "Pekingese",
    "Shih-Tzu": "Shih Tzu",
    "Blenheim_spaniel": "Cavalier King Charles Spaniel",
    "toy_terrier": "Manchester Terrier (Toy)",
    "basset": "Basset Hound",
    "bluetick": "Bluetick Coonhound",
    "black-and-tan_coonhound": "Black and Tan Coonhound",
    "Walker_hound": "Treeing Walker Coonhound",
    "redbone": "Redbone Coonhound",
    "Staffordshire_bullterrier": "Staffordshire Bull Terrier",
    "wire-haired_fox_terrier": "Wire Fox Terrier",
    "Airedale": "Airedale Terrier",
    "cairn": "Cairn Terrier",
    "Dandie_Dinmont": "Dandie Dinmont Terrier",
    "Boston_bull": "Boston Terrier",
    "Scotch_terrier": "Scottish Terrier",
    "soft-coated_wheaten_terrier": "Soft Coated Wheaten Terrier",
    "Lhasa": "Lhasa Apso",
    "German_short-haired_pointer": "German Shorthaired Pointer",
    "Brittany_spaniel": "Brittany",
    "clumber": "Clumber Spaniel",
    "English_springer": "English Springer Spaniel",
    "groenendael": "Belgian Sheepdog",
    "malinois": "Belgian Malinois",
    "kelpie": "Australian Kelpie",
    "German_shepherd": "German Shepherd Dog",
    "Doberman": "Doberman Pinscher",
    "Appenzeller": "Appenzeller Sennenhund",
    "EntleBucher": "Entlebucher Mountain Dog",
    "bull_mastiff": "Bullmastiff",
    "malamute": "Alaskan Malamute",
    "Leonberg": "Leonberger",
    "chow": "Chow Chow",
    "Brabancon_griffon": "Brussels Griffon",
    "Pembroke": "Pembroke Welsh Corgi",
    "Cardigan": "Cardigan Welsh Corgi",
    "toy_poodle": "Poodle (Toy)",
    "miniature_poodle": "Poodle (Miniature)",
    "standard_poodle": "Poodle (Standard)",
    "Mexican_hairless": "Xoloitzcuintli",
    # No AKC equivalent (American Eskimo Dog is a different breed)
    "Eskimo_dog": "Eskimo Dog",
    # Wild canids: in Stanford Dogs, but not dog breeds
    "dingo": "Dingo",
    "dhole": "Dhole",
    "African_hunting_dog": "African Wild Dog",
}
NO_AKC = {"Eskimo Dog", "Dingo", "Dhole", "African Wild Dog"}
WILD_CANIDS = {"Dingo", "Dhole", "African Wild Dog"}

# PetFinder / TheDogAPI spellings that differ from the display name. Fill in from the
# "unmatched" list this script prints once the source files are downloaded (#5).
OTHER_SOURCE_ALIASES: dict[str, str] = {}


def akc_names(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as file:
        return [row[0] for row in list(csv.reader(file))[1:] if row and row[0]]


def stanford_labels(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):  # plain list of folder names
        return sorted(set(data))
    return sorted({Path(p).parts[0] for p in data["val_files"]})


def petfinder_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return [row["BreedName"] for row in csv.DictReader(file) if row.get("Type") == "1"]  # 1 = dog


def thedogapi_names(path: Path) -> list[str]:
    return [item["name"] for item in json.loads(path.read_text(encoding="utf-8"))] if path.exists() else []


def match_names(source: str, names: list[str], displays: dict[str, str]) -> dict[str, str]:
    """display -> source spelling, using exact (normalised) names plus OTHER_SOURCE_ALIASES."""
    matched: dict[str, str] = {}
    unmatched = []
    for name in names:
        display = displays.get(normalise(OTHER_SOURCE_ALIASES.get(name, name)))
        if display and display not in matched:
            matched[display] = name
        else:
            unmatched.append(name)
    if unmatched:
        print(f"{source}: {len(unmatched)} names without a display match (add to OTHER_SOURCE_ALIASES "
              f"if they are the same breed): {unmatched}")
    return matched


def build(akc: list[str], stanford: list[str], petfinder: list[str], dogapi: list[str]) -> dict[str, dict]:
    akc_by_norm = {normalise(name): name for name in akc}
    entries: dict[str, dict] = {
        name: {"display": name, "kind": "breed", "stanford_label": None, "akc_name": name,
               "petfinder_name": None, "thedogapi_name": None}
        for name in akc
    }
    for label in stanford:
        folder = label.split("-", 1)[1]
        display = STANFORD_TO_DISPLAY.get(folder) or akc_by_norm.get(normalise(stanford_readable_name(label)))
        if display is None:
            sys.exit(f"No display name for Stanford folder {label!r}: add it to STANFORD_TO_DISPLAY")
        if display in NO_AKC:
            entries[display] = {"display": display, "kind": "wild_canid" if display in WILD_CANIDS else "breed",
                                "stanford_label": label, "akc_name": None,
                                "petfinder_name": None, "thedogapi_name": None}
        elif display not in entries:
            sys.exit(f"{label!r} maps to {display!r}, which is not in the AKC table")
        elif entries[display]["stanford_label"]:
            sys.exit(f"{display!r} has two Stanford folders")
        else:
            entries[display]["stanford_label"] = label

    displays = {normalise(d): d for d in entries}
    for display, name in match_names("PetFinder", petfinder, displays).items():
        entries[display]["petfinder_name"] = name
    for display, name in match_names("TheDogAPI", dogapi, displays).items():
        entries[display]["thedogapi_name"] = name
    return dict(sorted(entries.items(), key=lambda item: item[0].casefold()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stanford-labels", type=Path, default=STANFORD_SPLIT_FILE,
                        help="JSON list of the 120 folder names, or the #6 split file")
    parser.add_argument("--out", type=Path, default=BREEDS_MAP_PATH)
    args = parser.parse_args()

    if not args.stanford_labels.exists():
        sys.exit(f"{args.stanford_labels} not found (it comes from #6). Pass --stanford-labels.")
    breeds_map = build(akc_names(AKC_TABLE_PATH), stanford_labels(args.stanford_labels),
                       petfinder_names(PETFINDER_PATH), thedogapi_names(THEDOGAPI_PATH))
    args.out.write_text(json.dumps(breeds_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    n_stanford = sum(1 for e in breeds_map.values() if e["stanford_label"])
    n_pf = sum(1 for e in breeds_map.values() if e["petfinder_name"])
    n_api = sum(1 for e in breeds_map.values() if e["thedogapi_name"])
    print(f"Wrote {len(breeds_map)} breeds ({n_stanford} Stanford, {n_pf} PetFinder, {n_api} TheDogAPI) to {args.out}")


if __name__ == "__main__":
    main()
