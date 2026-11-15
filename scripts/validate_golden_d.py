"""Golden Part D (issue #12): derive and check the expected breeds for the fit questionnaire.

The expected breeds come from data/akc_breeds.csv and the milestone2 rules table ("How the answers
map to the AKC data"), written here independently. This file must NOT import
mcp_server/tools/traits.py: Part D would then only check recommend_breeds against itself.

Hard filters (a breed that breaks one can never be recommended). The milestone table names them in
words; the numbers below make them exact:
    home == "flat"        -> remove "the largest breeds":     AKC max_weight >= 40 kg (52 of 277 breeds)
    hours_alone >= 8      -> remove "the highest energy":     energy score 5/5, "Needs Lots of Activity" (41)
    has_allergies         -> remove "high shedding":          shedding score 4-5/5, "Regularly" or "Frequent" (36)
Scores are round(AKC value * 5), as in shared/schemas.py BreedTraits. A breed whose value for an
active filter is missing in the CSV is in neither list: we cannot say it passes or breaks the rule.

Per item:
    forbidden_breeds   every AKC breed that breaks at least one active hard filter
    acceptable_breeds  every AKC breed that passes all active hard filters (never empty)
    hard_filters       the active filters, for people reading the file

Usage:
    python scripts/validate_golden_d.py           # check eval/golden.jsonl, exit 1 on any problem
    python scripts/validate_golden_d.py --write   # recompute "expected" of every Part D item from its "input"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from shared.schemas import GoldenItem, RecommendBreedsInput  # noqa: E402

FLAT_MAX_WEIGHT_KG = 40.0
LONG_HOURS_ALONE = 8
HIGHEST_ENERGY = 5
HIGH_SHEDDING = 4
MIN_ITEMS = 20
MIN_ITEMS_PER_FILTER = 2


def _number(cell: str) -> Optional[float]:
    try:
        value = float(cell)
    except ValueError:
        return None
    return value if value > 0 else None  # the CSV uses 0.0 for a few unknown weights


def _score(cell: str) -> Optional[int]:
    value = _number(cell)
    return None if value is None else min(5, max(1, round(value * 5)))


def load_akc(path: Path = config.AKC_TABLE_PATH) -> dict[str, dict]:
    """AKC breed name -> the three values the hard filters use (None when missing)."""
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = [dict(zip(header, row)) for row in reader if row and row[0]]
    return {
        row[header[0]]: {
            "max_weight_kg": _number(row["max_weight"]),
            "energy": _score(row["energy_level_value"]),
            "shedding": _score(row["shedding_value"]),
        }
        for row in rows
    }


# A filter returns True when the breed breaks it, False when it passes, None when the value is missing.
Filter = Callable[[dict], Optional[bool]]


def _breaks(value: Optional[float], limit: float) -> Optional[bool]:
    return None if value is None else value >= limit


def active_filters(answers: dict) -> dict[str, Filter]:
    filters: dict[str, Filter] = {}
    if answers["home"] == "flat":
        filters["flat: max weight >= 40 kg"] = lambda b: _breaks(b["max_weight_kg"], FLAT_MAX_WEIGHT_KG)
    if answers["hours_alone"] >= LONG_HOURS_ALONE:
        filters["8+ hours alone: energy 5/5"] = lambda b: _breaks(b["energy"], HIGHEST_ENERGY)
    if answers["has_allergies"]:
        filters["allergies: shedding 4-5/5"] = lambda b: _breaks(b["shedding"], HIGH_SHEDDING)
    return filters


def expected_breeds(answers: dict, akc: dict[str, dict]) -> dict:
    filters = active_filters(answers)
    forbidden, acceptable = [], []
    for breed, values in akc.items():
        results = [rule(values) for rule in filters.values()]
        if any(result is True for result in results):
            forbidden.append(breed)
        elif all(result is False for result in results):
            acceptable.append(breed)
    return {"forbidden_breeds": sorted(forbidden), "acceptable_breeds": sorted(acceptable),
            "hard_filters": list(filters)}


def read_golden(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(items: list[dict], akc: dict[str, dict]) -> list[str]:
    errors: list[str] = []
    part_d = [item for item in items if item.get("part") == "D"]
    if len(part_d) < MIN_ITEMS:
        errors.append(f"{len(part_d)} Part D items, need at least {MIN_ITEMS}")
    ids = [item.get("id") for item in part_d]
    if len(set(ids)) != len(ids):
        errors.append("Part D ids are not unique")

    for item in part_d:
        label = item.get("id", "?")
        try:
            GoldenItem.model_validate(item)
            RecommendBreedsInput.model_validate(item["input"])
        except Exception as error:  # noqa: BLE001 - report every schema problem, not only the first
            errors.append(f"{label}: {error}")
            continue
        expected = item["expected"]
        forbidden, acceptable = set(expected.get("forbidden_breeds", [])), set(expected.get("acceptable_breeds", []))
        rules = active_filters(item["input"])
        if not acceptable:
            errors.append(f"{label}: acceptable_breeds is empty")
        if forbidden & acceptable:
            errors.append(f"{label}: in both lists: {sorted(forbidden & acceptable)}")
        for breed in sorted((forbidden | acceptable) - akc.keys()):
            errors.append(f"{label}: {breed!r} is not in the AKC table")
        for breed in sorted(forbidden & akc.keys()):
            if not any(rule(akc[breed]) is True for rule in rules.values()):
                errors.append(f"{label}: forbidden {breed!r} breaks no hard filter")
        for breed in sorted(acceptable & akc.keys()):
            if not all(rule(akc[breed]) is False for rule in rules.values()):
                errors.append(f"{label}: acceptable {breed!r} does not pass every hard filter")
        if item.get("source_ids") != ["akc_data"]:
            errors.append(f"{label}: source_ids must be ['akc_data']")

    # Coverage targets from issue #12
    valid = [item for item in part_d if _valid_input(item)]
    for name in ("flat: max weight >= 40 kg", "8+ hours alone: energy 5/5", "allergies: shedding 4-5/5"):
        count = sum(name in active_filters(item["input"]) for item in valid)
        if count < MIN_ITEMS_PER_FILTER:
            errors.append(f"hard filter '{name}' is triggered by {count} items, need {MIN_ITEMS_PER_FILTER}")
    for field in RecommendBreedsInput.model_fields:
        values = {json.dumps(item["input"].get(field)) for item in valid}
        if len(values) < 2:
            errors.append(f"questionnaire field {field!r} has only one value across Part D")
    return errors


def _valid_input(item: dict) -> bool:
    try:
        RecommendBreedsInput.model_validate(item.get("input"))
        return True
    except Exception:  # noqa: BLE001
        return False


def write_expected(path: Path, akc: dict[str, dict]) -> int:
    """Recompute "expected" for every Part D line; other lines are kept as they are."""
    lines = path.read_text(encoding="utf-8").splitlines()
    updated = 0
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("part") == "D":
            item["expected"] = expected_breeds(item["input"], akc)
            lines[index] = json.dumps(item, ensure_ascii=False)
            updated += 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", type=Path, default=config.GOLDEN_PATH)
    parser.add_argument("--write", action="store_true", help="recompute expected breeds before checking")
    args = parser.parse_args()

    akc = load_akc()
    if args.write:
        print(f"Recomputed {write_expected(args.golden, akc)} Part D items in {args.golden}")
    errors = validate(read_golden(args.golden), akc)
    for error in errors:
        print("ERROR:", error)
    if errors:
        print(f"{len(errors)} problem(s) in Part D")
        return 1
    part_d = [item for item in read_golden(args.golden) if item["part"] == "D"]
    print(f"OK: {len(part_d)} Part D items, rules checked against {len(akc)} AKC breeds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
