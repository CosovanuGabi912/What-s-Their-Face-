"""get_breed_traits and recommend_breeds (issue #3).

get_breed_traits already reads the real AKC table (data/akc_breeds.csv) and handles unknown
breeds and breeds without AKC data (wild canids, Eskimo Dog) without raising.
recommend_breeds is still the stub until the AKC rules are implemented (Part C, Week 8).
"""

from __future__ import annotations

import csv
from functools import lru_cache

import config
from mcp_server.tools.stubs import recommend_breeds  # noqa: F401  (placeholder, Week 8)
from shared.breeds import closest_breeds, get_entry
from shared.schemas import BreedTraits, GetBreedTraitsOutput

# breed traits field -> AKC CSV column
_SCORE_COLUMNS = {
    "energy": "energy_level_value",
    "shedding": "shedding_value",
    "grooming": "grooming_frequency_value",
    "trainability": "trainability_value",
    "demeanor": "demeanor_value",
}
_RANGE_COLUMNS = {
    "height_cm_min": "min_height",
    "height_cm_max": "max_height",
    "weight_kg_min": "min_weight",
    "weight_kg_max": "max_weight",
}


@lru_cache(maxsize=1)
def _akc_rows() -> dict[str, dict[str, str]]:
    with config.AKC_TABLE_PATH.open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        return {row[0]: dict(zip(header, row)) for row in reader if row and row[0]}


def _float(cell: str | None) -> float | None:
    try:
        return float(cell) if cell not in (None, "") else None
    except ValueError:
        return None


def _score(cell: str | None) -> int | None:
    """AKC 0.2-1.0 -> 1-5. Empty cell -> None."""
    value = _float(cell)
    return None if value is None else min(5, max(1, round(value * 5)))


def get_breed_traits(breed: str) -> GetBreedTraitsOutput:
    entry = get_entry(breed)
    if entry is None:
        return GetBreedTraitsOutput(status="not_found", closest_matches=closest_breeds(breed, 3))
    display = entry["display"]
    if entry["akc_name"] is None:
        reason = "is a wild canid, not a dog breed" if entry["kind"] == "wild_canid" else "is not an AKC breed"
        return GetBreedTraitsOutput(status="no_akc_data",
                                    message=f"{display} {reason}; there are no AKC trait scores for it.")
    row = _akc_rows().get(entry["akc_name"])
    if row is None:  # breeds_map and the CSV disagree; validate_breeds_map.py should catch this
        return GetBreedTraitsOutput(status="no_akc_data", message=f"{display} is missing from the AKC table.")
    traits = BreedTraits(
        breed=display,
        **{field: _score(row.get(col)) for field, col in _SCORE_COLUMNS.items()},
        **{field: _float(row.get(col)) for field, col in _RANGE_COLUMNS.items()},
    )
    return GetBreedTraitsOutput(status="found", traits=traits)

