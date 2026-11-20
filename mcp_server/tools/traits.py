"""get_breed_traits and recommend_breeds (issues #3 and #11): both read the AKC table.

data/akc_breeds.csv is loaded once into a pandas DataFrame (indexed by AKC name) the first time a
tool needs it; server.py warms it at startup. Display names map to AKC rows via breeds_map.json's
`akc_name` field.

get_breed_traits never raises: unknown breeds get `not_found` with the closest names, breeds without
AKC data (wild canids, Eskimo Dog) get `no_akc_data` with a message and no fuzzy suggestions
(difflib would offer "Jindo" for "Dingo").

recommend_breeds is rules + scoring in code, never the SLM (docs/milestone2.md, "Why recommend_breeds
does not use the SLM"). Golden Part D is built from the same rules:
    hard filters (a breed that fails one can never appear; thresholds in config.py)
        home == "flat"      -> weight_kg_max <= FLAT_MAX_WEIGHT_KG
        hours_alone >= 8    -> energy != 5
        has_allergies       -> shedding < 4
    soft score (top 3, ties broken by display name)
        2 - |energy - target_energy(daily_walks_min)|
        + (5 - energy) * 0.5                 if home == "flat"
        + (demeanor or 3)                    if kids_at_home
        + ((trainability or 3) - 1)          if experience == "none"
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache

import pandas as pd

import config
from shared.breeds import closest_breeds, get_entry, load_breeds_map
from shared.schemas import (BreedTraits, GetBreedTraitsOutput, RecommendBreedsInput, RecommendBreedsOutput,
                            RecommendedBreed)

logger = logging.getLogger(__name__)

# breed traits field -> AKC CSV column
_SCORE_COLUMNS = {
    "energy": "energy_level_value",
    "shedding": "shedding_value",
    "grooming": "grooming_frequency_value",
    "trainability": "trainability_value",
    "demeanor": "demeanor_value",
}
_RANGE_COLUMNS = {  # the CSV already stores cm and kg
    "height_cm_min": "min_height",
    "height_cm_max": "max_height",
    "weight_kg_min": "min_weight",
    "weight_kg_max": "max_weight",
}


# --- AKC table ---------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def akc_table() -> pd.DataFrame:
    """The AKC CSV, indexed by AKC breed name (its first, unnamed column)."""
    return pd.read_csv(config.AKC_TABLE_PATH, index_col=0)


def _float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _score(value) -> int | None:
    """AKC 0.2-1.0 -> 1-5. Empty cell -> None."""
    number = _float(value)
    return None if number is None else min(5, max(1, round(number * 5)))


def _traits(display: str, akc_name: str) -> BreedTraits | None:
    table = akc_table()
    if akc_name not in table.index:
        return None
    row = table.loc[akc_name]
    return BreedTraits(
        breed=display,
        **{field: _score(row.get(col)) for field, col in _SCORE_COLUMNS.items()},
        **{field: _float(row.get(col)) for field, col in _RANGE_COLUMNS.items()},
    )


# --- get_breed_traits --------------------------------------------------------------------------
def get_breed_traits(breed: str) -> GetBreedTraitsOutput:
    entry = get_entry(breed)
    if entry is None:
        return GetBreedTraitsOutput(status="not_found", closest_matches=closest_breeds(breed, 3))
    display = entry["display"]
    if entry["akc_name"] is None:
        reason = "is a wild canid, not a dog breed" if entry["kind"] == "wild_canid" else "is not an AKC breed"
        return GetBreedTraitsOutput(status="no_akc_data",
                                    message=f"{display} {reason}; there are no AKC trait scores for it.")
    traits = _traits(display, entry["akc_name"])
    if traits is None:  # breeds_map and the CSV disagree; validate_breeds_map.py should catch this
        return GetBreedTraitsOutput(status="no_akc_data", message=f"{display} is missing from the AKC table.")
    return GetBreedTraitsOutput(status="found", traits=traits)


# --- recommend_breeds --------------------------------------------------------------------------
def _breeds_with_knowledge() -> set[str] | None:
    """Display names with at least one chunk in the Chroma knowledge collection, so the agent can
    cite a reason for every recommended breed. None when the collection has not been ingested yet
    (then no breed is dropped for this reason; every AKC breed has a raw Wikipedia profile)."""
    if not config.CHROMA_KNOWLEDGE_DIR.exists():
        logger.warning("knowledge collection not found at %s; recommend_breeds skips the coverage check",
                       config.CHROMA_KNOWLEDGE_DIR)
        return None
    try:
        import chromadb

        collection = chromadb.PersistentClient(path=str(config.CHROMA_KNOWLEDGE_DIR)).get_collection("knowledge")
        metadatas = collection.get(include=["metadatas"])["metadatas"] or []
    except Exception:  # noqa: BLE001  (missing collection, chromadb not installed, ...)
        logger.warning("could not read the knowledge collection; recommend_breeds skips the coverage check",
                       exc_info=True)
        return None
    return {meta["breed"] for meta in metadatas if meta and meta.get("breed")}


@lru_cache(maxsize=1)
def candidate_pool() -> tuple[BreedTraits, ...]:
    """Breeds recommend_breeds may return: kind "breed", an AKC row, and the energy, shedding and
    max-weight values the hard filters need (a breed missing one can never be checked, so it is out)."""
    covered = _breeds_with_knowledge()
    pool = []
    for display, entry in sorted(load_breeds_map().items()):
        if entry["kind"] != "breed" or not entry["akc_name"]:
            continue
        if covered is not None and display not in covered:
            continue
        traits = _traits(display, entry["akc_name"])
        if traits is None or None in (traits.energy, traits.shedding, traits.weight_kg_max):
            continue
        pool.append(traits)
    return tuple(pool)


def target_energy(daily_walks_min: int) -> int:
    """Energy score that fits the minutes of walking the user can give per day."""
    if daily_walks_min < 30:
        return 2
    if daily_walks_min < 60:
        return 3
    if daily_walks_min < 90:
        return 4
    return 5


def passes_hard_filters(traits: BreedTraits, args: RecommendBreedsInput) -> bool:
    """The milestone2 "remove" rules. Golden Part D's forbidden breeds are exactly those failing one."""
    if args.home == "flat" and traits.weight_kg_max > config.FLAT_MAX_WEIGHT_KG:
        return False
    if args.hours_alone >= config.LONG_HOURS_ALONE and traits.energy >= config.HIGH_ENERGY_SCORE:
        return False
    if args.has_allergies and traits.shedding >= config.HIGH_SHEDDING_SCORE:
        return False
    return True


def _score_and_reasons(traits: BreedTraits, args: RecommendBreedsInput) -> tuple[float, list[str], list[str]]:
    """Soft score plus the traits that matched and the trade-offs worth mentioning."""
    matched: list[str] = []
    close: list[str] = []
    energy = traits.energy
    weight = f"{traits.weight_kg_max:.0f} kg"

    target = target_energy(args.daily_walks_min)
    gap = energy - target
    score = 2 - abs(gap)
    walks = f"{args.daily_walks_min} min of walks a day"
    if gap == 0:
        matched.append(f"energy {energy}/5 fits {walks}")
    else:
        close.append(f"energy {energy}/5 is {'higher' if gap > 0 else 'lower'} than {walks} suggests ({target}/5)")

    if args.home == "flat":
        score += (5 - energy) * 0.5
        matched.append(f"up to {weight} suits a flat")
    if args.hours_alone >= config.LONG_HOURS_ALONE:
        matched.append(f"energy {energy}/5 (not the top score) for {args.hours_alone} h alone a day")
    if args.has_allergies:
        matched.append(f"low shedding ({traits.shedding}/5) suits allergies")

    if args.kids_at_home:
        score += traits.demeanor or 3
        if traits.demeanor is None:
            close.append("no AKC demeanor score; check how it does with children")
        elif traits.demeanor >= 4:
            matched.append(f"outgoing demeanor ({traits.demeanor}/5) with children")
        else:
            close.append(f"reserved demeanor ({traits.demeanor}/5) with children")
        if traits.weight_kg_max > config.FLAT_MAX_WEIGHT_KG:
            close.append(f"large dog (up to {weight}); supervise with small children")

    if args.experience == "none":
        score += (traits.trainability or 3) - 1
        if traits.trainability is None:
            close.append("no AKC trainability score for a first dog")
        elif traits.trainability >= 4:
            matched.append(f"easy to train ({traits.trainability}/5) for a first dog")
        else:
            close.append(f"harder to train ({traits.trainability}/5) for a first dog")

    return float(score), matched, close


def recommend_breeds(home: str, hours_alone: int, daily_walks_min: int, kids_at_home: bool,
                     has_allergies: bool, experience: str) -> RecommendBreedsOutput:
    args = RecommendBreedsInput(home=home, hours_alone=hours_alone, daily_walks_min=daily_walks_min,
                                kids_at_home=kids_at_home, has_allergies=has_allergies, experience=experience)
    ranked = []
    for traits in candidate_pool():
        if not passes_hard_filters(traits, args):
            continue
        score, matched, close = _score_and_reasons(traits, args)
        ranked.append((score, traits.breed, matched, close))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return RecommendBreedsOutput(top3=[
        RecommendedBreed(breed=breed, score=score, matched_traits=matched, close_traits=close)
        for score, breed, matched, close in ranked[:3]
    ])
