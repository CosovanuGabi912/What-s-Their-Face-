"""Lookups on data/breeds_map.json (issue #3).

Every component stores and compares breeds by their canonical `display` name. `to_display()`
turns any known spelling into it:

    display / AKC / PetFinder / TheDogAPI name     "German Shepherd Dog"
    Stanford label, WordNet ID or folder name      "n02106662-German_shepherd", "n02106662"
    readable Stanford name (#6 class_names.json)   "German Shepherd", "Japanese Spaniel"

Matching ignores case, accents, '_' and '-' so "shih-tzu" and "Shih Tzu" are the same breed.
"""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from functools import lru_cache

import config


def normalise(name: str) -> str:
    """'Löwchen' -> 'lowchen', 'German_short-haired pointer' -> 'german short haired pointer'."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[_\-]", " ", text.casefold())
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return " ".join(text.split())


def slugify(name: str) -> str:
    """Display name -> ID slug used in chunk IDs: 'Poodle (Toy)' -> 'poodle-toy'."""
    return normalise(name).replace(" ", "-")


def stanford_readable_name(label: str) -> str:
    """'n02085782-Japanese_spaniel' -> 'Japanese Spaniel' (the rule train_classifier.py used before #3)."""
    words = label.split("-", 1)[-1].replace("_", " ").split()
    return " ".join(word[0].upper() + word[1:] for word in words)


@lru_cache(maxsize=1)
def load_breeds_map() -> dict[str, dict]:
    """The full map, keyed by display name."""
    return json.loads(config.BREEDS_MAP_PATH.read_text(encoding="utf-8"))


def display_names() -> list[str]:
    """All display names, sorted case-insensitively (what `breeds://list` serves)."""
    return sorted(load_breeds_map(), key=str.casefold)


@lru_cache(maxsize=1)
def _alias_index() -> dict[str, str]:
    index: dict[str, str] = {}

    def add(alias: str | None, display: str) -> None:
        if alias:
            index.setdefault(normalise(alias), display)

    entries = load_breeds_map()
    for display in entries:  # display names first so they always win
        add(display, display)
    for display, entry in entries.items():
        for key in ("akc_name", "petfinder_name", "thedogapi_name"):
            add(entry.get(key), display)
        label = entry.get("stanford_label")
        if label:
            add(label, display)
            add(label.split("-", 1)[0], display)           # WordNet ID
            add(label.split("-", 1)[1], display)           # folder name
            add(stanford_readable_name(label), display)    # pre-#3 class name
    return index


def to_display(name: str | None) -> str | None:
    """Canonical display name for any known spelling, or None when the breed is unknown."""
    if not name:
        return None
    return _alias_index().get(normalise(name))


def closest_breeds(name: str, n: int = 3) -> list[str]:
    """Up to n display names that look most like `name` (for 'did you mean' replies)."""
    index = _alias_index()
    found: list[str] = []
    for alias in difflib.get_close_matches(normalise(name), list(index), n=n * 4, cutoff=0.0):
        if index[alias] not in found:
            found.append(index[alias])
        if len(found) == n:
            break
    return found


def get_entry(name: str) -> dict | None:
    display = to_display(name)
    return load_breeds_map()[display] if display else None
