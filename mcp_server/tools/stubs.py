"""Stub implementations of the MCP tools (issue #3).

They ignore models and databases and return fixed, schema-valid data, so the agent (Part C), the
UI (Part E) and the contract tests work on any laptop without model weights or Chroma.
This file stays after the real tools land; the server uses a stub for a tool when the env var
WTF_STUB_TOOLS lists it (e.g. "identify_breed,describe_pet") or is "all". See config.STUB_TOOLS.

identify_breed / describe_pet ignore the photo, but the image_id prefix picks a scenario so
every agent branch can be exercised:
    dingo*   -> wild canid (out-of-scope reply)
    unsure*  -> decision "top2"
    blurry*  -> decision "retry"
    cat*     -> describe_pet says species "cat"
    anything else -> German Shepherd Dog at 0.91 (confident)
"""

from __future__ import annotations

from shared.breeds import load_breeds_map, slugify
from shared.images import is_valid_image_id
from shared.schemas import (BreedCandidate, DescribePetOutput, IdentifyBreedOutput, KnowledgeChunk,
                            NameSuggestion, RecommendBreedsInput, RecommendBreedsOutput,
                            RecommendedBreed, SearchKnowledgeInput, SearchKnowledgeOutput,
                            SuggestNamesInput, SuggestNamesOutput, breed_decision)


def _check_image_id(image_id: str) -> None:
    if not is_valid_image_id(image_id):
        raise ValueError(f"Invalid image_id: {image_id!r} (expected a file name like 'abc123.jpg')")


# --- identify_breed (real one: #10) --------------------------------------------------------
_BREED_SCENARIOS = {
    "dingo": [("Dingo", 0.88), ("Basenji", 0.06), ("Australian Kelpie", 0.03)],
    "unsure": [("Siberian Husky", 0.64), ("Eskimo Dog", 0.29), ("Alaskan Malamute", 0.04)],
    "blurry": [("Labrador Retriever", 0.31), ("Golden Retriever", 0.22), ("Beagle", 0.12)],
}
_BREED_DEFAULT = [("German Shepherd Dog", 0.91), ("Belgian Malinois", 0.05), ("Belgian Sheepdog", 0.02)]


def identify_breed(image_id: str) -> IdentifyBreedOutput:
    _check_image_id(image_id)
    top3 = next((v for k, v in _BREED_SCENARIOS.items() if image_id.lower().startswith(k)), _BREED_DEFAULT)
    decision = breed_decision(top3[0][1])
    return IdentifyBreedOutput(
        top3=[BreedCandidate(breed=b, confidence=c) for b, c in top3],
        decision=decision,
        is_confident=decision == "confident",
        kind=load_breeds_map()[top3[0][0]]["kind"],
    )


# --- describe_pet (real one: Week 9, Qwen2.5-VL-3B) ----------------------------------------
def describe_pet(image_id: str) -> DescribePetOutput:
    _check_image_id(image_id)
    if image_id.lower().startswith("cat"):
        return DescribePetOutput(species="cat", main_colour="gray", size="small",
                                 description="A small grey tabby cat with green eyes.")
    return DescribePetOutput(species="dog", main_colour="brown", size="large",
                             description="A large brown and black dog with upright ears and an alert look.")


# --- search_knowledge (real one: Part A, #5 / Week 8) --------------------------------------
_KB_TEXT = {
    "Overview": "{breed} is a well-known breed. This is placeholder text from the stub knowledge tool.",
    "Exercise": "{breed} needs regular daily exercise; most adults do well with one to two hours of activity.",
    "Temperament": "{breed} is usually loyal and alert, and needs early socialisation.",
}


def search_knowledge(query: str, breed: str | None = None, section: str | None = None) -> SearchKnowledgeOutput:
    if not (query or "").strip():  # same as the real tool: nothing to search for
        return SearchKnowledgeOutput(chunks=[])
    args = SearchKnowledgeInput(query=query, breed=breed, section=section)  # same breed / section rules as the real tool
    target = args.breed or "German Shepherd Dog"
    sections = [args.section] if args.section else list(_KB_TEXT)
    return SearchKnowledgeOutput(chunks=[
        KnowledgeChunk(
            chunk_id=f"kb:{slugify(target)}:{slugify(name)}:0",
            breed=target,
            section=name,
            text=_KB_TEXT.get(name, "Placeholder {section} text for {breed}.").format(breed=target, section=name),
            score=round(0.9 - 0.1 * i, 2),
        )
        for i, name in enumerate(sections[:3])
    ])


# --- recommend_breeds (real one: Part C, Week 8) -------------------------------------------
def recommend_breeds(home: str, hours_alone: int, daily_walks_min: int, kids_at_home: bool,
                     has_allergies: bool, experience: str) -> RecommendBreedsOutput:
    RecommendBreedsInput(home=home, hours_alone=hours_alone, daily_walks_min=daily_walks_min,
                         kids_at_home=kids_at_home, has_allergies=has_allergies, experience=experience)
    return RecommendBreedsOutput(top3=[
        RecommendedBreed(breed="Cavalier King Charles Spaniel", score=0.86,
                         matched_traits=["small size suits a flat", "friendly with children"],
                         close_traits=["sheds moderately"]),
        RecommendedBreed(breed="Bichon Frise", score=0.81,
                         matched_traits=["low shedding suits allergies", "small size suits a flat"],
                         close_traits=["needs regular grooming"]),
        RecommendedBreed(breed="Poodle (Miniature)", score=0.78,
                         matched_traits=["low shedding suits allergies", "easy to train"],
                         close_traits=["energy is a bit high for long hours alone"]),
    ])


# --- suggest_names (real one: Part A, Week 9) ----------------------------------------------
_NAMES = {
    "dog": [("Max", 4, ["pf:86e1089a3", "pf:3422e4906"]), ("Bella", 3, ["pf:6296e909a"]),
            ("Rocky", 2, ["pf:5842f1ff5"]), ("Luna", 2, ["pf:850a43f90"]), ("Buddy", 1, ["pf:d24c30b4b"])],
    "cat": [("Kitty", 3, ["pf:a9ee5ac1b"]), ("Tiger", 2, ["pf:c5e5f9cc4"]), ("Mimi", 2, ["pf:0e5a1b7d2"]),
            ("Oreo", 1, ["pf:4b1e2f6a0"]), ("Snowy", 1, ["pf:7d3c9e8b1"])],
}


def suggest_names(species: str, colour: str | None, description: str,
                  exclude_names: list[str] | None = None) -> SuggestNamesOutput:
    args = SuggestNamesInput(species=species, colour=colour, description=description, exclude_names=exclude_names)
    excluded = {name.casefold() for name in args.exclude_names or []}
    return SuggestNamesOutput(names=[
        NameSuggestion(name=name, count=count, example_listing_ids=ids)
        for name, count, ids in _NAMES[args.species] if name.casefold() not in excluded
    ])
