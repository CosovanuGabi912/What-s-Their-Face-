"""Shared contracts (issue #3): inputs and outputs of the 6 MCP tools, the `breeds://list`
resource, golden-dataset items and tool-call trace logs.

The MCP server validates every tool result against these models before returning it, so a
tool that drifts from the contract fails loudly in its own tests instead of confusing the agent.

IDs:
    chunk IDs    "kb:<breed-slug>:<section-slug>:<idx>"  e.g. "kb:siberian-husky:exercise:0"
    listing IDs  "pf:<PetID>"                             e.g. "pf:86e1089a3"
    image IDs    a plain file name under IMAGE_UPLOAD_DIR  e.g. "abc123.jpg" (see shared/images.py)
Answers cite chunks as "[kb:siberian-husky:exercise:0]" (CHUNK_ID_RE finds them).
"""

from __future__ import annotations

import re
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator, model_validator

import config
from shared.breeds import closest_breeds, load_breeds_map, resolve_breed, to_display
from shared.images import IMAGE_ID_PATTERN

CHUNK_ID_RE = r"\[(kb:[a-z0-9-]+:[a-z0-9-]+:\d+)\]"     # finds citations in answer text
CHUNK_ID_PATTERN = r"^kb:[a-z0-9-]+:[a-z0-9-]+:\d+$"    # one chunk ID
LISTING_ID_PATTERN = r"^pf:[A-Za-z0-9]+$"               # one PetFinder listing ID
BREEDS_LIST_URI = "breeds://list"

Colour = Literal["black", "brown", "golden", "yellow", "cream", "gray", "white"]  # == config.PET_COLOURS
Species = Literal["dog", "cat"]
Decision = Literal["confident", "top2", "retry"]
BreedKind = Literal["breed", "wild_canid"]
Score = Optional[int]  # 1-5, None when the AKC cell is empty


class Contract(BaseModel):
    """Base for every contract model: unknown fields are an error, not silently dropped."""
    model_config = ConfigDict(extra="forbid")


def _display_name(value: str) -> str:
    """Output fields must already hold the canonical breeds_map display name."""
    if value not in load_breeds_map():
        suggestion = to_display(value)
        hint = f" (use the display name {suggestion!r})" if suggestion else ""
        raise ValueError(f"{value!r} is not a breeds_map display name{hint}")
    return value


def _resolve_breed(value: Optional[str]) -> Optional[str]:
    """Inputs may use any known spelling, or a partial name that fits one breed; it is turned into the display
    name or rejected."""
    if value is None or value.strip() == "":
        return None
    display = resolve_breed(value)
    if display is None:
        raise ValueError(f"Unknown breed {value!r}. Closest breeds: {closest_breeds(value)}")
    return display


def breed_decision(top1_confidence: float) -> Decision:
    """Agent decision 2, in code: thresholds come from config (tuned by #10)."""
    if top1_confidence >= config.CONFIDENCE_HIGH:
        return "confident"
    if top1_confidence >= config.CONFIDENCE_LOW:
        return "top2"
    return "retry"


# --- 1. identify_breed -------------------------------------------------------------------
class IdentifyBreedInput(Contract):
    image_id: str = Field(pattern=IMAGE_ID_PATTERN)  # filename under IMAGE_UPLOAD_DIR, e.g. "abc123.jpg"


class BreedCandidate(Contract):
    breed: str                                   # breeds_map display name
    confidence: float = Field(ge=0.0, le=1.0)    # calibrated (temperature-scaled) probability

    _breed = field_validator("breed")(_display_name)


class IdentifyBreedOutput(Contract):
    top3: List[BreedCandidate] = Field(min_length=1, max_length=3)  # sorted, most likely first
    decision: Decision          # breed_decision(top3[0].confidence)
    is_confident: bool          # == (decision == "confident")
    kind: BreedKind             # breeds_map[top1]["kind"]; wild_canid -> out-of-scope reply

    @model_validator(mode="after")
    def _consistent(self) -> "IdentifyBreedOutput":
        confidences = [c.confidence for c in self.top3]
        if confidences != sorted(confidences, reverse=True):
            raise ValueError("top3 must be sorted by confidence, highest first")
        if self.is_confident != (self.decision == "confident"):
            raise ValueError("is_confident must equal (decision == 'confident')")
        if self.kind != load_breeds_map()[self.top3[0].breed]["kind"]:
            raise ValueError("kind must be breeds_map[top3[0].breed]['kind']")
        return self


# --- 2. search_knowledge -----------------------------------------------------------------
class SearchKnowledgeInput(Contract):
    query: str = Field(min_length=1)
    breed: Optional[str] = None     # any spelling; resolved to the display name (Chroma filter)
    section: Optional[str] = None   # one of config.KNOWLEDGE_SECTIONS

    _breed = field_validator("breed")(_resolve_breed)

    @field_validator("section")
    @classmethod
    def _known_section(cls, value: Optional[str]) -> Optional[str]:
        if value is None or value.strip() == "":
            return None
        wanted = value.strip().casefold()
        for section in config.KNOWLEDGE_SECTIONS:
            if section.casefold() == wanted:
                return section
        if wanted in config.SECTION_SYNONYMS:  # "Diet" -> "Feeding"
            return config.SECTION_SYNONYMS[wanted]
        raise ValueError(f"Unknown section {value!r}. Use one of {config.KNOWLEDGE_SECTIONS}")


class KnowledgeChunk(Contract):
    chunk_id: str = Field(pattern=CHUNK_ID_PATTERN)  # "kb:<breed-slug>:<section-slug>:<idx>"
    breed: str
    section: str
    text: str
    score: Optional[float] = None   # similarity score, for logging and eval only

    _breed = field_validator("breed")(_display_name)

    @field_validator("section")
    @classmethod
    def _section(cls, value: str) -> str:
        if value not in config.KNOWLEDGE_SECTIONS:
            raise ValueError(f"section must be one of {config.KNOWLEDGE_SECTIONS}")
        return value


class SearchKnowledgeOutput(Contract):
    chunks: List[KnowledgeChunk] = Field(max_length=config.SEARCH_TOP_K)  # empty = nothing relevant


# --- 3. get_breed_traits -----------------------------------------------------------------
class GetBreedTraitsInput(Contract):
    breed: str = Field(min_length=1)  # any spelling; the tool reports not_found itself


class BreedTraits(Contract):
    # AKC CSV stores scores as 0.2-1.0 floats with gaps (e.g. Bedlington Terrier has no energy/shedding).
    # Score = round(value * 5); None when the CSV cell is empty.
    breed: str                       # display name
    energy: Score = Field(default=None, ge=1, le=5)
    shedding: Score = Field(default=None, ge=1, le=5)
    grooming: Score = Field(default=None, ge=1, le=5)
    trainability: Score = Field(default=None, ge=1, le=5)
    demeanor: Score = Field(default=None, ge=1, le=5)
    height_cm_min: Optional[float] = None
    height_cm_max: Optional[float] = None
    weight_kg_min: Optional[float] = None
    weight_kg_max: Optional[float] = None

    _breed = field_validator("breed")(_display_name)


class GetBreedTraitsOutput(Contract):
    status: Literal["found", "not_found", "no_akc_data"]
    traits: Optional[BreedTraits] = None
    closest_matches: List[str] = Field(default_factory=list, max_length=3)  # only for not_found
    message: Optional[str] = None    # only for no_akc_data, e.g. "Dingo is a wild canid, ..."

    @model_validator(mode="after")
    def _consistent(self) -> "GetBreedTraitsOutput":
        if (self.status == "found") != (self.traits is not None):
            raise ValueError("traits must be set exactly when status == 'found'")
        if self.status == "no_akc_data" and not self.message:
            raise ValueError("no_akc_data needs a message")
        if self.status != "not_found" and self.closest_matches:
            raise ValueError("closest_matches is only for not_found")
        for name in self.closest_matches:
            _display_name(name)
        return self


# --- 4. recommend_breeds -----------------------------------------------------------------
class RecommendBreedsInput(Contract):
    home: Literal["flat", "house", "house_with_garden"]
    hours_alone: int = Field(ge=0, le=24)
    daily_walks_min: int = Field(ge=0, le=600)
    kids_at_home: bool
    has_allergies: bool
    experience: Literal["none", "some", "experienced"]


class RecommendedBreed(Contract):
    breed: str                  # display name
    score: float
    matched_traits: List[str]   # e.g. ["low shedding (2/5) suits allergies"]
    close_traits: List[str]     # near misses / trade-offs to mention

    _breed = field_validator("breed")(_display_name)


class RecommendBreedsOutput(Contract):
    top3: List[RecommendedBreed] = Field(max_length=3)


# --- 5. describe_pet ---------------------------------------------------------------------
class DescribePetInput(Contract):
    image_id: str = Field(pattern=IMAGE_ID_PATTERN)


class DescribePetOutput(Contract):
    species: Species
    main_colour: Optional[Colour]   # None when the VLM fails; the agent then asks the user
    size: Literal["small", "medium", "large"]
    description: str                # one sentence, <= 20 words

    @field_validator("description")
    @classmethod
    def _short(cls, value: str) -> str:
        if not value.strip() or len(value.split()) > 20:
            raise ValueError("description must be one sentence of 1-20 words")
        return value


# --- 6. suggest_names --------------------------------------------------------------------
class SuggestNamesInput(Contract):
    species: Species
    colour: Optional[Colour] = None   # None -> no colour filter
    description: str
    exclude_names: Optional[List[str]] = None


class NameSuggestion(Contract):
    name: str = Field(min_length=1)
    count: int = Field(ge=1)                    # listings among the neighbours with this name
    example_listing_ids: List[str] = Field(min_length=1)  # "pf:<PetID>"

    @field_validator("example_listing_ids")
    @classmethod
    def _listing_ids(cls, values: List[str]) -> List[str]:
        bad = [v for v in values if not re.match(LISTING_ID_PATTERN, v)]
        if bad:
            raise ValueError(f"listing IDs must look like 'pf:<PetID>': {bad}")
        return values


class SuggestNamesOutput(Contract):
    names: List[NameSuggestion]


# --- breeds://list resource ---------------------------------------------------------------
class BreedsList(RootModel[List[str]]):
    """Content of the breeds://list resource: every breeds_map display name, sorted."""


# --- Evaluation and logging ----------------------------------------------------------------
class GoldenItem(Contract):
    """One line of eval/golden.jsonl."""
    id: str                 # e.g. "A_001"
    part: Literal["A", "B", "C", "D", "E", "F"]
    input: dict             # photos are referenced as {"image_id": "A_001.jpg"} (file in eval/images/), never a path
    expected: dict
    source_ids: List[str]   # chunk IDs / listing IDs / AKC breed names the answer must rest on


class ToolCallLog(Contract):
    """One tool call in a conversation trace (JSON lines under config.LOG_DIR)."""
    turn: int = Field(ge=0)
    tool: str
    arguments: dict
    result_ids: List[str]   # chunk IDs, listing IDs or breed names the tool returned
    duration_ms: int = Field(ge=0)


TOOL_SCHEMAS: dict[str, tuple[type[Contract], type[Contract]]] = {
    "identify_breed": (IdentifyBreedInput, IdentifyBreedOutput),
    "search_knowledge": (SearchKnowledgeInput, SearchKnowledgeOutput),
    "get_breed_traits": (GetBreedTraitsInput, GetBreedTraitsOutput),
    "recommend_breeds": (RecommendBreedsInput, RecommendBreedsOutput),
    "describe_pet": (DescribePetInput, DescribePetOutput),
    "suggest_names": (SuggestNamesInput, SuggestNamesOutput),
}
