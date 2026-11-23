"""MCP server `pet-advisor` (issue #3): 6 tools + the `breeds://list` resource over stdio.

Run:
    python mcp_server/server.py          # stdio, as the agent / Chainlit start it
    mcp dev mcp_server/server.py         # MCP Inspector

Tool bodies live in mcp_server/tools/ and are owned by each part; this file owns the tool names,
the descriptions the SLM reads, and the contract check: every result is validated against
shared/schemas.py before it is returned, so a tool that drifts from the contract fails here.
Tools named in WTF_STUB_TOOLS (or all, with "all") are served from mcp_server/tools/stubs.py.

Rules for tool code:
- Never write to stdout (no print): over stdio, stdout carries the MCP protocol and any other
  output breaks it. Use logging, which goes to stderr.
- Clients start this server with shared.mcp_client.server_params(), so WTF_* and model variables
  set in code or in the terminal reach the tools.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Annotated, Callable, Literal, Optional, TypeVar

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # `mcp dev mcp_server/server.py` / `python mcp_server/server.py`
    sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import config  # noqa: E402
from mcp_server.tools import breed_id, knowledge, names, stubs, traits  # noqa: E402
from shared.breeds import display_names  # noqa: E402
from shared.images import IMAGE_ID_PATTERN  # noqa: E402
from shared.schemas import (BREEDS_LIST_URI, BreedsList, Colour, DescribePetOutput,  # noqa: E402
                            GetBreedTraitsOutput, IdentifyBreedOutput, RecommendBreedsOutput,
                            SearchKnowledgeOutput, Species, SuggestNamesOutput)

mcp = FastMCP("pet-advisor")

try:  # load the AKC CSV and the recommend_breeds pool once at startup; a missing CSV only breaks those tools
    traits.candidate_pool()
except Exception:  # noqa: BLE001
    logging.getLogger(__name__).exception("could not warm the AKC table; get_breed_traits and "
                                          "recommend_breeds will fail until it is readable")

M = TypeVar("M", bound=BaseModel)
ImageId = Annotated[str, Field(pattern=IMAGE_ID_PATTERN,
                               description="ID of the uploaded photo, e.g. 'abc123.jpg'. Never a path or base64.")]


def _impl(module: ModuleType, name: str) -> Callable:
    """The real tool function, or its stub when config.STUB_TOOLS asks for it (read per call)."""
    if ("all" in config.STUB_TOOLS or name in config.STUB_TOOLS) and hasattr(stubs, name):
        return getattr(stubs, name)
    return getattr(module, name)


def _validated(model: type[M], result: BaseModel | dict) -> M:
    """Enforce the contract on whatever a tool module returns (model or plain dict)."""
    data = result.model_dump() if isinstance(result, BaseModel) else result
    return model.model_validate(data)


@mcp.tool()
def identify_breed(image_id: ImageId) -> IdentifyBreedOutput:
    """Identify the dog breed in an uploaded photo.
    Returns the 3 most likely breeds with calibrated confidence and a decision:
    "confident" (name the breed), "top2" (show the two most likely) or "retry" (ask for a clearer photo).
    kind "wild_canid" means the animal is not a dog breed.
    Example: identify_breed(image_id="abc123.jpg")"""
    return _validated(IdentifyBreedOutput, _impl(breed_id, "identify_breed")(image_id))


@mcp.tool()
def search_knowledge(
    query: Annotated[str, Field(description="The question in plain words.")],
    breed: Annotated[Optional[str], Field(description="Breed name, when it is known.")] = None,
    section: Annotated[Optional[str], Field(description="Overview, Temperament, Exercise, Grooming, Health, "
                                                        "Training, Feeding, Living Environment, History or Other.")] = None,
) -> SearchKnowledgeOutput:
    """Search breed profiles (facts, care, temperament, exercise). Use for any question about a breed.
    Pass `breed` when it is known. Cite returned chunks as [chunk_id].
    Example: search_knowledge(query="how much exercise", breed="Siberian Husky", section="Exercise")"""
    return _validated(SearchKnowledgeOutput, _impl(knowledge, "search_knowledge")(query, breed, section))


@mcp.tool()
def get_breed_traits(breed: Annotated[str, Field(description="Breed name.")]) -> GetBreedTraitsOutput:
    """AKC scores (1-5) for energy, shedding, grooming, trainability and demeanor, plus height (cm)
    and weight (kg). status "not_found" lists the closest breed names; "no_akc_data" explains why
    there are no scores (e.g. wild canids).
    Example: get_breed_traits(breed="Border Collie")"""
    return _validated(GetBreedTraitsOutput, _impl(traits, "get_breed_traits")(breed))


@mcp.tool()
def recommend_breeds(
    home: Literal["flat", "house", "house_with_garden"],
    hours_alone: Annotated[int, Field(ge=0, le=24, description="Hours the dog is alone on a normal day.")],
    daily_walks_min: Annotated[int, Field(ge=0, le=600, description="Minutes of walks the user can give per day.")],
    kids_at_home: bool,
    has_allergies: bool,
    experience: Literal["none", "some", "experienced"],
) -> RecommendBreedsOutput:
    """Recommend the 3 breeds that best fit the user's questionnaire answers, each with the traits
    that matched and the ones that were close. Ask all six questions before calling.
    Example: recommend_breeds(home="flat", hours_alone=8, daily_walks_min=30, kids_at_home=False,
    has_allergies=True, experience="none")"""
    return _validated(RecommendBreedsOutput, _impl(traits, "recommend_breeds")(
        home, hours_alone, daily_walks_min, kids_at_home, has_allergies, experience))


@mcp.tool()
def describe_pet(image_id: ImageId) -> DescribePetOutput:
    """Describe the pet in an uploaded photo: species, main colour, size and one short sentence.
    Use before suggest_names.
    Example: describe_pet(image_id="abc123.jpg")"""
    return _validated(DescribePetOutput, _impl(breed_id, "describe_pet")(image_id))


@mcp.tool()
def suggest_names(
    species: Species,
    colour: Annotated[Optional[Colour], Field(description="Main colour from describe_pet; null for any colour.")],
    description: Annotated[str, Field(description="Short description from describe_pet.")],
    exclude_names: Annotated[Optional[list[str]], Field(description="Names the user already rejected.")] = None,
) -> SuggestNamesOutput:
    """Suggest pet names: the most common names among the most similar adoption listings, with
    how often each appears and example listing IDs.
    Example: suggest_names(species="dog", colour="golden", description="A fluffy golden puppy", exclude_names=["Max"])"""
    return _validated(SuggestNamesOutput, _impl(names, "suggest_names")(species, colour, description, exclude_names))


@mcp.resource(BREEDS_LIST_URI, name="breeds", mime_type="application/json",
              description="All supported breed display names (use these exact spellings).")
def breeds_list() -> str:
    return json.dumps(BreedsList(display_names()).root, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()  # stdio
