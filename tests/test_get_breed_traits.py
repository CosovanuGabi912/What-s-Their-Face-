"""get_breed_traits on the real AKC table (issue #11)."""

import pytest

from mcp_server.tools.traits import get_breed_traits

NUMERIC_FIELDS = ["energy", "shedding", "grooming", "trainability", "demeanor",
                  "height_cm_min", "height_cm_max", "weight_kg_min", "weight_kg_max"]


def test_known_breed_returns_every_numeric_field():
    result = get_breed_traits("Siberian Husky")
    assert result.status == "found" and result.closest_matches == []
    traits = result.traits
    assert traits.breed == "Siberian Husky"
    for field in NUMERIC_FIELDS:
        assert isinstance(getattr(traits, field), (int, float)), field
    assert traits.energy == 4                      # AKC 0.8 x 5
    assert traits.height_cm_min <= traits.height_cm_max
    assert traits.weight_kg_min <= traits.weight_kg_max


def test_any_spelling_resolves_to_the_display_name():
    assert get_breed_traits("siberian husky").traits.breed == "Siberian Husky"


@pytest.mark.parametrize("name", ["Siberian_HuskyXYZ", "Golden Retreiver", "Dragon Hound"])
def test_unknown_breed_is_not_found_with_at_most_three_matches(name):
    result = get_breed_traits(name)
    assert result.status == "not_found" and result.traits is None
    assert len(result.closest_matches) <= 3


def test_misspelt_husky_suggests_siberian_husky():
    assert "Siberian Husky" in get_breed_traits("Siberian_HuskyXYZ").closest_matches


@pytest.mark.parametrize("name", ["Dingo", "Dhole", "African Wild Dog"])
def test_wild_canids_have_no_akc_data_and_no_fuzzy_matches(name):
    result = get_breed_traits(name)
    assert result.status == "no_akc_data"
    assert result.closest_matches == [] and result.traits is None
    assert name in result.message
