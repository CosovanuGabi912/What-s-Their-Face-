"""recommend_breeds: AKC hard filters and soft scoring in code (issue #11)."""

import itertools

import pytest
from pydantic import ValidationError

import config
from mcp_server.tools import traits as traits_tool
from mcp_server.tools.traits import candidate_pool, get_breed_traits, recommend_breeds, target_energy


def _traits(name):
    return get_breed_traits(name).traits


def test_flat_long_hours_no_experience_excludes_high_energy_and_large_breeds():
    result = recommend_breeds(home="flat", hours_alone=9, daily_walks_min=30, kids_at_home=False,
                              has_allergies=False, experience="none")
    assert 1 <= len(result.top3) <= 3
    for item in result.top3:
        traits = _traits(item.breed)
        assert traits.energy < 5, item.breed
        assert traits.weight_kg_max <= config.FLAT_MAX_WEIGHT_KG, item.breed


def test_allergies_exclude_high_shedding_breeds():
    result = recommend_breeds(home="house", hours_alone=2, daily_walks_min=60, kids_at_home=True,
                              has_allergies=True, experience="some")
    assert result.top3
    for item in result.top3:
        assert _traits(item.breed).shedding < config.HIGH_SHEDDING_SCORE, item.breed


@pytest.mark.parametrize("home,hours,walks,kids,allergies,experience", list(itertools.product(
    ["flat", "house", "house_with_garden"], [0, 8, 12], [0, 45, 120], [False, True], [False, True],
    ["none", "experienced"])))
def test_every_profile_gets_breeds_that_pass_the_hard_filters(home, hours, walks, kids, allergies, experience):
    args = dict(home=home, hours_alone=hours, daily_walks_min=walks, kids_at_home=kids,
                has_allergies=allergies, experience=experience)
    result = recommend_breeds(**args)
    assert len(result.top3) == 3
    scores = [item.score for item in result.top3]
    assert scores == sorted(scores, reverse=True)
    for item in result.top3:
        traits = _traits(item.breed)
        if home == "flat":
            assert traits.weight_kg_max <= 25
        if hours >= 8:
            assert traits.energy != 5
        if allergies:
            assert traits.shedding < 4
        assert item.matched_traits or item.close_traits


def test_is_deterministic():
    args = ("flat", 8, 30, True, True, "none")
    assert recommend_breeds(*args) == recommend_breeds(*args)


def test_ties_are_broken_by_display_name():
    result = recommend_breeds("house", 0, 45, False, False, "experienced")   # score = 2 - |energy - 3|
    top = [item for item in result.top3 if item.score == result.top3[0].score]
    assert [item.breed for item in top] == sorted(item.breed for item in top)


@pytest.mark.parametrize("minutes,energy", [(0, 2), (29, 2), (30, 3), (59, 3), (60, 4), (89, 4), (90, 5), (300, 5)])
def test_target_energy(minutes, energy):
    assert target_energy(minutes) == energy


def test_candidate_pool_only_has_dog_breeds_with_the_filter_fields():
    pool = candidate_pool()
    assert pool
    names = {traits.breed for traits in pool}
    assert not names & {"Dingo", "Dhole", "African Wild Dog", "Eskimo Dog"}
    assert all(None not in (t.energy, t.shedding, t.weight_kg_max) for t in pool)
    assert "Bedlington Terrier" not in names        # no AKC shedding score -> cannot be checked


def test_candidate_pool_drops_breeds_without_knowledge_chunks(monkeypatch):
    candidate_pool.cache_clear()
    monkeypatch.setattr(traits_tool, "_breeds_with_knowledge", lambda: {"Beagle", "Pug"})
    try:
        assert {t.breed for t in candidate_pool()} == {"Beagle", "Pug"}
    finally:
        candidate_pool.cache_clear()


def test_invalid_answers_raise():
    with pytest.raises(ValidationError):
        recommend_breeds("castle", 2, 30, False, False, "none")
