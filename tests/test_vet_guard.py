"""vet_check: the code-enforced vet/danger guard (issue #11)."""

import json

import pytest

import config
from agent.guards import vet_check


@pytest.mark.parametrize("message", [
    "my dog ate chocolate",
    "He ate grapes this morning",
    "My dog is not breathing!",
    "our puppy had a seizure",
    "My dog collapsed in the park",
    "I think my dog was poisoned",
    "My dog overdosed on his pills",
    "my dog is sick and won't eat",
    "My cat keeps vomiting",
    "our puppy has diarrhoea",
    "My dog is unconscious",
    "This is an emergency, my dog is bleeding",
])
def test_vet_check_fires(message):
    assert vet_check(message)


@pytest.mark.parametrize("message", [
    "how much exercise",
    "How much exercise does a Siberian Husky need?",
    "Is a Beagle good with kids?",
    "What bleeding disorder is genetically inherited in Dobermans?",
    "What acute gastrointestinal emergency is common in Great Danes?",
    "Which veterinary emergency clinic in downtown Cluj-Napoca is open right now?",
])
def test_vet_check_does_not_fire(message):
    assert not vet_check(message)


def _golden(part):
    with config.GOLDEN_PATH.open(encoding="utf-8") as file:
        items = [json.loads(line) for line in file if line.strip()]
    return [item for item in items if item["part"] == part]


def test_golden_part_c_vet_referrals_fire_and_the_rest_do_not():
    for item in _golden("C"):
        expected = item["expected"]["response_type"] == "vet_referral"
        assert vet_check(item["input"]["message"]) == expected, item["id"]


def test_golden_part_b_breed_questions_are_never_blocked():
    for item in _golden("B"):
        assert not vet_check(item["input"]["query"]), item["id"]
