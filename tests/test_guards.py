"""Guards in agent/guards.py (issue #16): citation_check, safety_claim_check, vet_check, out_of_scope_check.
vet_check's full pattern list is covered in tests/test_vet_guard.py (#11)."""

import json

import pytest

import config
from agent.guards import (cited_ids, citation_check, out_of_scope_check, safety_claim_check, safety_claims,
                          unretrieved_ids, vet_check)


# --- citation_check -------------------------------------------------------------------------------
def test_citation_of_a_chunk_that_was_not_retrieved_fails():
    assert citation_check("see [kb:beagle:health:0]", retrieved_chunk_ids=["kb:beagle:exercise:0"]) is False


def test_citations_of_retrieved_chunks_pass():
    answer = "Beagles need an hour a day [kb:beagle:exercise:0] and shed [kb:beagle:grooming:1]."
    assert citation_check(answer, {"kb:beagle:exercise:0", "kb:beagle:grooming:1", "kb:beagle:health:0"})


def test_answer_without_citations_passes():
    assert citation_check("I don't know.", [])


def test_one_bad_citation_among_good_ones_fails_and_is_named():
    answer = "Fact [kb:beagle:exercise:0]. Invented [kb:beagle:health:7]."
    assert not citation_check(answer, ["kb:beagle:exercise:0"])
    assert unretrieved_ids(answer, ["kb:beagle:exercise:0"]) == ["kb:beagle:health:7"]


def test_cited_ids_are_unique_and_in_order():
    assert cited_ids("[kb:a:b:1] x [kb:c:d:0] y [kb:a:b:1]") == ["kb:a:b:1", "kb:c:d:0"]


# --- safety_claim_check ---------------------------------------------------------------------------
@pytest.mark.parametrize("answer", [
    "Carrots are safe for dogs.",
    "Yes, peanut butter is perfectly safe.",
    "Grapes are not toxic to Beagles.",
    "It's harmless in small amounts.",
    "A little cheese won't hurt him.",
    "Apples are safe to give as a treat.",
])
def test_uncited_safety_claims_fail(answer):
    assert safety_claims(answer)
    assert safety_claim_check(answer, ["kb:beagle:feeding:0"]) is False


def test_safety_claim_citing_a_retrieved_chunk_passes():
    answer = "Plain cooked rice is safe for Beagles [kb:beagle:feeding:0]."
    assert safety_claim_check(answer, ["kb:beagle:feeding:0"])


def test_safety_claim_citing_a_chunk_from_another_turn_fails():
    answer = "Plain cooked rice is safe for Beagles [kb:beagle:feeding:0]."
    assert not safety_claim_check(answer, ["kb:beagle:exercise:0"])


def test_each_safety_claim_needs_its_own_citation():
    answer = "Rice is safe [kb:beagle:feeding:0]. Grapes are harmless too."
    assert not safety_claim_check(answer, ["kb:beagle:feeding:0"])


def test_citation_right_after_the_full_stop_backs_the_claim():
    answer = "Rice is safe for Beagles. [kb:beagle:feeding:0] Grapes are toxic."
    assert safety_claim_check(answer, ["kb:beagle:feeding:0"])


def test_warning_and_claim_in_one_sentence_is_still_a_claim():
    assert not safety_claim_check("Bones are not always safe, but rice is safe for dogs.", [])


@pytest.mark.parametrize("answer", [
    "Beagles need about an hour of exercise a day [kb:beagle:exercise:0].",
    "Chocolate is toxic to dogs; please contact your vet.",
    "I don't know.",
    "Grapes are not safe for dogs, contact your vet.",       # a warning, not a safety claim
    "Grapes aren\u2019t safe for dogs.",
    "Cooked bones are never safe to give.",
    "It won't hurt to ask your vet.",
])
def test_answers_without_safety_claims_pass(answer):
    assert safety_claim_check(answer, [])


# --- vet_check (acceptance criterion; the full list is in test_vet_guard.py) -------------------------
def test_vet_check_my_dog_ate_chocolate():
    assert vet_check("my dog ate chocolate") is True
    assert vet_check("How much exercise does a Beagle need?") is False


# --- out_of_scope_check ---------------------------------------------------------------------------
@pytest.mark.parametrize("message", [
    "Where can I adopt a stray kitten in my local city shelter?",
    "Where can I adopt a Beagle?",
    "My pet parrot is losing feathers, what seed should I feed it?",
    "How fast can a cheetah run?",
    "What should I feed my cat?",
])
def test_out_of_scope_fires(message):
    assert out_of_scope_check(message)


@pytest.mark.parametrize("message", [
    "Is a Beagle good with cats?",
    "Do Huskies get along with other dogs and cats?",
    "Name ideas for my grey cat?",                 # suggest_names covers cats
    "How much exercise does an American Foxhound need?",
    "What kind of exercise is recommended for Australian Cattle Dogs?",
    "Which dog breed suits a small flat?",
])
def test_out_of_scope_does_not_fire_on_dog_questions(message):
    assert not out_of_scope_check(message)


def _golden(part):
    with config.GOLDEN_PATH.open(encoding="utf-8") as file:
        return [item for item in map(json.loads, filter(str.strip, file)) if item["part"] == part]


def test_out_of_scope_never_blocks_golden_breed_questions():
    for item in _golden("B"):
        assert not out_of_scope_check(item["input"]["query"]), item["id"]


def test_out_of_scope_only_fires_on_golden_out_of_scope_items():
    for item in _golden("C"):
        if out_of_scope_check(item["input"]["message"]):
            assert item["expected"]["response_type"] == "out_of_scope", item["id"]


@pytest.mark.parametrize("message", ["Can he eat fish?", "Is he OK around cats?", "Will she chase birds?"])
def test_out_of_scope_lets_follow_ups_about_the_users_dog_through(message):
    assert out_of_scope_check(message)                       # no dog in the message alone
    assert not out_of_scope_check(message, dog_context=True)


def test_adoption_stays_out_of_scope_in_a_dog_conversation():
    assert out_of_scope_check("Where can I adopt a cat?", dog_context=True)


@pytest.mark.parametrize("message", ["Which cat breeds shed the least?", "What are the best rabbit breeds for kids?"])
def test_other_animal_breeds_are_out_of_scope(message):
    assert out_of_scope_check(message)


def test_breed_question_that_mentions_a_cat_stays_in_scope():
    assert not out_of_scope_check("Which breed is best for a flat with a cat?")


def test_mentions_dogs_falls_back_to_dog_words_without_breeds_map(monkeypatch):
    from agent import guards

    def missing():
        raise FileNotFoundError("breeds_map.json")

    monkeypatch.setattr(guards, "_breed_name_re", missing)
    assert guards.mentions_dogs("Is my dog OK with cats?")
    assert not guards.mentions_dogs("Tell me about beagles")    # breed names need breeds_map

