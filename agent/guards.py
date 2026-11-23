"""Guards that run in code, not in the prompt (issues #7, #11 and #16). The SLM cannot overrule them.

Before the SLM:
- vet_check: vet or danger question -> the agent replies VET_REPLY and calls no tools
  (agent decision 6 in docs/milestone2.md).
- out_of_scope_check: another animal (cats, birds, wild animals) or adoption -> OUT_OF_SCOPE_REPLY.

On every final answer:
- citation_check: each cited [chunk_id] must have been returned by search_knowledge in this turn.
- safety_claim_check: a sentence saying something is safe ("X is safe", "X is not toxic") must cite a
  chunk retrieved in this turn.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

from shared.breeds import aliases, load_breeds_map, normalise
from shared.schemas import CHUNK_ID_RE

VET_REPLY = ("This sounds like it could be a health emergency, and I can't give medical advice. "
             "Please contact your vet or an emergency animal clinic right away.")
OUT_OF_SCOPE_REPLY = ("Sorry, that's outside what I can help with: I only cover dog breeds, breed "
                      "recommendations and pet names.")

# --- vet_check ------------------------------------------------------------------------------------
# Phrases that mean "see a vet", matched case-insensitively anywhere in the message, but only when the
# message is about the user's own animal (OWN_PET_RE): "What bleeding disorder is inherited in Dobermans?"
# and "What emergency is common in Great Danes?" (golden Part B) are breed questions for search_knowledge,
# "my dog is bleeding" is an emergency.
OWN_PET_RE = re.compile(r"\b(my|our)\b")
# Whole words only. No "toxic"/"poisonous" ("Is chocolate toxic for my dog?", "non-toxic chew toys for my
# puppy" are questions for search_knowledge, whose answers safety_claim_check guards) and no "emergency"
# ("What emergency is common in my Great Dane?"); "my dog ate X" and the symptom check below cover those.
VET_KEYWORDS_RE = re.compile(
    r"\b(bleeding|not breathing|can[’']?t breathe|cannot breathe|seizures?|collaps(e|es|ed|ing)|unconscious|"
    r"poison(ed|ing)?|overdos(e|es|ed|ing)|vomiting blood|throwing up|blood in|hit by a (car|bike|truck)|"
    r"choking|broken (leg|bone)|swallowed)\b")

# Patterns from issue #11. They fire without "my"/"our":
#   "he ate chocolate", "she has eaten grapes": an eating verb followed by something toxic.
_TOXIC = r"(chocolate|grapes?|raisins?|xylitol|onions?|garlic|ibuprofen|paracetamol|medicine|pills?|antifreeze|rat bait)"
VET_PATTERNS = (
    re.compile(rf"\b(ate|eaten|eating|eat|chewed|licked)\b.*\b{_TOXIC}\b"),
)

# Own pet with symptoms: "my dog is sick", "our dogs keep vomiting", "my Labrador has diarrhoea".
# Checked per clause (no match across sentences), with the pet named within a few words of "my"/"our"
# ("my son is sick, which dog breed ..." is not about the dog) and the symptom after it.
_CLAUSE_RE = re.compile(r"[^.?!;\n]+")
_OWNER_RE = re.compile(r"\b(my|our)\b((?: \w+){1,5})")
_PET_WORDS_RE = re.compile(
    r"\b(dogs?|doggy|doggies|pupp(y|ies)|pups?|pets?|cats?|kittens?|labs?|labradors?|goldens?|frenchies|frenchie|"
    r"retrievers?|terriers?|spaniels?|shepherds?|hounds?|collies?|poodles?|bulldogs?|pointers?|setters?)\b")
# "car sick", "travel sick", "motion sick", "sea sick" are not symptoms to send to a vet.
_SYMPTOM_RE = re.compile(r"(?<!car )(?<!travel )(?<!motion )(?<!sea )\bsick\b|\bvomit(s|ed|ing)?\b|"
                         r"\bdiarrh(o)?ea\b|\bunconscious\b|\blethargic\b|\bthrowing up\b")


def _own_pet_has_symptoms(message: str) -> bool:
    for clause in _CLAUSE_RE.findall(message.casefold()):
        text = normalise(clause)
        for owner in _OWNER_RE.finditer(text):
            subject = owner.group(2)
            pets = [m for m in (_PET_WORDS_RE.search(subject), _breed_name_re().search(subject)) if m]
            if pets and _SYMPTOM_RE.search(text, owner.start(2) + min(m.end() for m in pets)):
                return True
    return False


def vet_check(message: str) -> bool:
    """True when the message looks like a vet or danger question (the agent must not answer it)."""
    text = message.casefold()
    if OWN_PET_RE.search(text) and VET_KEYWORDS_RE.search(text):
        return True
    return any(pattern.search(text) for pattern in VET_PATTERNS) or _own_pet_has_symptoms(message)


# --- out_of_scope_check ---------------------------------------------------------------------------
# Animals we do not cover. Only out of scope when the message is not also about dogs:
# "Is a Beagle good with cats?" is a breed question, "Where can I adopt a kitten?" is not.
OTHER_ANIMAL_RE = re.compile(
    r"\b(cats?|kittens?|kitty|parrots?|birds?|budgies?|canar(y|ies)|hamsters?|guinea pigs?|rabbits?|bunn(y|ies)|"
    r"ferrets?|horses?|ponies|pony|fish|goldfish|turtles?|tortoises?|snakes?|lizards?|reptiles?|"
    r"cheetahs?|lions?|tigers?|leopards?|bears?|wolf|wolves|foxes|fox|jackals?|hyenas?|coyotes?)\b")
# Adoption and shelters are out of scope whatever the animal (docs/milestone2.md).
ADOPTION_RE = re.compile(r"\b(adopt|adopting|adoption|shelters?)\b")
# Pet names are in scope for cats too (suggest_names takes species "cat").
NAMES_RE = re.compile(r"\bnames?\b")
DOG_WORDS_RE = re.compile(r"\b(dogs?|doggy|pupp(y|ies)|pups?|canines?|breeds?|hounds?|terriers?)\b")


@lru_cache(maxsize=1)
def _breed_name_re() -> re.Pattern[str]:
    """Every display name and alias in breeds_map (normalised), as words, with an optional plural s."""
    names = set()
    for display, entry in load_breeds_map().items():
        for name in (display, *aliases(entry)):
            normalised = normalise(name)
            if len(normalised) > 3 and not normalised.startswith("n0"):   # skip WordNet ids
                names.add(normalised)
    alternatives = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
    return re.compile(rf"\b({alternatives})s?\b")


def mentions_dogs(message: str) -> bool:
    """True when the message talks about dogs: a dog word or any breed name we know."""
    text = normalise(message)
    return bool(DOG_WORDS_RE.search(text) or _breed_name_re().search(text))


def out_of_scope_check(message: str) -> bool:
    """True for adoption questions and for questions about animals other than dogs (not name requests)."""
    text = message.casefold()
    if ADOPTION_RE.search(text):
        return True
    return bool(OTHER_ANIMAL_RE.search(text)) and not NAMES_RE.search(text) and not mentions_dogs(message)


# --- citation_check / safety_claim_check ----------------------------------------------------------
def cited_ids(answer: str) -> list[str]:
    """Chunk ids cited as [kb:...] in the answer, in order, without duplicates."""
    return list(dict.fromkeys(re.findall(CHUNK_ID_RE, answer)))


def unretrieved_ids(answer: str, retrieved_chunk_ids: Iterable[str]) -> list[str]:
    """Cited chunk ids that search_knowledge did not return in this turn (invented or from an old turn)."""
    retrieved = set(retrieved_chunk_ids)
    return [chunk_id for chunk_id in cited_ids(answer) if chunk_id not in retrieved]


def citation_check(answer: str, retrieved_chunk_ids: Iterable[str]) -> bool:
    """True if every chunk id mentioned in `answer` was actually retrieved this turn."""
    return not unretrieved_ids(answer, retrieved_chunk_ids)


# "X is safe", "grapes are not toxic", "it's harmless", "safe for dogs to eat", "won't hurt him", ...
SAFETY_CLAIM_RE = re.compile(
    r"\b(is|are|it's|its|they're|be)\s+(perfectly\s+|completely\s+|totally\s+|generally\s+|usually\s+)?"
    r"(safe|harmless|non-?toxic|not\s+(toxic|poisonous|dangerous|harmful))\b"
    r"|\bsafe\s+(for|to)\b"
    r"|\b(won't|will\s+not|doesn't|does\s+not|can't|cannot)\s+(hurt|harm)\b")
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]*")


def safety_claims(answer: str) -> list[str]:
    """Sentences of the answer that claim something is safe."""
    return [s.strip() for s in _SENTENCE_RE.findall(answer) if SAFETY_CLAIM_RE.search(s.casefold())]


def unbacked_safety_claims(answer: str, retrieved_chunk_ids: Iterable[str]) -> list[str]:
    """Safety-claim sentences without a citation of a chunk retrieved in this turn."""
    retrieved = set(retrieved_chunk_ids)
    return [claim for claim in safety_claims(answer) if not any(c in retrieved for c in cited_ids(claim))]


def safety_claim_check(answer: str, retrieved_chunk_ids: Iterable[str]) -> bool:
    """True if every safety claim ("X is safe", "X is not toxic") is backed by a retrieved source.
    An answer without safety claims passes."""
    return not unbacked_safety_claims(answer, retrieved_chunk_ids)
