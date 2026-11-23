"""Guards that run in code, not in the prompt (issues #7 and #11; citation check in #16).

- vet_check runs BEFORE the SLM: when it fires, the agent replies with VET_REPLY and calls no tools.
  The SLM cannot overrule it (agent decision 6 in docs/milestone2.md).
- citation_check runs on every final answer: each cited [chunk_id] must have been returned by
  search_knowledge in this turn. Stub for now: always passes.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from shared.breeds import aliases, load_breeds_map, normalise

logger = logging.getLogger(__name__)

VET_REPLY = ("This sounds like it could be a health emergency, and I can't give medical advice. "
             "Please contact your vet or an emergency animal clinic right away.")

# Phrases that mean "see a vet", matched case-insensitively anywhere in the message, but only when the
# message is about the user's own animal (OWN_PET_RE): "What bleeding disorder is inherited in Dobermans?"
# and "What emergency is common in Great Danes?" (golden Part B) are breed questions for search_knowledge,
# "my dog is bleeding" is an emergency.
OWN_PET_RE = re.compile(r"\b(my|our)\b")
# Whole words only. No "toxic"/"poisonous" ("Is chocolate toxic for my dog?", "non-toxic chew toys for my
# puppy" are questions for search_knowledge, answered with citations) and no "emergency"
# ("What emergency is common in my Great Dane?"); "my dog ate X" and the symptom check below cover those.
VET_KEYWORDS_RE = re.compile(
    r"\b(bleeding|not breathing|can[’']?t breathe|cannot breathe|seizures?|collaps(e|es|ed|ing)|unconscious|"
    r"poison(ed|ing)?|overdos(e|es|ed|ing)|vomiting blood|throwing up|thr(ew|ows|own) up|blood in|hit by a (car|bike|truck)|"
    r"choking|broken (legs?|bones?)|swallowed)\b")

# Patterns from issue #11. They fire without "my"/"our":
#   "he ate chocolate", "she has eaten grapes": an eating verb followed by something toxic.
_TOXIC = r"(chocolate|grapes?|raisins?|xylitol|onions?|garlic|ibuprofen|paracetamol|medicine|pills?|antifreeze|rat bait)"
VET_PATTERNS = (
    re.compile(rf"\b(ate|eaten|eating|eat|chewed|licked)\b.*\b{_TOXIC}\b"),
)

# Own pet with symptoms: "my dog is sick", "our dogs keep vomiting", "my Labrador has diarrhoea".
# Checked per clause (no match across sentences), with the pet named within 8 words of "my"/"our"
# ("my son is sick, which dog breed ..." is not about the dog) and the symptom after it.
_CLAUSE_RE = re.compile(r"[^.?!;\n]+")
_OWNER_RE = re.compile(r"\b(my|our)\b((?: \w+){1,8})")   # "my lovely 3 year old female lab"
_PET_WORDS_RE = re.compile(
    r"\b(dogs?|doggy|doggies|pupp(y|ies)|pups?|pets?|cats?|kittens?|labs?|labradors?|goldens?|frenchies|frenchie|"
    r"retrievers?|terriers?|spaniels?|shepherds?|hounds?|collies?|poodles?|bulldogs?|pointers?|setters?)\b")
# "car sick", "travel sick", "motion sick", "sea sick" are not symptoms to send to a vet.
_SYMPTOM_RE = re.compile(r"(?<!car )(?<!travel )(?<!motion )(?<!sea )\bsick\b|\bvomit(s|ed|ing)?\b|"
                         r"\bdiarrh(o)?ea\b|\bunconscious\b|\blethargic\b|\bthr(ow|owing|ew|ows|own) up\b")


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


def _breed_names() -> re.Pattern[str] | None:
    """_breed_name_re(), or None when breeds_map cannot be loaded: the vet guard then falls back to the
    plain pet words instead of crashing the turn. A failure is not cached, so a fixed file is picked up."""
    try:
        return _breed_name_re()
    except Exception:  # noqa: BLE001  (missing or malformed breeds_map.json)
        logger.warning("vet_check: could not load breed names; matching pet words only", exc_info=True)
        return None


def _own_pet_has_symptoms(message: str) -> bool:
    breed_re = _breed_names()
    for clause in _CLAUSE_RE.findall(message.casefold()):
        text = normalise(clause)
        for owner in _OWNER_RE.finditer(text):
            subject = owner.group(2)
            pets = [m for m in (_PET_WORDS_RE.search(subject), breed_re and breed_re.search(subject)) if m]
            if pets and _SYMPTOM_RE.search(text, owner.start(2) + min(m.end() for m in pets)):
                return True
    return False


def vet_check(message: str) -> bool:
    """True when the message looks like a vet or danger question (the agent must not answer it)."""
    text = message.casefold()
    if OWN_PET_RE.search(text) and VET_KEYWORDS_RE.search(text):
        return True
    return any(pattern.search(text) for pattern in VET_PATTERNS) or _own_pet_has_symptoms(message)


def citation_check(answer: str, retrieved_ids: set[str] | list[str]) -> bool:
    """True when every [chunk_id] cited in `answer` is in `retrieved_ids`.

    Stub until #16: always True. The real version will find citations with
    shared.schemas.CHUNK_ID_RE and also flag "X is safe" claims made without any citation.
    """
    return True
