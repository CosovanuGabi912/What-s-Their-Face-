"""Guards that run in code, not in the prompt (issues #7 and #11; citation check in #16).

- vet_check runs BEFORE the SLM: when it fires, the agent replies with VET_REPLY and calls no tools.
  The SLM cannot overrule it (agent decision 6 in docs/milestone2.md).
- citation_check runs on every final answer: each cited [chunk_id] must have been returned by
  search_knowledge in this turn. Stub for now: always passes.
"""

from __future__ import annotations

import re

VET_REPLY = ("This sounds like it could be a health emergency, and I can't give medical advice. "
             "Please contact your vet or an emergency animal clinic right away.")

# Phrases that mean "see a vet", matched case-insensitively anywhere in the message, but only when the
# message is about the user's own animal (OWN_PET_RE): "What bleeding disorder is inherited in Dobermans?"
# and "What emergency is common in Great Danes?" (golden Part B) are breed questions for search_knowledge,
# "my dog is bleeding" is an emergency.
OWN_PET_RE = re.compile(r"\b(my|our)\b")
VET_KEYWORDS = (
    "bleeding", "not breathing", "can't breathe", "cannot breathe", "seizure", "collapse",
    "unconscious", "poison", "toxic", "overdose", "emergency", "vomiting blood", "throwing up",
    "blood in", "hit by a car", "choking", "broken leg", "swallowed",
)

# Patterns from issue #11. They fire without "my"/"our":
#   "he ate chocolate", "she has eaten grapes": an eating verb followed by something toxic;
#   "my dog is sick", "our puppy has diarrhoea": an own pet with symptoms.
_TOXIC = r"(chocolate|grapes?|raisins?|xylitol|onions?|garlic|ibuprofen|paracetamol|medicine|pills?|antifreeze|rat bait)"
_PET = r"(dog|puppy|pup|pet|cat|kitten)"
_SYMPTOM = r"(sick|vomiting|diarrh(o)?ea|unconscious)"
VET_PATTERNS = (
    re.compile(rf"\b(ate|eaten|eating|eat|chewed|licked)\b.*\b{_TOXIC}\b"),
    re.compile(rf"\b(my|our)\b.*\b{_PET}\b.*\b{_SYMPTOM}\b"),
)


def vet_check(message: str) -> bool:
    """True when the message looks like a vet or danger question (the agent must not answer it)."""
    text = message.casefold()
    if OWN_PET_RE.search(text) and any(keyword in text for keyword in VET_KEYWORDS):
        return True
    return any(pattern.search(text) for pattern in VET_PATTERNS)


def citation_check(answer: str, retrieved_ids: set[str] | list[str]) -> bool:
    """True when every [chunk_id] cited in `answer` is in `retrieved_ids`.

    Stub until #16: always True. The real version will find citations with
    shared.schemas.CHUNK_ID_RE and also flag "X is safe" claims made without any citation.
    """
    return True
