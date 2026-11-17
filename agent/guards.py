"""Guards that run in code, not in the prompt (issue #7; real logic in #16).

- vet_check runs BEFORE the SLM: when it fires, the agent replies with VET_REPLY and calls no tools.
  The SLM cannot overrule it (agent decision 6 in docs/milestone2.md).
- citation_check runs on every final answer: each cited [chunk_id] must have been returned by
  search_knowledge in this turn. Stub for now: always passes.
"""

from __future__ import annotations

import re

VET_REPLY = ("This sounds like it could be a health emergency, and I can't give medical advice. "
             "Please contact your vet or an emergency animal clinic right away.")

# Phrases that always mean "see a vet", matched case-insensitively anywhere in the message.
VET_KEYWORDS = (
    "bleeding", "not breathing", "can't breathe", "cannot breathe", "seizure", "collapsed",
    "unconscious", "poison", "vomiting blood", "blood in", "hit by a car", "choking",
    "broken leg", "swallowed",
)

# "my dog ate chocolate", "she has eaten grapes", ...: an eating verb followed by something toxic.
_TOXIC = r"(chocolate|grapes?|raisins?|xylitol|onions?|garlic|ibuprofen|paracetamol|medicine|pills?|antifreeze|rat bait)"
VET_PATTERNS = (
    re.compile(rf"\b(ate|eaten|eating|eat|chewed|licked)\b.*\b{_TOXIC}\b"),
)


def vet_check(message: str) -> bool:
    """True when the message looks like a vet or danger question (the agent must not answer it)."""
    text = message.casefold()
    if any(keyword in text for keyword in VET_KEYWORDS):
        return True
    return any(pattern.search(text) for pattern in VET_PATTERNS)


def citation_check(answer: str, retrieved_ids: set[str] | list[str]) -> bool:
    """True when every [chunk_id] cited in `answer` is in `retrieved_ids`.

    Stub until #16: always True. The real version will find citations with
    shared.schemas.CHUNK_ID_RE and also flag "X is safe" claims made without any citation.
    """
    return True
