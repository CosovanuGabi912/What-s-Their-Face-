"""Conversation state kept in code next to the chat history (issues #7 and #16).

Small models forget or miscopy details from earlier turns, so the agent loop stores them here and
adds to_prompt() to the system prompt on every turn.

Questionnaire answers come from two places: extract_answers() reads the obvious ones from the user's
own words ("I live in a flat", "I'm allergic"), and a successful recommend_breeds call stores all six.
The prompt lists the answers still missing, so the SLM only asks for those (agent decision 5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

QUESTIONNAIRE_FIELDS = ("home", "hours_alone", "daily_walks_min", "kids_at_home", "has_allergies", "experience")


@dataclass
class ConversationState:
    current_breed: str | None = None
    breed_confidence: float | None = None
    questionnaire: dict = field(default_factory=dict)
    chosen_breed: str | None = None
    pet_description: str | None = None
    rejected_names: list[str] = field(default_factory=list)
    talked_about_dogs: bool = False   # an earlier user message named a dog or breed (out_of_scope_check)

    def about_a_dog(self) -> bool:
        """The conversation is already about a dog: follow-ups about other animals are about that dog."""
        return self.talked_about_dogs or bool(self.current_breed or self.chosen_breed)

    def missing_answers(self) -> list[str]:
        return [name for name in QUESTIONNAIRE_FIELDS if name not in self.questionnaire]

    def to_prompt(self) -> str:
        """The known fields, one per line, for the {state} slot of the system prompt."""
        lines = []
        if self.current_breed and self.breed_confidence is not None:
            lines.append(f"current_breed: {self.current_breed} (confidence {self.breed_confidence:.2f})")
        elif self.current_breed:
            lines.append(f"current_breed: {self.current_breed}")

        if self.questionnaire:
            answers = ", ".join(f"{key}={self.questionnaire[key]}"
                                for key in QUESTIONNAIRE_FIELDS if key in self.questionnaire)
            lines.append(f"questionnaire: {answers}")
            if self.missing_answers():
                lines.append("questionnaire still to ask: " + ", ".join(self.missing_answers()))

        if self.chosen_breed:
            lines.append(f"chosen_breed: {self.chosen_breed}")

        if self.pet_description:
            lines.append(f"pet_description: {self.pet_description}")

        if self.rejected_names:
            lines.append("rejected_names: " + ", ".join(self.rejected_names))

        return "\n".join(lines) if lines else "Nothing known yet."


# --- Questionnaire answers from the user's words ---------------------------------------------------
# Conservative on purpose: only sentences in the first person are read ("I live in a flat"), never
# questions about a breed ("Is a Husky OK in a flat?", "Does it need two hours of walks?").
# Anything unclear is left for the SLM to ask.
_FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'd|me|my|we|we're|we've|us|our)\b")
_SENTENCE_RE = re.compile(r"[^.?!]+")
_CLAUSE_SPLIT = re.compile(r"[;,]|\bbut\b|\band\b")
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
_NUM = r"(?P<n>\d+(?:\.\d+)?|" + "|".join(_NUMBER_WORDS) + ")"
_HOURS = r"\s*(?:h|hrs?|hours?)\b"


def _number(text: str) -> float:
    return float(_NUMBER_WORDS.get(text, text))


def _home(text: str) -> str | None:
    if not re.search(r"\b(live|living|stay|rent|own|have|my|our)\b", text):
        return None
    if re.search(r"\b(flat|apartment|studio|condo)\b", text):
        return "flat"
    if re.search(r"\bhouse\b", text):
        return "house_with_garden" if re.search(r"\b(garden|yard|backyard)\b", text) else "house"
    return None


def _hours_alone(text: str) -> int | None:
    if not re.search(r"\b(alone|work|working|office|out)\b", text) or re.search(r"\bwalk", text):
        return None
    match = re.search(rf"\b{_NUM}{_HOURS}", text)
    if not match:
        return None
    hours = round(_number(match.group("n")))
    return hours if 0 <= hours <= 24 else None


def _walks(text: str) -> int | None:
    if not re.search(r"\bwalk", text):
        return None
    if re.search(r"\bhalf an hour\b", text):
        return 30
    match = re.search(rf"\b{_NUM}\s*(?:min|mins|minutes)\b", text)
    if match:
        return round(_number(match.group("n")))
    match = re.search(rf"\b{_NUM}{_HOURS}", text)
    if match:
        return round(_number(match.group("n")) * 60)
    if re.search(r"\b(an|one) hour\b", text):
        return 60
    return None


def _kids(text: str) -> bool | None:
    if re.search(r"\b(no|don't have|do not have|without|haven't got)\s+(any\s+)?(kids|children)\b", text):
        return False
    if re.search(r"\b(have|got|with)\b[^.?!]*\b(kids|children|a baby|a toddler|a son|a daughter)\b"
                 r"|\b(my|our)\s+(kids|children|son|daughter|baby|toddler)\b", text):
        return True
    return None


def _allergies(text: str) -> bool | None:
    if not re.search(r"\ballerg", text):
        return None
    if re.search(r"\b(no|not|don't|do not|without|aren't|isn't|nobody|none)\b", text):
        return False
    return True


def _experience(text: str) -> str | None:
    if re.search(r"\b(first dog|first-time|first time|never had a dog|never owned a dog|no experience)\b", text):
        return "none"
    if re.search(r"\b(experienced|had dogs all my life|grew up with dogs|lots of experience)\b", text):
        return "experienced"
    if re.search(r"\b(had a dog before|had dogs before|some experience|owned a dog before)\b", text):
        return "some"
    return None


_EXTRACTORS = {"home": _home, "hours_alone": _hours_alone, "daily_walks_min": _walks,
               "kids_at_home": _kids, "has_allergies": _allergies, "experience": _experience}


def extract_answers(message: str) -> dict:
    """Questionnaire answers stated plainly in a user message (only the ones found).
    Each clause of a first-person sentence is read on its own; the first clause that answers a field wins."""
    found: dict = {}
    for sentence in _SENTENCE_RE.findall(message.casefold().replace("\u2019", "'")):
        if not _FIRST_PERSON.search(sentence):
            continue
        for clause in _CLAUSE_SPLIT.split(sentence):
            for name, extract in _EXTRACTORS.items():
                if name not in found:
                    value = extract(clause)
                    if value is not None:
                        found[name] = value
    return found
