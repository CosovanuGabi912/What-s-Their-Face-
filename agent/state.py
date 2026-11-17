"""Conversation state kept in code next to the chat history (issue #7).

Small models forget or miscopy details from earlier turns, so the agent loop stores them here and
adds to_prompt() to the system prompt on every turn.
"""

from dataclasses import dataclass, field


@dataclass
class ConversationState:
    current_breed: str | None = None
    breed_confidence: float | None = None
    questionnaire: dict = field(default_factory=dict)
    chosen_breed: str | None = None
    pet_description: str | None = None
    rejected_names: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """The known fields, one per line, for the {state} slot of the system prompt."""
        lines = []
        if self.current_breed and self.breed_confidence is not None:
            lines.append(f"current_breed: {self.current_breed} (confidence {self.breed_confidence:.2f})")
        elif self.current_breed:
            lines.append(f"current_breed: {self.current_breed}")

        if self.questionnaire:
            answers = ", ".join(f"{key}={value}" for key, value in self.questionnaire.items())
            lines.append(f"questionnaire: {answers}")

        if self.chosen_breed:
            lines.append(f"chosen_breed: {self.chosen_breed}")

        if self.pet_description:
            lines.append(f"pet_description: {self.pet_description}")

        if self.rejected_names:
            lines.append("rejected_names: " + ", ".join(self.rejected_names))

        return "\n".join(lines) if lines else "Nothing known yet."
