"""Capture text tokens from PersonaPlex during voice bridge for memory extraction."""

from __future__ import annotations


class TranscriptCapture:
    """Accumulates text tokens from PersonaPlex 0x02 messages into a transcript."""

    def __init__(self) -> None:
        self._tokens: list[str] = []
        self._assistant_text: list[str] = []
        self._turns: list[dict[str, str]] = []

    def add_token(self, text: str) -> None:
        """Add a text token from a 0x02 message."""
        self._tokens.append(text)
        self._assistant_text.append(text)

    def end_turn(self) -> None:
        """Mark the end of an assistant turn."""
        if self._assistant_text:
            full_text = "".join(self._assistant_text).strip()
            if full_text:
                self._turns.append({"role": "assistant", "content": full_text})
            self._assistant_text = []

    def add_user_note(self, note: str) -> None:
        """Add a user-side note (e.g., from speech-to-text if available)."""
        if note.strip():
            self._turns.append({"role": "user", "content": note.strip()})

    def get_transcript(self) -> list[dict[str, str]]:
        """Return the accumulated transcript as a list of {role, content} dicts."""
        # Flush any remaining assistant text
        self.end_turn()
        return list(self._turns)

    def get_full_text(self) -> str:
        """Return all captured assistant text as a single string."""
        return "".join(self._tokens)
