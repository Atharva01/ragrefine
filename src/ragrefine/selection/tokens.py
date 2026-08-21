"""Framework-independent token-counting contracts."""

from typing import Protocol


class TokenCounter(Protocol):
    """Count text tokens for a caller-selected model or deployment target."""

    def count(self, text: str) -> int:
        """Return the non-negative token count for one unmodified candidate text."""
