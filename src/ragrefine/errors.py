"""Typed package exceptions for refinement failures."""


class RagRefineError(Exception):
    """Base exception for explicit ragrefine failures."""


class RerankerError(RagRefineError):
    """Raised when neural reranking cannot complete safely."""


class PatternError(RagRefineError):
    """Raised when configured pattern processing is invalid or unsafe."""


class ChannelExecutionError(RagRefineError):
    """Raised when a required ranking channel cannot complete."""


class MultipleActiveChannelsError(RagRefineError):
    """Raised when independent rankings need fusion but it is disabled."""
