"""Typed package exceptions for refinement failures."""


class RagRefineError(Exception):
    """Base exception for explicit ragrefine failures."""


class RerankerError(RagRefineError):
    """Raised when neural reranking cannot complete safely."""
