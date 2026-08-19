"""Candidate-filtering package."""

from ragrefine.filtering.deduplicate import (
    CandidateDeduplicator,
    DeduplicationConfig,
    DeduplicationResult,
    SuppressedCandidate,
)

__all__ = [
    "CandidateDeduplicator",
    "DeduplicationConfig",
    "DeduplicationResult",
    "SuppressedCandidate",
]
