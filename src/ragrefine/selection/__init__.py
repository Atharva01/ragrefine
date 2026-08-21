"""Deterministic duplicate suppression and final context selection."""

from ragrefine.selection.deduplicate import (
    CandidateDeduplicator,
    DeduplicationConfig,
    DeduplicationResult,
    SuppressedCandidate,
)
from ragrefine.selection.selector import (
    ContextSelectionResult,
    ContextSelector,
    RankPreservingContextSelector,
    SelectionRecord,
)
from ragrefine.selection.tokens import TokenCounter

__all__ = [
    "CandidateDeduplicator",
    "ContextSelectionResult",
    "ContextSelector",
    "DeduplicationConfig",
    "DeduplicationResult",
    "RankPreservingContextSelector",
    "SelectionRecord",
    "SuppressedCandidate",
    "TokenCounter",
]
