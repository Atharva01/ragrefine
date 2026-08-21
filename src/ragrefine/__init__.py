"""Retriever-agnostic post-retrieval refinement for RAG candidate sets."""

from ragrefine.config import ChannelConfig, RefinerConfig
from ragrefine.models import Candidate, CandidateSet, RankingSignal, RefinedCandidate
from ragrefine.refiner import Refiner
from ragrefine.results import RefinementResult
from ragrefine.selection import (
    CandidateDeduplicator,
    ContextSelectionResult,
    ContextSelector,
    DeduplicationConfig,
    DeduplicationResult,
    RankPreservingContextSelector,
    SelectionRecord,
    SuppressedCandidate,
    TokenCounter,
)
from ragrefine.tracing.models import RefinementTrace, StageTrace

__all__ = [
    "Candidate",
    "CandidateDeduplicator",
    "CandidateSet",
    "ChannelConfig",
    "ContextSelectionResult",
    "ContextSelector",
    "DeduplicationConfig",
    "DeduplicationResult",
    "RefinedCandidate",
    "RefinerConfig",
    "RefinementResult",
    "RefinementTrace",
    "Refiner",
    "RankingSignal",
    "RankPreservingContextSelector",
    "StageTrace",
    "SuppressedCandidate",
    "SelectionRecord",
    "TokenCounter",
]
