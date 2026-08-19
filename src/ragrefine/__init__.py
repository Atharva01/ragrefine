"""Retriever-agnostic post-retrieval refinement for RAG candidate sets."""

from ragrefine.context import (
    ContextSelectionResult,
    ContextSelector,
    RankPreservingContextSelector,
    SelectionRecord,
    TokenCounter,
)
from ragrefine.filtering import (
    CandidateDeduplicator,
    DeduplicationConfig,
    DeduplicationResult,
    SuppressedCandidate,
)
from ragrefine.models import Candidate, CandidateSet, RankingSignal, RefinedCandidate
from ragrefine.orchestration import ChannelConfig, RefinerConfig
from ragrefine.refiner import Refiner
from ragrefine.results import RefinementResult
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
