"""Core retriever-agnostic candidate domain models."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Candidate:
    """One immutable evidence unit supplied by an external retriever.

    The model preserves the retriever's identifier, text, metadata, and optional
    retrieval annotations without depending on any retrieval framework.
    """

    id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    retrieval_score: float | None = None
    retrieval_rank: int | None = None


@dataclass(frozen=True, slots=True)
class CandidateSet:
    """An immutable, ordered candidate collection from one named retriever."""

    name: str
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True, slots=True)
class RankingSignal:
    """One optional, traceable ranking observation for a candidate."""

    rank: int | None = None
    score: float | None = None
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RefinedCandidate:
    """A candidate with its final ranking position and named evidence.

    The wrapped candidate is always supplied by the caller; refinement never
    creates or rewrites evidence. Retrieval provenance remains on ``candidate``.
    ``rank`` is the position after ranking; a later context-selection position
    is recorded separately in :class:`ragrefine.selection.SelectionRecord`.
    """

    candidate: Candidate
    rank: int
    signals: Mapping[str, RankingSignal] = field(default_factory=dict)
