"""Typed contracts for framework-independent rank fusion."""

from dataclasses import dataclass

from ragrefine.models import Candidate


@dataclass(frozen=True, slots=True)
class RankingChannel:
    """One named ordered ranking, with an explicit non-negative weight."""

    name: str
    candidates: tuple[Candidate, ...]
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    """One preserved candidate and its explainable rank-based RRF result."""

    candidate: Candidate
    rank: int
    score: float
    channel_ranks: dict[str, int]
    contributions: dict[str, float]
