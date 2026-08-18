"""Framework-independent neural reranking contracts."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ragrefine.models import Candidate


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """One input candidate with its raw neural score and reranker rank."""

    candidate: Candidate
    score: float
    rank: int
    model: str
    model_revision: str | None
    backend: str


class NeuralReranker(Protocol):
    """Rank an already-retrieved candidate pool with a neural backend."""

    name: str

    def rank(
        self, query: str, candidates: Sequence[Candidate]
    ) -> Sequence[ScoredCandidate]:
        """Return every supplied candidate exactly once in reranked order."""
