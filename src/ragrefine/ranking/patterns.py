"""Deterministic ranking from exact structured-pattern agreement."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from ragrefine.models import Candidate
from ragrefine.query.patterns import PatternMatch, PatternRegistry


class PatternAgreement(StrEnum):
    """The comparison outcome for one query pattern constraint."""

    EXACT = "exact_agreement"
    CONFLICT = "conflicting_value"
    ABSENT = "constraint_absent"


@dataclass(frozen=True, slots=True)
class PatternConstraintEvidence:
    """Candidate evidence for one structured query constraint."""

    query_match: PatternMatch
    agreement: PatternAgreement
    candidate_matches: tuple[PatternMatch, ...]


@dataclass(frozen=True, slots=True)
class PatternRanking:
    """An auditable, signal-specific ranking for one original candidate."""

    candidate: Candidate
    rank: int
    evidence: tuple[PatternConstraintEvidence, ...]
    has_query_constraints: bool

    @property
    def exact_agreements(self) -> int:
        """Return the number of query constraints with exact candidate support."""
        return sum(item.agreement is PatternAgreement.EXACT for item in self.evidence)

    @property
    def conflicts(self) -> int:
        """Return the number of query constraints contradicted by the candidate."""
        return sum(
            item.agreement is PatternAgreement.CONFLICT for item in self.evidence
        )

    @property
    def absent_constraints(self) -> int:
        """Return the number of query constraints absent from the candidate."""
        return sum(item.agreement is PatternAgreement.ABSENT for item in self.evidence)


class PatternRanker:
    """Rank candidates by agreement with exact, configured query constraints.

    It only ranks the supplied pool.  Conflicts and absent constraints remain
    trace evidence; no candidate is filtered or modified.
    """

    def __init__(self, registry: PatternRegistry) -> None:
        self._registry = registry

    def rank(
        self, query: str, candidates: Sequence[Candidate]
    ) -> tuple[PatternRanking, ...]:
        """Return a stable ranking and constraint evidence for every candidate."""
        query_matches = self._registry.extract(query)
        unranked = tuple(
            self._score(candidate, query_matches, input_order)
            for input_order, candidate in enumerate(candidates)
        )
        ordered = sorted(
            unranked,
            key=lambda item: (
                -item.exact_agreements,
                item.conflicts,
                item.absent_constraints,
                _retrieval_tie_break(item.candidate.retrieval_rank),
                item.candidate.id,
                item.input_order,
            ),
        )
        return tuple(
            PatternRanking(
                candidate=item.candidate,
                rank=rank,
                evidence=item.evidence,
                has_query_constraints=bool(query_matches),
            )
            for rank, item in enumerate(ordered, start=1)
        )

    def _score(
        self,
        candidate: Candidate,
        query_matches: tuple[PatternMatch, ...],
        input_order: int,
    ) -> "_UnrankedPatternResult":
        candidate_matches = self._registry.extract(candidate.text)
        evidence = tuple(
            _constraint_evidence(query_match, candidate_matches)
            for query_match in query_matches
        )
        return _UnrankedPatternResult(
            candidate=candidate,
            evidence=evidence,
            input_order=input_order,
        )


@dataclass(frozen=True, slots=True)
class _UnrankedPatternResult:
    candidate: Candidate
    evidence: tuple[PatternConstraintEvidence, ...]
    input_order: int

    @property
    def exact_agreements(self) -> int:
        return sum(item.agreement is PatternAgreement.EXACT for item in self.evidence)

    @property
    def conflicts(self) -> int:
        return sum(
            item.agreement is PatternAgreement.CONFLICT for item in self.evidence
        )

    @property
    def absent_constraints(self) -> int:
        return sum(item.agreement is PatternAgreement.ABSENT for item in self.evidence)


def _constraint_evidence(
    query_match: PatternMatch, candidate_matches: tuple[PatternMatch, ...]
) -> PatternConstraintEvidence:
    same_label = tuple(
        match for match in candidate_matches if match.label == query_match.label
    )
    if any(
        match.normalized_value == query_match.normalized_value for match in same_label
    ):
        agreement = PatternAgreement.EXACT
    elif same_label:
        agreement = PatternAgreement.CONFLICT
    else:
        agreement = PatternAgreement.ABSENT
    return PatternConstraintEvidence(query_match, agreement, same_label)


def _retrieval_tie_break(retrieval_rank: int | None) -> tuple[int, int]:
    """Place known original ranks before absent ranks without sentinels."""
    return (1, 0) if retrieval_rank is None else (0, retrieval_rank)
