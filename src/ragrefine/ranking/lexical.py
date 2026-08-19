"""Deterministic lexical-coverage ranking."""

from collections.abc import Sequence
from dataclasses import dataclass

from ragrefine.models import Candidate
from ragrefine.query.normalize import normalized_terms


@dataclass(frozen=True, slots=True)
class LexicalRanking:
    """One candidate's auditable lexical-ranking result.

    ``candidate`` is the exact input instance; this signal never rewrites
    evidence or retrieval annotations.  ``matched_terms`` and
    ``missing_terms`` preserve normalized query-term order for stable traces.
    """

    candidate: Candidate
    rank: int
    coverage: float
    matched_terms: tuple[str, ...]
    missing_terms: tuple[str, ...]


class LexicalRanker:
    """Rank existing candidates by normalized unique query-term coverage.

    This is a separate ranking signal, not BM25, retrieval, or score fusion.
    Equal coverage uses original retrieval rank and then candidate identifier,
    matching the project's deterministic ranking contract.
    """

    def rank(
        self, query: str, candidates: Sequence[Candidate]
    ) -> tuple[LexicalRanking, ...]:
        """Return a complete, stable lexical ranking for ``candidates``."""
        query_terms = normalized_terms(query)
        term_count = len(query_terms)
        unranked = tuple(
            self._score(candidate, query_terms, term_count, input_order)
            for input_order, candidate in enumerate(candidates)
        )
        ordered = sorted(
            unranked,
            key=lambda item: (
                -item.coverage,
                _retrieval_tie_break(item.candidate.retrieval_rank),
                item.candidate.id,
                item.input_order,
            ),
        )
        return tuple(
            LexicalRanking(
                candidate=item.candidate,
                rank=rank,
                coverage=item.coverage,
                matched_terms=item.matched_terms,
                missing_terms=item.missing_terms,
            )
            for rank, item in enumerate(ordered, start=1)
        )

    @staticmethod
    def _score(
        candidate: Candidate,
        query_terms: tuple[str, ...],
        term_count: int,
        input_order: int,
    ) -> "_UnrankedLexicalResult":
        candidate_terms = set(normalized_terms(candidate.text))
        matched_terms = tuple(term for term in query_terms if term in candidate_terms)
        missing_terms = tuple(
            term for term in query_terms if term not in candidate_terms
        )
        coverage = len(matched_terms) / term_count if term_count else 0.0
        return _UnrankedLexicalResult(
            candidate=candidate,
            coverage=coverage,
            matched_terms=matched_terms,
            missing_terms=missing_terms,
            input_order=input_order,
        )


@dataclass(frozen=True, slots=True)
class _UnrankedLexicalResult:
    candidate: Candidate
    coverage: float
    matched_terms: tuple[str, ...]
    missing_terms: tuple[str, ...]
    input_order: int


def _retrieval_tie_break(retrieval_rank: int | None) -> tuple[int, int]:
    """Place known original ranks before absent ranks without sentinels."""
    return (1, 0) if retrieval_rank is None else (0, retrieval_rank)
