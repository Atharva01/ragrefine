"""Property tests for structured-pattern ranking invariants."""

from hypothesis import given
from hypothesis import strategies as st

from ragrefine.models import Candidate
from ragrefine.query.patterns import PatternRegistry, PatternRule
from ragrefine.ranking.patterns import PatternRanker

_REGISTRY = PatternRegistry((PatternRule("number", r"\b\d+\b"),))


@given(st.text())
def test_pattern_extraction_is_deterministic_and_spans_source_text(text: str) -> None:
    """Every reported match is reproducible and points to its original source."""
    matches = _REGISTRY.extract(text)
    assert matches == _REGISTRY.extract(text)
    assert all(text[match.start : match.end] == match.value for match in matches)


@given(
    query=st.text(),
    candidate_data=st.lists(
        st.tuples(
            st.text(min_size=1), st.text(), st.integers(min_value=1, max_value=20)
        ),
        max_size=12,
        unique_by=lambda item: item[0],
    ),
)
def test_pattern_ranking_is_deterministic_and_preserves_candidates(
    query: str, candidate_data: list[tuple[str, str, int]]
) -> None:
    """Pattern output is a complete ordered view over the original objects."""
    candidates = tuple(
        Candidate(id=identifier, text=text, retrieval_rank=retrieval_rank)
        for identifier, text, retrieval_rank in candidate_data
    )

    first = PatternRanker(_REGISTRY).rank(query, candidates)
    second = PatternRanker(_REGISTRY).rank(query, candidates)

    assert first == second
    assert {id(item.candidate) for item in first} == {
        id(candidate) for candidate in candidates
    }
    assert tuple(item.rank for item in first) == tuple(range(1, len(candidates) + 1))
