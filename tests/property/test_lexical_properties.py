"""Property tests for lexical ranking invariants."""

from hypothesis import given
from hypothesis import strategies as st

from ragrefine.models import Candidate
from ragrefine.query.normalize import normalize_text, normalized_terms
from ragrefine.ranking.lexical import LexicalRanker


@given(st.text())
def test_normalization_is_idempotent(text: str) -> None:
    """Normalizing an already normalized value does not change it."""
    assert normalize_text(normalize_text(text)) == normalize_text(text)


@given(st.text())
def test_normalized_terms_are_unique_and_stable(text: str) -> None:
    """Term extraction remains deterministic and has no repeated coverage terms."""
    first = normalized_terms(text)
    assert first == normalized_terms(text)
    assert len(first) == len(set(first))


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
def test_lexical_ranking_is_deterministic_and_preserves_candidates(
    query: str, candidate_data: list[tuple[str, str, int]]
) -> None:
    """Ranking outputs are a complete ordered view over the original objects."""
    candidates = tuple(
        Candidate(id=identifier, text=text, retrieval_rank=retrieval_rank)
        for identifier, text, retrieval_rank in candidate_data
    )

    first = LexicalRanker().rank(query, candidates)
    second = LexicalRanker().rank(query, candidates)

    assert first == second
    assert tuple(item.candidate for item in first) == tuple(
        item.candidate for item in second
    )
    assert {id(item.candidate) for item in first} == {
        id(candidate) for candidate in candidates
    }
    assert tuple(item.rank for item in first) == tuple(range(1, len(candidates) + 1))
    assert all(0.0 <= item.coverage <= 1.0 for item in first)
