"""Property tests for no-op refinement invariants."""

from hypothesis import given
from hypothesis import strategies as st

from ragrefine.models import Candidate, CandidateSet
from ragrefine.refiner import Refiner


@given(
    candidate_data=st.lists(
        st.tuples(st.text(min_size=1), st.text()),
        max_size=12,
        unique_by=lambda item: item[0],
    ),
    top_k=st.integers(min_value=0, max_value=20),
)
def test_no_op_output_is_deterministic_subset_with_contiguous_ranks(
    candidate_data: list[tuple[str, str]], top_k: int
) -> None:
    """No-op refinement preserves candidate identity, text, order, and limits."""
    candidates = tuple(
        Candidate(id=candidate_id, text=text, retrieval_rank=index)
        for index, (candidate_id, text) in enumerate(candidate_data, start=1)
    )
    candidate_set = CandidateSet(name="dense", candidates=candidates)

    first = Refiner().refine("same query", (candidate_set,), top_k=top_k)
    second = Refiner().refine("same query", (candidate_set,), top_k=top_k)
    first_candidates = tuple(item.candidate for item in first.candidates)
    second_candidates = tuple(item.candidate for item in second.candidates)

    assert first_candidates == second_candidates
    assert first_candidates == candidates[:top_k]
    assert {candidate.id for candidate in first_candidates}.issubset(
        {candidate.id for candidate in candidates}
    )
    assert tuple(item.final_rank for item in first.candidates) == tuple(
        range(1, len(first.candidates) + 1)
    )
