"""Property tests for core candidate-model invariants."""

from hypothesis import given
from hypothesis import strategies as st

from ragrefine.models import Candidate, CandidateSet


@given(st.dictionaries(st.text(), st.text()))
def test_candidate_does_not_mutate_source_metadata(metadata: dict[str, str]) -> None:
    """Constructing a Candidate does not modify caller-owned metadata."""
    original_metadata = dict(metadata)

    candidate = Candidate(id="chunk", text="Evidence", metadata=metadata)

    assert metadata == original_metadata
    assert candidate.metadata == original_metadata


@given(st.lists(st.tuples(st.text(), st.text()), max_size=10))
def test_candidate_set_preserves_deterministic_order(
    candidate_data: list[tuple[str, str]],
) -> None:
    """CandidateSet retains the source tuple exactly as supplied."""
    candidates = tuple(
        Candidate(id=candidate_id, text=text) for candidate_id, text in candidate_data
    )

    candidate_set = CandidateSet(name="retriever", candidates=candidates)

    assert candidate_set.candidates == candidates
    assert tuple(candidate.id for candidate in candidate_set.candidates) == tuple(
        candidate_id for candidate_id, _ in candidate_data
    )
