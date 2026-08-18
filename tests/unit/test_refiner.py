"""Unit tests for the no-op refinement baseline."""

import pytest

from ragrefine.models import Candidate, CandidateSet
from ragrefine.refiner import Refiner


def test_refiner_preserves_input_order_and_evidence() -> None:
    """The no-op pipeline returns caller-owned candidates in input order."""
    first = Candidate(id="first", text="First evidence", retrieval_rank=4)
    second = Candidate(id="second", text="Second evidence", retrieval_rank=9)
    result = Refiner().refine(
        "query",
        (
            CandidateSet(name="dense", candidates=(first,)),
            CandidateSet(name="bm25", candidates=(second,)),
        ),
        top_k=2,
    )

    assert tuple(item.candidate for item in result.candidates) == (first, second)
    assert tuple(item.final_rank for item in result.candidates) == (1, 2)
    assert tuple(item.original_rank for item in result.candidates) == (4, 9)


def test_refiner_handles_empty_and_oversized_selection() -> None:
    """Empty input and a large requested limit both complete successfully."""
    refiner = Refiner()
    empty_result = refiner.refine("query", (), top_k=5)
    candidate = Candidate(id="only", text="Evidence")
    oversized_result = refiner.refine(
        "query",
        (CandidateSet(name="dense", candidates=(candidate,)),),
        top_k=10,
    )

    assert empty_result.candidates == ()
    assert tuple(item.candidate for item in oversized_result.candidates) == (candidate,)


def test_refiner_honors_zero_and_rejects_negative_top_k() -> None:
    """Selection limits are explicit and never invoke a hidden fallback."""
    candidate_set = CandidateSet(
        name="dense", candidates=(Candidate(id="one", text="Evidence"),)
    )

    assert Refiner().refine("query", (candidate_set,), top_k=0).candidates == ()
    with pytest.raises(ValueError, match="top_k must be non-negative"):
        Refiner().refine("query", (candidate_set,), top_k=-1)


def test_refiner_records_minimal_no_op_trace() -> None:
    """The trace captures stage timing and configuration without integrations."""
    result = Refiner().refine("query", (), top_k=3)

    assert result.trace.config_fingerprint == "no-op-v1"
    assert result.trace.stages[0].name == "no_op_selection"
    assert result.trace.stages[0].configuration == {"top_k": 3}
    assert result.trace.duration_ms >= 0
