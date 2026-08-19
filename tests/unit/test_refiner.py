"""Unit tests for the no-op refinement baseline."""

import pytest

from ragrefine.errors import RerankerError
from ragrefine.models import Candidate, CandidateSet
from ragrefine.refiner import Refiner
from ragrefine.rerank.base import ScoredCandidate


class ReversingReranker:
    """Fast fake that makes neural-stage behaviour observable without a model."""

    name = "test-reranker"

    def rank(
        self, query: str, candidates: tuple[Candidate, ...]
    ) -> tuple[ScoredCandidate, ...]:
        del query
        return tuple(
            ScoredCandidate(
                candidate=candidate,
                score=float(index),
                rank=index,
                model="test-model",
                model_revision="abc123",
                backend="test-backend",
            )
            for index, candidate in enumerate(reversed(candidates), start=1)
        )


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


def test_refiner_applies_neural_order_before_top_k_and_records_trace() -> None:
    """The neural stage preserves evidence and exposes ranking provenance."""
    first = Candidate(
        id="first",
        text="First evidence",
        metadata={"source": "dense"},
        retrieval_score=0.7,
        retrieval_rank=1,
    )
    second = Candidate(id="second", text="Second evidence", retrieval_rank=2)
    result = Refiner(reranker=ReversingReranker()).refine(
        "query", (CandidateSet(name="dense", candidates=(first, second)),), top_k=1
    )
    repeated = Refiner(reranker=ReversingReranker()).refine(
        "query", (CandidateSet(name="dense", candidates=(first, second)),), top_k=1
    )

    selected = result.candidates[0]
    stage = result.trace.stages[0]
    assert selected.candidate is second
    assert tuple(item.candidate for item in repeated.candidates) == (second,)
    assert selected.original_rank == 2
    assert selected.signals["neural"].rank == 1
    assert selected.signals["neural"].score == 1.0
    assert selected.signals["neural"].details == {
        "model": "test-model",
        "revision": "abc123",
        "backend": "test-backend",
    }
    assert stage.name == "neural_reranking"
    assert stage.duration_ms >= 0
    assert stage.configuration == {
        "top_k": 1,
        "reranker": "test-reranker",
        "model": "test-model",
        "revision": "abc123",
        "backend": "test-backend",
    }


def test_refiner_surfaces_neural_reranker_failure() -> None:
    """The pipeline never silently changes back to B0 after a backend failure."""

    class FailingReranker:
        name = "failing-reranker"

        def rank(
            self, query: str, candidates: tuple[Candidate, ...]
        ) -> tuple[ScoredCandidate, ...]:
            del query, candidates
            raise RerankerError("backend unavailable")

    with pytest.raises(RerankerError, match="backend unavailable"):
        Refiner(reranker=FailingReranker()).refine(
            "query",
            (CandidateSet(name="dense", candidates=(Candidate(id="one", text="e"),)),),
        )
