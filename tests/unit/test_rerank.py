"""Fast, download-free neural reranking tests."""

import pytest

from ragrefine.errors import RerankerError
from ragrefine.models import Candidate, CandidateSet
from ragrefine.refiner import Refiner
from ragrefine.rerank.sentence_transformers import SentenceTransformersReranker


class FakeCrossEncoder:
    def predict(
        self, sentences: list[tuple[str, str]], *, batch_size: int
    ) -> list[float]:
        assert batch_size == 2
        return [0.1, 0.9]


def test_cross_encoder_adapter_reranks_and_preserves_candidates() -> None:
    """Batched raw scores create deterministic ranks without changing evidence."""
    first = Candidate(id="first", text="first", retrieval_rank=1)
    second = Candidate(id="second", text="second", retrieval_rank=2)
    reranker = SentenceTransformersReranker(
        "test-model", revision="abc", batch_size=2, cross_encoder=FakeCrossEncoder()
    )

    scored = reranker.rank("query", (first, second))

    assert tuple(item.candidate for item in scored) == (second, first)
    assert tuple(item.rank for item in scored) == (1, 2)
    assert tuple(item.score for item in scored) == (0.9, 0.1)


def test_refiner_uses_opt_in_reranker_and_keeps_b0_unchanged() -> None:
    """Neural ranking happens only when a reranker is configured."""
    candidates = (
        Candidate(id="first", text="first"),
        Candidate(id="second", text="second"),
    )
    candidate_set = CandidateSet(name="dense", candidates=candidates)
    reranker = SentenceTransformersReranker(
        "test-model", batch_size=2, cross_encoder=FakeCrossEncoder()
    )

    result = Refiner(reranker=reranker).refine("query", (candidate_set,), top_k=1)
    baseline = Refiner().refine("query", (candidate_set,), top_k=1)

    assert result.candidates[0].candidate is candidates[1]
    assert result.candidates[0].signals["neural"].score == 0.9
    assert result.trace.stages[0].name == "neural_reranking"
    assert baseline.candidates[0].candidate is candidates[0]


def test_reranker_failure_is_explicit() -> None:
    """A malformed backend result is never silently replaced by B0 ordering."""

    class BrokenCrossEncoder:
        def predict(
            self, sentences: list[tuple[str, str]], *, batch_size: int
        ) -> list[float]:
            return []

    reranker = SentenceTransformersReranker(
        "test-model", cross_encoder=BrokenCrossEncoder()
    )
    with pytest.raises(RerankerError):
        reranker.rank("query", (Candidate(id="one", text="evidence"),))
