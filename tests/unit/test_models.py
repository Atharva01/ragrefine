"""Unit tests for core candidate domain models."""

from dataclasses import FrozenInstanceError

import pytest

from ragrefine import Candidate as PublicCandidate
from ragrefine import CandidateSet as PublicCandidateSet
from ragrefine.models import Candidate, CandidateSet


def test_candidate_preserves_retrieval_evidence() -> None:
    """Candidate retains the supplied identifier, evidence, and annotations."""
    metadata = {"source": "dense", "document_id": "doc-42"}

    candidate = Candidate(
        id="chunk-7",
        text="The original retrieved evidence.",
        metadata=metadata,
        retrieval_score=0.82,
        retrieval_rank=3,
    )

    assert candidate.id == "chunk-7"
    assert candidate.text == "The original retrieved evidence."
    assert candidate.metadata == metadata
    assert candidate.retrieval_score == 0.82
    assert candidate.retrieval_rank == 3
    assert metadata == {"source": "dense", "document_id": "doc-42"}


def test_candidate_is_immutable() -> None:
    """Candidate fields cannot be reassigned after construction."""
    candidate = Candidate(id="chunk-7", text="Evidence")

    with pytest.raises(FrozenInstanceError):
        candidate.text = "Changed evidence"  # type: ignore[misc]


def test_candidate_set_preserves_candidate_order() -> None:
    """CandidateSet preserves its named, ordered candidate tuple."""
    candidates = (
        Candidate(id="first", text="First"),
        Candidate(id="second", text="Second"),
    )

    candidate_set = CandidateSet(name="dense", candidates=candidates)

    assert candidate_set.name == "dense"
    assert candidate_set.candidates == candidates
    assert tuple(candidate.id for candidate in candidate_set.candidates) == (
        "first",
        "second",
    )


def test_candidate_set_is_immutable() -> None:
    """CandidateSet fields cannot be reassigned after construction."""
    candidate_set = CandidateSet(name="dense", candidates=())

    with pytest.raises(FrozenInstanceError):
        candidate_set.name = "bm25"  # type: ignore[misc]


def test_candidate_models_are_exposed_from_the_public_api() -> None:
    """Consumers can import the approved candidate contracts from ragrefine."""
    assert PublicCandidate is Candidate
    assert PublicCandidateSet is CandidateSet
