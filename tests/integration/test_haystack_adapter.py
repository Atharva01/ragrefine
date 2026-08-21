"""Focused tests for the pinned optional Haystack adapter."""

import pytest

haystack = pytest.importorskip("haystack")
Document = haystack.Document

from integrations.haystack.adapter import HaystackDocumentAdapter  # noqa: E402
from ragrefine.models import Candidate  # noqa: E402


def test_round_trip_preserves_original_document_identity_and_provenance() -> None:
    first = Document(id="one", content="Evidence", meta={"source": "a"}, score=0.8)
    second = Document(id="two", content="More", meta={"source": "b"})
    adapter = HaystackDocumentAdapter((first, second), name="retriever")

    candidates = adapter.candidate_set()

    assert candidates.name == "retriever"
    assert candidates.candidates[0] == Candidate(
        id="one",
        text="Evidence",
        metadata={"source": "a"},
        retrieval_score=0.8,
        retrieval_rank=1,
    )
    assert candidates.candidates[1].retrieval_score is None
    assert adapter.documents_for((candidates.candidates[1],)) == (second,)
    assert adapter.documents_for((candidates.candidates[0],))[0] is first


def test_adapter_rejects_invalid_or_foreign_inputs() -> None:
    with pytest.raises(TypeError, match="Haystack Document"):
        HaystackDocumentAdapter((object(),))  # type: ignore[arg-type]

    adapter = HaystackDocumentAdapter((Document(id="one", content="Evidence"),))
    with pytest.raises(ValueError, match="not created"):
        adapter.documents_for((Candidate(id="other", text="Evidence"),))
    with pytest.raises(ValueError, match="text does not match"):
        adapter.documents_for((Candidate(id="one", text="changed"),))
