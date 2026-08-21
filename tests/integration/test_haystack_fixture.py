"""Integration tests for the frozen real Haystack retrieval fixture."""

import pytest

pytest.importorskip("haystack")

from integrations.haystack.fixture import FrozenHaystackFixture  # noqa: E402


def test_fixture_retrieves_scored_documents_and_has_stable_identity() -> None:
    fixture = FrozenHaystackFixture(top_n=2)

    first = fixture.retrieve(fixture.frozen_query("python"))
    second = fixture.retrieve(fixture.frozen_query("python"))

    assert len(first) == 2
    assert all(document.score is not None for document in first)
    assert [document.id for document in first] == [document.id for document in second]
    assert fixture.identity() == FrozenHaystackFixture(top_n=2).identity()


def test_baseline_and_treatment_can_share_the_exact_same_pool() -> None:
    fixture = FrozenHaystackFixture()
    pool = fixture.retrieve(fixture.frozen_query("rag"))

    baseline_input = pool
    treatment_input = pool

    assert baseline_input is treatment_input
    assert [document.id for document in baseline_input] == [
        document.id for document in treatment_input
    ]
