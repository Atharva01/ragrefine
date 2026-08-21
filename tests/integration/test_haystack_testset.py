"""Frozen Ragas test-set loading, checksum, and validation tests."""

import pytest

from benchmarks.haystack.testset import (
    DEFAULT_TESTSET_PATH,
    checksum,
    load_testset,
    validate,
)


def test_load_testset_verifies_checksum_and_shape() -> None:
    testset = load_testset()

    assert len(testset.corpus) == 15
    assert len(testset.qa_pairs) == 10
    expected = (
        DEFAULT_TESTSET_PATH.with_suffix(".json.sha256")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert testset.sha256 == expected

    corpus_ids = {document.id for document in testset.corpus}
    for pair in testset.qa_pairs:
        assert set(pair.reference_context_ids) <= corpus_ids


def test_validate_rejects_unknown_reference_context() -> None:
    data = {
        "schema_version": "1.0",
        "corpus": [{"id": "a", "content": "text", "meta": {"topic": "t"}}],
        "qa_pairs": [
            {
                "id": "q1",
                "question": "question?",
                "reference_answer": "answer",
                "reference_context_ids": ["missing"],
            }
        ],
    }
    with pytest.raises(ValueError):
        validate(data)


def test_validate_rejects_empty_collections() -> None:
    with pytest.raises(ValueError):
        validate({"schema_version": "1.0", "corpus": [], "qa_pairs": []})
    with pytest.raises(ValueError):
        validate(
            {
                "schema_version": "1.0",
                "corpus": [{"id": "a", "content": "text", "meta": {}}],
                "qa_pairs": [],
            }
        )


def test_validate_rejects_duplicate_corpus_ids() -> None:
    document = {"id": "dup", "content": "text", "meta": {"topic": "t"}}
    with pytest.raises(ValueError):
        validate(
            {
                "schema_version": "1.0",
                "corpus": [document, dict(document)],
                "qa_pairs": [
                    {
                        "id": "q1",
                        "question": "question?",
                        "reference_answer": "answer",
                        "reference_context_ids": ["dup"],
                    }
                ],
            }
        )


def test_checksum_is_deterministic() -> None:
    value = {"schema_version": "1.0", "corpus": [], "qa_pairs": []}
    assert checksum(value) == checksum(value)
