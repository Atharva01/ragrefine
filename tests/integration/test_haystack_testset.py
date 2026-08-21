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

    assert len(testset.corpus) == 21
    assert len(testset.qa_pairs) == 13
    expected = (
        DEFAULT_TESTSET_PATH.with_suffix(".json.sha256")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert testset.sha256 == expected

    corpus_ids = {document.id for document in testset.corpus}
    for pair in testset.qa_pairs:
        assert set(pair.reference_context_ids) <= corpus_ids
        assert pair.category
    assert {pair.category for pair in testset.qa_pairs} == {
        "date",
        "factual",
        "identifier",
        "numeric_value",
        "version",
    }
    assert testset.review.status == "human-reviewed"
    assert testset.contract.top_n == 5
    assert testset.contract.top_k == 3
    assert testset.contract.max_tokens is None


def test_validate_rejects_unknown_reference_context() -> None:
    data = {
        "schema_version": "2.0",
        "review": _review(),
        "evaluation_contract": _contract(),
        "corpus": [{"id": "a", "content": "text", "meta": {"topic": "t"}}],
        "qa_pairs": [
            {
                "id": "q1",
                "question": "question?",
                "reference_answer": "answer",
                "reference_context_ids": ["missing"],
                "category": "factual",
            }
        ],
    }
    with pytest.raises(ValueError):
        validate(data)


def test_validate_rejects_empty_collections() -> None:
    with pytest.raises(ValueError):
        validate(
            {
                "schema_version": "2.0",
                "review": _review(),
                "evaluation_contract": _contract(),
                "corpus": [],
                "qa_pairs": [],
            }
        )
    with pytest.raises(ValueError):
        validate(
            {
                "schema_version": "2.0",
                "review": _review(),
                "evaluation_contract": _contract(),
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
                        "category": "factual",
                    }
                ],
            }
        )


def test_checksum_is_deterministic() -> None:
    value = {
        "schema_version": "2.0",
        "review": _review(),
        "evaluation_contract": _contract(),
        "corpus": [],
        "qa_pairs": [],
    }
    assert checksum(value) == checksum(value)


def test_validate_rejects_missing_category_or_changed_contract() -> None:
    data = {
        "schema_version": "2.0",
        "review": _review(),
        "evaluation_contract": _contract(),
        "corpus": [{"id": "a", "content": "text", "meta": {}}],
        "qa_pairs": [
            {
                "id": "q1",
                "question": "question?",
                "reference_answer": "answer",
                "reference_context_ids": ["a"],
            }
        ],
    }
    with pytest.raises(ValueError):
        validate(data)

    data["qa_pairs"][0]["category"] = "factual"
    data["evaluation_contract"]["top_k"] = 6
    with pytest.raises(ValueError):
        validate(data)


def _review() -> dict[str, str]:
    return {
        "status": "human-reviewed",
        "reviewer_role": "reviewer",
        "reviewed_on": "2026-08-21",
    }


def _contract() -> dict[str, object]:
    return {
        "top_n": 5,
        "top_k": 3,
        "max_tokens": None,
        "prompt_template": "ragrefine-haystack-prompt-v1",
        "generator": {
            "model": "deepseek-chat",
            "api_base_url": "https://api.deepseek.com/v1",
            "temperature": 0,
            "max_tokens": 512,
        },
        "metrics": ["faithfulness", "context_recall", "factual_correctness"],
        "refinement_profile": "B2-L lexical",
    }
