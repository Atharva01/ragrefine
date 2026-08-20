"""Tests for frozen-pool redundancy characterization."""

from benchmarks.beir.redundancy import analyze_snapshot, word_token_count


def _snapshot() -> dict[str, object]:
    base = " ".join(f"term{index}" for index in range(30))
    near = " ".join([*(f"term{index}" for index in range(29)), "different"])
    return {
        "dataset": {"name": "fixture"},
        "queries": [
            {
                "id": "q1",
                "candidates": [
                    {"id": "d1", "text": base, "rank": 1, "score": 0.9},
                    {"id": "d2", "text": base.upper(), "rank": 2, "score": 0.8},
                    {"id": "d3", "text": near, "rank": 3, "score": 0.7},
                ],
            },
            {
                "id": "q2",
                "candidates": [
                    {
                        "id": "d4",
                        "text": "independent evidence",
                        "rank": 1,
                        "score": 0.6,
                    }
                ],
            },
        ],
        "qrels": {"q1": {"d1": 1}, "q2": {"d4": 1}},
    }


def test_redundancy_audit_reports_exact_and_predeclared_near_duplicates() -> None:
    """The audit reports deterministic pool pressure at each fixed threshold."""
    result = analyze_snapshot(_snapshot())

    assert result["query_count"] == 2
    assert result["measurement"]["relevance_labels_used"] is False
    assert result["candidate_pool"]["candidates_per_query"]["mean"] == 2.0
    assert result["candidate_pool"]["word_tokens"] == 92
    assert result["exact_normalized_duplicates"]["duplicate_candidates"] == 1
    assert result["near_duplicate_thresholds"]["0.80"]["duplicate_candidates"] == 2
    assert result["near_duplicate_thresholds"]["0.95"]["duplicate_candidates"] == 1


def test_redundancy_audit_does_not_depend_on_qrels() -> None:
    """Pool characterization is independent of relevance labels and outcomes."""
    snapshot = _snapshot()
    changed_qrels = {**snapshot, "qrels": {"q1": {"d3": 99}}}

    assert analyze_snapshot(snapshot) == analyze_snapshot(changed_qrels)


def test_word_token_count_is_dependency_free_and_unicode_aware() -> None:
    """The documented proxy counts word-like Unicode tokens deterministically."""
    assert word_token_count("CAFÉ release-v17") == 3
