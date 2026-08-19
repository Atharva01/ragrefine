"""Contract tests for the frozen-artifact B3 comparison."""

from benchmarks.beir import analyze_b3


def test_per_query_outcomes_distinguish_wins_losses_and_ties() -> None:
    """The ledger exposes regressions instead of masking them with aggregates."""
    baseline = {
        "a": {metric: 1.0 for metric in analyze_b3.PRIMARY_METRICS},
        "b": {metric: 1.0 for metric in analyze_b3.PRIMARY_METRICS},
        "c": {metric: 1.0 for metric in analyze_b3.PRIMARY_METRICS},
    }
    candidate = {
        "a": {metric: 2.0 for metric in analyze_b3.PRIMARY_METRICS},
        "b": {metric: 0.0 for metric in analyze_b3.PRIMARY_METRICS},
        "c": {metric: 1.0 for metric in analyze_b3.PRIMARY_METRICS},
    }

    outcomes = analyze_b3._outcomes(baseline, candidate)

    assert outcomes["ndcg@5"] == {
        "wins": ["a"],
        "losses": ["b"],
        "unchanged": ["c"],
    }


def test_hard_diagnostics_preserve_category_level_results() -> None:
    """Hard-set analysis records Top-1, MRR, and relevant rank by category."""
    snapshot = {
        "qrels": {"q": {"relevant": 1}},
        "queries": [
            {
                "id": "q",
                "category": "wrong_version",
                "candidates": [{"id": "other"}, {"id": "relevant"}],
            }
        ],
    }

    diagnostics = analyze_b3._hard_diagnostics(snapshot)

    assert diagnostics["by_category"]["wrong_version"] == {
        "top_1_accuracy": 0.0,
        "mrr": 0.5,
        "mean_relevant_rank": 2.0,
    }


def test_b3_decisions_do_not_retain_fusion_without_evidence() -> None:
    """The final decision keeps a stronger single channel over weak fusion."""
    metrics = {
        name: {
            metric: (1.0 if name in {"B1-reference", "B2-lexical"} else 0.5)
            for metric in analyze_b3.PRIMARY_METRICS
        }
        for name in (
            "B1-reference",
            "B2-lexical",
            "B3-original-lexical",
            "B3-original-pattern",
            "B3-light",
            "B3-reference",
        )
    }
    decisions = analyze_b3._decisions(metrics)

    assert decisions["B3-reference"]["decision"] == "reject"
    assert "B1-reference" in decisions["recommended_default"]["decision"]
