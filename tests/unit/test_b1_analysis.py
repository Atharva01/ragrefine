"""Fast tests for frozen B0-to-B1 comparison artifacts."""

from benchmarks.beir.analyze_b1 import analyze
from benchmarks.beir.snapshot import snapshot_checksum


def test_analysis_reports_reproducible_gain_and_retain_decision() -> None:
    """A reordered fixed pool yields per-query deltas without model inference."""
    baseline = {
        "schema_version": "1.0",
        "retriever": {"top_n": 2},
        "qrels": {"q1": {"relevant": 1}},
        "queries": [
            {
                "id": "q1",
                "text": "query",
                "candidates": [
                    {"id": "other", "text": "other", "rank": 1, "score": 0.2},
                    {"id": "relevant", "text": "evidence", "rank": 2, "score": 0.1},
                ],
            }
        ],
    }
    ranking = {
        **baseline,
        "queries": [
            {
                **baseline["queries"][0],
                "candidates": list(reversed(baseline["queries"][0]["candidates"])),
            }
        ],
    }
    environment = {
        "b0_snapshot_checksum": snapshot_checksum(baseline),
        "model": "test-model",
        "revision": "abc",
        "device": "cpu",
        "backend": "test-backend",
        "batch_size": 2,
    }
    latency = [{"query_id": "q1", "duration_ms": 10.0}]

    report = analyze(baseline, ranking, environment, latency)

    assert report["reproducibility"]["identical"] is True
    assert report["effectiveness"]["per_query_outcomes"]["ndcg@5"] == {
        "wins": 1,
        "losses": 0,
        "unchanged": 0,
    }
    assert report["reranking_cost"]["query_document_pairs_per_second"] == 200.0
    assert report["decision"]["outcome"] == "retain"
    assert report["effectiveness"]["representative_wins"][0]["category"] == (
        "relevance_reordered_within_top_5"
    )
