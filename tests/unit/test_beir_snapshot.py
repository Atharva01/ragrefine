"""Tests for frozen benchmark artifacts without BEIR/model dependencies."""

from pathlib import Path

from benchmarks.beir.metrics import evaluate_snapshot
from benchmarks.beir.snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    load_snapshot,
    write_snapshot,
)


def _snapshot() -> dict[str, object]:
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "retriever": {"top_n": 3},
        "queries": [
            {
                "id": "q1",
                "text": "query",
                "candidates": [
                    {"id": "d1", "text": "one", "rank": 1, "score": 0.9},
                    {"id": "d2", "text": "two", "rank": 2, "score": 0.8},
                ],
            }
        ],
        "qrels": {"q1": {"d1": 2, "d3": 1}},
    }


def test_snapshot_round_trip_and_metrics_are_reproducible(tmp_path: Path) -> None:
    """A frozen ranking has stable content, checksum, and baseline metrics."""
    path = tmp_path / "snapshot.json"
    write_snapshot(_snapshot(), path)

    first = evaluate_snapshot(load_snapshot(path))
    second = evaluate_snapshot(load_snapshot(path))

    assert first == second
    assert first["ndcg@5"] > 0
    assert first["candidate_pool_recall@3"] == 0.5
