"""Tests for frozen benchmark artifacts without BEIR/model dependencies."""

from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.beir.config import RETRIEVER, DatasetConfig, baseline_config
from benchmarks.beir.metrics import evaluate_snapshot
from benchmarks.beir.snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    load_snapshot,
    load_verified_snapshot,
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


def test_baseline_config_records_the_generation_device() -> None:
    """A frozen snapshot identifies the device used to generate its candidates."""
    config = baseline_config(
        DatasetConfig("fiqa"),
        replace(RETRIEVER, device="cuda"),
    )

    assert config["retriever"] == {
        "backend": "torch",
        "device": "cuda",
        "model": "sentence-transformers/msmarco-MiniLM-L6-cos-v5",
        "revision": "14ca9be4bbcf1402eac0f43a2e2ccb6e0f994ba3",
        "top_n": 50,
    }


def test_verified_snapshot_requires_checksum_sidecar(tmp_path: Path) -> None:
    """Published ranking runs cannot consume an unchecked frozen input."""
    path = tmp_path / "snapshot.json"
    path.write_text('{"schema_version":"1.0"}', encoding="utf-8")

    with pytest.raises(ValueError, match="missing required checksum sidecar"):
        load_verified_snapshot(path)
