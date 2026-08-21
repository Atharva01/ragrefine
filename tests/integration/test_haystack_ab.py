"""Frozen-pool Haystack A/B integration test."""

import json
from pathlib import Path

import pytest

pytest.importorskip("haystack")

from benchmarks.haystack.ab import run  # noqa: E402


def test_ab_persists_shared_pools_and_traces(tmp_path: Path) -> None:
    result = run(tmp_path / "ab")

    assert result["pool_reused"] is True
    for row in result["per_query"]:
        assert [item["id"] for item in row["input_pool"]] == row["baseline_ids"]
        assert set(row["refined_ids"]) == set(row["baseline_ids"])
        assert row["trace"]["stages"]
    assert json.loads((tmp_path / "ab" / "ab-results.json").read_text()) == result
