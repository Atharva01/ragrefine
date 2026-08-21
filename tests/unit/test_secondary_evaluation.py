"""Tests for qualified secondary-dataset profile evaluation."""

import copy
import json
from pathlib import Path

import pytest

from benchmarks.beir import secondary
from benchmarks.beir.snapshot import snapshot_checksum, write_snapshot


def _baseline() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "dataset": {"name": "fixture", "version": "1"},
        "retriever": {"top_n": 6},
        "qrels": {"q1": {"q1-a": 1}, "q2": {"q2-a": 1}},
        "queries": [
            {
                "id": query_id,
                "text": f"query {query_id}",
                "candidates": [
                    {
                        "id": f"{query_id}-{letter}",
                        "text": f"document {query_id} {letter}",
                        "rank": rank,
                        "score": 1.0 / rank,
                    }
                    for rank, letter in enumerate("bcdefa", start=1)
                ]
                if query_id == "q1"
                else [
                    {
                        "id": f"{query_id}-{letter}",
                        "text": f"document {query_id} {letter}",
                        "rank": rank,
                        "score": 1.0 / rank,
                    }
                    for rank, letter in enumerate("abcdef", start=1)
                ],
            }
            for query_id in ("q1", "q2")
        ],
    }


def _rerank(baseline: dict[str, object], *, profile: str) -> dict[str, object]:
    ranking = copy.deepcopy(baseline)
    if profile == "b1":
        q1_candidates = ranking["queries"][0]["candidates"]
        q1_candidates[-1], q1_candidates[-2] = q1_candidates[-2], q1_candidates[-1]
        q2_candidates = ranking["queries"][1]["candidates"]
        q2_candidates.append(q2_candidates.pop(0))
        for query in ranking["queries"]:
            for rank, candidate in enumerate(query["candidates"], start=1):
                candidate["rank"] = rank
        ranking["b1"] = {
            "model": secondary.B1_MODEL,
            "revision": secondary.B1_REVISION,
        }
    else:
        ranking["b2"] = {"signal": "lexical", "signal_version": "1"}
    return ranking


def _environments(
    baseline: object, b1: object, b2: object
) -> tuple[dict[str, object], dict[str, object]]:
    baseline_checksum = snapshot_checksum(baseline)
    return (
        {
            "b0_snapshot_checksum": baseline_checksum,
            "ranking_checksum": snapshot_checksum(b1),
            "model": secondary.B1_MODEL,
            "revision": secondary.B1_REVISION,
            "backend": "sentence-transformers",
            "device": "cuda",
            "batch_size": 8,
            "reranking_wall_clock": {"elapsed_ms": 10.0},
        },
        {
            "b0_snapshot_checksum": baseline_checksum,
            "ranking_checksum": snapshot_checksum(b2),
            "signal": "lexical",
            "backend": "stdlib",
            "device": "cpu",
            "runtime": {"total_ms": 1.0},
        },
    )


def test_secondary_analysis_reports_metrics_outcomes_and_regressions() -> None:
    """Quality deltas and query outcomes remain separate from runtime evidence."""
    baseline = _baseline()
    b1, b2 = _rerank(baseline, profile="b1"), _rerank(baseline, profile="b2")
    b1_environment, b2_environment = _environments(baseline, b1, b2)

    summary, per_query = secondary.analyze(
        baseline,
        b1,
        b2,
        b1_environment,
        b2_environment,
        qualified_snapshots={"fixture": snapshot_checksum(baseline)},
    )

    quality = summary["retrieval_quality"]
    assert quality["metrics"].keys() == {"B0", "B1-reference", "B2-L"}
    assert quality["per_query_outcomes_vs_b0"]["B1-reference"]["mrr@5"] == {
        "wins": 1,
        "losses": 1,
        "unchanged": 0,
    }
    assert quality["dataset_regressions"]["B1-reference"]
    assert per_query[0]["outcome_vs_b0"]["B1-reference"]["mrr@5"] == "win"
    assert per_query[1]["outcome_vs_b0"]["B1-reference"]["mrr@5"] == "loss"
    assert "runtime_observations" in summary
    assert "runtime" not in quality


def test_secondary_analysis_rejects_changed_candidate_population() -> None:
    """A retained profile cannot invent, omit, or rewrite frozen candidates."""
    baseline = _baseline()
    b1, b2 = _rerank(baseline, profile="b1"), _rerank(baseline, profile="b2")
    b2["queries"][0]["candidates"][0]["text"] = "rewritten evidence"
    b1_environment, b2_environment = _environments(baseline, b1, b2)

    with pytest.raises(ValueError, match="does not preserve"):
        secondary.analyze(
            baseline,
            b1,
            b2,
            b1_environment,
            b2_environment,
            qualified_snapshots={"fixture": snapshot_checksum(baseline)},
        )


def test_secondary_run_reproduces_from_persisted_rankings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduction re-evaluates checked artifacts without model execution."""
    baseline = _baseline()
    b1, b2 = _rerank(baseline, profile="b1"), _rerank(baseline, profile="b2")
    b1_environment, b2_environment = _environments(baseline, b1, b2)
    monkeypatch.setitem(
        secondary.QUALIFIED_SNAPSHOTS, "fixture", snapshot_checksum(baseline)
    )
    baseline_path = tmp_path / "baseline.json"
    b1_path = tmp_path / "b1.json"
    b2_path = tmp_path / "b2.json"
    write_snapshot(baseline, baseline_path)
    write_snapshot(b1, b1_path)
    write_snapshot(b2, b2_path)
    b1_environment_path = tmp_path / "b1-environment.json"
    b2_environment_path = tmp_path / "b2-environment.json"
    b1_environment_path.write_text(json.dumps(b1_environment), encoding="utf-8")
    b2_environment_path.write_text(json.dumps(b2_environment), encoding="utf-8")

    output_dir = tmp_path / "analysis"
    secondary.run(
        baseline_path,
        b1_path,
        b1_environment_path,
        b2_path,
        b2_environment_path,
        output_dir,
    )
    reproduction = secondary.reproduce(output_dir)

    assert reproduction["logical_results_match"] is True
    assert reproduction["retrieval_or_reranking_executed"] is False
    assert (output_dir / "per-query.json").exists()
