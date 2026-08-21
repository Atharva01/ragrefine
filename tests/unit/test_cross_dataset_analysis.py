"""Tests for persisted cross-dataset retained-profile analysis."""

import copy
import json
from pathlib import Path

import pytest

from benchmarks.beir import cross_dataset, secondary
from benchmarks.beir.snapshot import snapshot_checksum, write_snapshot


def _baseline(dataset: str) -> dict[str, object]:
    def candidates(query_id: str, order: str) -> list[dict[str, object]]:
        return [
            {
                "id": f"{query_id}-{letter}",
                "text": f"{dataset} evidence {query_id} {letter}",
                "rank": rank,
                "score": 1.0 / rank,
            }
            for rank, letter in enumerate(order, start=1)
        ]

    return {
        "schema_version": "1.0",
        "dataset": {"name": dataset, "version": "fixture"},
        "retriever": {"top_n": 6},
        "qrels": {"q1": {"q1-a": 1}, "q2": {"q2-z": 1}},
        "queries": [
            {
                "id": "q1",
                "text": "first query",
                "candidates": candidates("q1", "bcdefa"),
            },
            {
                "id": "q2",
                "text": "second query",
                "candidates": candidates("q2", "zbcdef"),
            },
        ],
    }


def _ranking(
    baseline: dict[str, object], *, profile: str, regress: bool = False
) -> dict[str, object]:
    ranking = copy.deepcopy(baseline)
    if profile == "b1" or not regress:
        first = ranking["queries"][0]["candidates"]
        first.insert(0, first.pop())
    if profile == "b2" and regress:
        second = ranking["queries"][1]["candidates"]
        second.append(second.pop(0))
    for query in ranking["queries"]:
        for rank, candidate in enumerate(query["candidates"], start=1):
            candidate["rank"] = rank
    if profile == "b1":
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
        },
        {
            "b0_snapshot_checksum": baseline_checksum,
            "ranking_checksum": snapshot_checksum(b2),
            "signal": "lexical",
            "backend": "stdlib",
            "device": "cpu",
        },
    )


def _write_inputs(
    tmp_path: Path, dataset: str, *, regress_b2: bool = False
) -> tuple[cross_dataset.DatasetPaths, str]:
    baseline = _baseline(dataset)
    b1 = _ranking(baseline, profile="b1")
    b2 = _ranking(baseline, profile="b2", regress=regress_b2)
    b1_environment, b2_environment = _environments(baseline, b1, b2)
    directory = tmp_path / dataset
    baseline_path = directory / "b0.json"
    b1_path = directory / "b1.json"
    b2_path = directory / "b2.json"
    write_snapshot(baseline, baseline_path)
    write_snapshot(b1, b1_path)
    write_snapshot(b2, b2_path)
    b1_environment_path = directory / "b1-environment.json"
    b2_environment_path = directory / "b2-environment.json"
    b1_environment_path.write_text(json.dumps(b1_environment), encoding="utf-8")
    b2_environment_path.write_text(json.dumps(b2_environment), encoding="utf-8")
    return (
        cross_dataset.DatasetPaths(
            baseline_path,
            b1_path,
            b1_environment_path,
            b2_path,
            b2_environment_path,
        ),
        snapshot_checksum(baseline),
    )


def test_cross_dataset_analysis_classifies_stable_and_dataset_specific_effects(
    tmp_path: Path,
) -> None:
    """B1 can be stable while B2-L changes direction across domains."""
    scifact, scifact_checksum = _write_inputs(tmp_path, "scifact")
    nfcorpus, nfcorpus_checksum = _write_inputs(tmp_path, "nfcorpus")
    fiqa, fiqa_checksum = _write_inputs(tmp_path, "fiqa", regress_b2=True)

    summary, representatives, manifest = cross_dataset.analyze_all(
        {"scifact": scifact, "nfcorpus": nfcorpus, "fiqa": fiqa},
        qualified_snapshots={
            "scifact": scifact_checksum,
            "nfcorpus": nfcorpus_checksum,
            "fiqa": fiqa_checksum,
        },
    )

    assert summary["stability"]["B1-reference"]["classification"] == (
        "consistent aggregate improvement"
    )
    assert summary["stability"]["B1-reference"]["all_primary_metrics_improve_on"] == [
        "fiqa",
        "nfcorpus",
        "scifact",
    ]
    assert summary["stability"]["B2-L"]["classification"] == (
        "dataset-specific aggregate effect"
    )
    assert (
        summary["datasets"]["scifact"]["metric_orderings"]["mrr"][0]["profile"]
        == "B1-reference"
    )
    fiqa_regression = next(
        item
        for item in representatives
        if item["dataset"] == "fiqa"
        and item["profile"] == "B2-L"
        and item["direction"] == "regression"
    )
    assert fiqa_regression["top_5"]["B0"][0]["relevance"] == 1
    assert fiqa_regression["top_5"]["B2-L"][0]["relevance"] == 0
    assert manifest["datasets"].keys() == {"scifact", "nfcorpus", "fiqa"}


def test_cross_dataset_run_reproduces_checked_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report replays exact rankings and does not invoke retrieval or models."""
    scifact, scifact_checksum = _write_inputs(tmp_path, "scifact")
    nfcorpus, nfcorpus_checksum = _write_inputs(tmp_path, "nfcorpus")
    fiqa, fiqa_checksum = _write_inputs(tmp_path, "fiqa", regress_b2=True)
    for dataset, checksum in {
        "scifact": scifact_checksum,
        "nfcorpus": nfcorpus_checksum,
        "fiqa": fiqa_checksum,
    }.items():
        monkeypatch.setitem(cross_dataset.QUALIFIED_SNAPSHOTS, dataset, checksum)

    output_dir = tmp_path / "report"
    cross_dataset.run(
        output_dir,
        inputs={"scifact": scifact, "nfcorpus": nfcorpus, "fiqa": fiqa},
    )
    reproduction = cross_dataset.reproduce(output_dir)

    assert reproduction["logical_results_match"] is True
    assert reproduction["retrieval_or_reranking_executed"] is False
    assert (output_dir / "representative-cases.json").exists()
