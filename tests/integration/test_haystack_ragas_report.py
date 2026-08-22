"""Ragas A/B report tests over hand-built artifacts with injected verifier."""

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.haystack.ragas_report import (
    _overall_decision,
    run,
)
from benchmarks.haystack.testset import load_testset

METRICS = [
    "context_precision",
    "id_based_context_recall",
    "faithfulness",
    "factual_correctness",
]
QA_IDS = ["q-01", "q-02", "q-03"]


def _write_artifacts(
    tmp_path: Path,
    *,
    baseline: dict[str, list[float]],
    refined: dict[str, list[float]],
) -> tuple[Path, Path]:
    """Write a schema-1.1 input artifact and schema-1.0 scores artifact."""
    testset = load_testset()
    corpus_ids = [document.id for document in testset.corpus][:3]
    rows = []
    for index, qa_id in enumerate(QA_IDS):
        rows.append(
            {
                "qa_id": qa_id,
                "question": f"Question {index}",
                "reference_answer": "reference",
                "reference_context_ids": [corpus_ids[0]],
                "pool_ids": list(corpus_ids),
                "pool_sha256": "d" * 64,
                "baseline_context_ids": list(corpus_ids),
                "refined_context_ids": list(corpus_ids),
                "refinement_trace": {
                    "stages": [],
                    "duration_ms": 0.0,
                    "config_fingerprint": "f" * 64,
                },
                "baseline_prompt": "Question " + "word " * (10 + index),
                "refined_prompt": "Question " + "word " * (12 + index),
                "baseline_answer": "baseline answer",
                "refined_answer": "refined answer",
                "baseline_failure": None,
                "refined_failure": None,
                "paired": True,
                "retrieval_latency_seconds": 0.0,
                "refinement_latency_seconds": 0.0,
                "baseline_generation_latency_seconds": 0.0,
                "refined_generation_latency_seconds": 0.0,
            }
        )
    artifact = {
        "schema_version": "1.1",
        "testset": {
            "path": "frozen",
            "sha256": testset.sha256,
            "corpus_size": len(testset.corpus),
            "qa_pair_count": len(QA_IDS),
        },
        "model": "deepseek-chat",
        "api_base_url": "https://api.deepseek.com/v1",
        "profile": "B2-L lexical",
        "configuration": {"top_n": 5, "top_k": 3, "pool_reused": True},
        "evaluation": {"queries_total": len(QA_IDS), "paired_query_count": len(QA_IDS)},
        "per_query": rows,
        "metrics": {"names": [], "baseline": {}, "refined": {}},
        "environment": {"python_version": "3.12"},
        "interpretation": "fixture",
    }
    artifact_path = tmp_path / "ragas-results.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (tmp_path / "ragas-results.json.sha256").write_text(
        hashlib.sha256(artifact_path.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )

    def _arm_block(scores: dict[str, list[float]]) -> dict[str, object]:
        block: dict[str, object] = {}
        for metric in METRICS:
            per_sample = dict(zip(QA_IDS, scores[metric]))
            aggregate = sum(per_sample.values()) / len(per_sample)
            block[metric] = {
                "aggregate": aggregate,
                "per_sample": per_sample,
                "failures": [],
                "judge_reasons": {},
            }
        return block

    scores = {
        "schema_version": "1.0",
        "input_artifact": {
            "path": str(artifact_path),
            "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            "schema_version": "1.1",
        },
        "testset": {"path": "frozen", "sha256": testset.sha256},
        "model": "deepseek-chat",
        "configuration": {
            "metrics": METRICS,
            "omitted_metrics": {
                "answer_relevancy": "no pinned embedding provider configured"
            },
            "paired_rows_scored": len(QA_IDS),
            "unpaired_rows_skipped": 0,
            "input_pool_reused": True,
        },
        "scores": {
            "baseline": _arm_block(baseline),
            "refined": _arm_block(refined),
        },
        "environment": {"python_version": "3.12"},
        "interpretation": "fixture",
    }
    scores_path = tmp_path / "ragas-scores.json"
    scores_path.write_text(
        json.dumps(scores, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (tmp_path / "ragas-scores.json.sha256").write_text(
        hashlib.sha256(scores_path.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    return artifact_path, scores_path


def _uniform(value: float) -> list[float]:
    return [value for _ in QA_IDS]


def _run_report(
    tmp_path: Path,
    *,
    baseline: dict[str, list[float]],
    refined: dict[str, list[float]],
    name: str = "report",
) -> dict[str, object]:
    artifact_path, scores_path = _write_artifacts(
        tmp_path, baseline=baseline, refined=refined
    )
    return run(
        tmp_path / name,
        artifact_path=artifact_path,
        scores_path=scores_path,
    )


def test_report_supported_improvement(tmp_path: Path) -> None:
    # context_precision: refined clearly better; id_recall: regression;
    # faithfulness: mixed; factual_correctness: ties.
    result = _run_report(
        tmp_path,
        baseline={
            "context_precision": [0.5, 0.5, 0.5],
            "id_based_context_recall": [0.8, 0.8, 0.8],
            "faithfulness": [0.9, 0.5, 0.3],
            "factual_correctness": [0.7, 0.7, 0.7],
        },
        refined={
            "context_precision": [0.9, 0.9, 0.9],
            "id_based_context_recall": [0.4, 0.4, 0.4],
            "faithfulness": [0.5, 0.6, 0.4],
            "factual_correctness": [0.7, 0.7, 0.7],
        },
    )
    precision = result["per_metric"]["context_precision"]
    assert precision["outcomes"] == {"wins": 3, "losses": 0, "ties": 0}
    assert precision["mean_paired_delta"] == pytest.approx(0.4)
    assert precision["paired_delta_ci95"][0] > 0
    assert precision["decision"]["outcome"] == "supported_improvement"
    assert precision["representative_regressions"] == []

    recall = result["per_metric"]["id_based_context_recall"]
    assert recall["outcomes"] == {"wins": 0, "losses": 3, "ties": 0}
    assert recall["decision"]["outcome"] == "regression"
    assert [item["qa_id"] for item in recall["representative_regressions"]] == QA_IDS

    faithfulness = result["per_metric"]["faithfulness"]
    assert faithfulness["outcomes"] == {"wins": 2, "losses": 1, "ties": 0}
    assert faithfulness["decision"]["outcome"] == "inconclusive"

    correctness = result["per_metric"]["factual_correctness"]
    assert correctness["outcomes"] == {"wins": 0, "losses": 0, "ties": 3}
    assert correctness["decision"]["outcome"] == "inconclusive"

    # Mixed signals -> overall inconclusive.
    assert result["decision"]["outcome"] == "inconclusive"
    assert result["statistics"]["bootstrap_samples"] == 10_000
    assert result["statistics"]["bootstrap_seed"] == 97


def test_report_overall_decision_rules() -> None:
    assert (
        _overall_decision(["supported_improvement", "inconclusive"])["outcome"]
        == "supported_improvement"
    )
    assert _overall_decision(["regression", "inconclusive"])["outcome"] == "regression"
    assert _overall_decision(["inconclusive"])["outcome"] == "inconclusive"
    assert (
        _overall_decision(["supported_improvement", "regression"])["outcome"]
        == "inconclusive"
    )


def test_report_context_sizes_from_prompts(tmp_path: Path) -> None:
    result = _run_report(
        tmp_path,
        baseline={metric: _uniform(0.5) for metric in METRICS},
        refined={metric: _uniform(0.5) for metric in METRICS},
    )
    context = result["context"]
    # Prompts contain "Question " + N * "word " -> N + 1 tokens each.
    assert context["aggregate"]["baseline_words"] == sum(11 + i for i in range(3))
    assert context["aggregate"]["refined_words"] == sum(13 + i for i in range(3))
    assert context["aggregate"]["delta_words"] == 6
    assert context["per_query"]["q-01"]["delta_words"] == 2


def test_report_is_deterministic(tmp_path: Path) -> None:
    first = _run_report(
        tmp_path,
        baseline={metric: _uniform(0.5) for metric in METRICS},
        refined={metric: _uniform(0.6) for metric in METRICS},
        name="a",
    )
    second = _run_report(
        tmp_path,
        baseline={metric: _uniform(0.5) for metric in METRICS},
        refined={metric: _uniform(0.6) for metric in METRICS},
        name="b",
    )
    assert first["per_metric"] == second["per_metric"]
    assert first["decision"] == second["decision"]


def test_report_embeds_verification_and_writes_artifacts(tmp_path: Path) -> None:
    artifact_path, scores_path = _write_artifacts(
        tmp_path,
        baseline={metric: _uniform(0.5) for metric in METRICS},
        refined={metric: _uniform(0.6) for metric in METRICS},
    )
    result = run(
        tmp_path / "report",
        artifact_path=artifact_path,
        scores_path=scores_path,
    )
    assert result["verification"]["ok"] is True
    assert result["verification"]["failures"] == []
    output_path = tmp_path / "report" / "ragas-report.json"
    assert json.loads(output_path.read_text()) == result
    sidecar = tmp_path / "report" / "ragas-report.json.sha256"
    assert (
        sidecar.read_text(encoding="utf-8").strip()
        == hashlib.sha256(output_path.read_bytes()).hexdigest()
    )
    markdown = (tmp_path / "report" / "ragas-report.md").read_text(encoding="utf-8")
    assert "# Haystack Ragas A/B report" in markdown
    assert "Overall decision" in markdown


def test_report_rejects_mismatched_scores_input(tmp_path: Path) -> None:
    artifact_path, _ = _write_artifacts(
        tmp_path,
        baseline={metric: _uniform(0.5) for metric in METRICS},
        refined={metric: _uniform(0.6) for metric in METRICS},
    )
    # A scores artifact that references a different input must be rejected.
    testset = load_testset()
    scores = {
        "schema_version": "1.0",
        "input_artifact": {
            "path": str(artifact_path),
            "sha256": "0" * 64,
            "schema_version": "1.1",
        },
        "testset": {"sha256": testset.sha256},
        "model": "deepseek-chat",
        "configuration": {"metrics": METRICS, "paired_rows_scored": 3},
        "scores": {
            "baseline": {
                metric: {
                    "aggregate": 0.5,
                    "per_sample": {},
                    "failures": [],
                    "judge_reasons": {},
                }
                for metric in METRICS
            },
            "refined": {
                metric: {
                    "aggregate": 0.6,
                    "per_sample": {},
                    "failures": [],
                    "judge_reasons": {},
                }
                for metric in METRICS
            },
        },
        "environment": {"python_version": "3.12"},
        "interpretation": "fixture",
    }
    scores_path = tmp_path / "ragas-scores-mismatched.json"
    scores_path.write_text(json.dumps(scores), encoding="utf-8")
    with pytest.raises(ValueError, match="not produced from this input artifact"):
        run(
            tmp_path / "bad",
            artifact_path=artifact_path,
            scores_path=scores_path,
        )


def test_report_refuses_existing_output(tmp_path: Path) -> None:
    _run_report(
        tmp_path,
        baseline={metric: _uniform(0.5) for metric in METRICS},
        refined={metric: _uniform(0.5) for metric in METRICS},
    )
    with pytest.raises(FileExistsError):
        _run_report(
            tmp_path,
            baseline={metric: _uniform(0.5) for metric in METRICS},
            refined={metric: _uniform(0.5) for metric in METRICS},
            name="report",
        )
