"""Offline Ragas scoring and verification tests using injected scorer seams.

These tests never invoke Ragas or DeepSeek: the scoring stage is exercised
with fake scorer/evaluator seams, and the real ``_RagasMetricScorer`` failure
isolation is tested through injected ``evaluate_fn`` and ``metric_factory``.
"""

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.haystack.ragas_score import (
    ARMS,
    CORE_METRICS,
    MetricOutcome,
    _RagasMetricScorer,
    run,
    verify,
)
from benchmarks.haystack.testset import load_testset


class _FakeScorer:
    """Scoring seam returning fixed scores, or injected metric failures."""

    def __init__(
        self, failures: dict[tuple[str, str], dict[str, object]] | None = None
    ):
        self._failures = failures or {}
        self.asked: list[tuple[str, str]] = []

    def score(
        self,
        metric: str,
        samples: list,
        *,
        arm: str,
    ) -> MetricOutcome:
        self.asked.append((arm, metric))
        key = (arm, metric)
        if key in self._failures:
            return MetricOutcome(
                metric=metric,
                aggregate=None,
                per_sample={},
                failures=(self._failures[key],),
                judge_reasons={},
            )
        per_sample = {sample.qa_id: 0.75 for sample in samples}
        judge_reasons = {
            sample.qa_id: {
                "judge_prompt": {"output": {"verdict": 1, "reason": "fixture"}}
            }
            for sample in samples
        }
        return MetricOutcome(
            metric=metric,
            aggregate=0.75,
            per_sample=per_sample,
            failures=(),
            judge_reasons=judge_reasons,
        )


class _FakeResult:
    """Duck-typed Ragas EvaluationResult exposing scores and traces only."""

    def __init__(self, scores: list[dict], traces: list[dict]) -> None:
        self.scores = scores
        self.traces = traces


class _DummyMetric:
    def __init__(self, name: str) -> None:
        self.name = name


def _dummy_factory(metric: str) -> _DummyMetric:
    return _DummyMetric(metric)


def _write_artifact(tmp_path: Path, rows: list[dict], *, testset_sha256: str) -> Path:
    artifact = {
        "schema_version": "1.1",
        "testset": {
            "path": "frozen",
            "sha256": testset_sha256,
            "corpus_size": 21,
            "qa_pair_count": len(rows),
        },
        "model": "deepseek-chat",
        "api_base_url": "https://api.deepseek.com/v1",
        "profile": "B2-L lexical",
        "generation": {
            "model": "deepseek-chat",
            "api_base_url": "https://api.deepseek.com/v1",
            "temperature": 0,
            "max_tokens": 512,
            "prompt_template": "ragrefine-haystack-prompt-v1",
        },
        "configuration": {
            "top_n": 5,
            "top_k": 3,
            "max_tokens": None,
            "prompt_template": "ragrefine-haystack-prompt-v1",
            "pool_reused": True,
        },
        "review": {
            "status": "human-reviewed",
            "reviewer_role": "human",
            "reviewed_on": "2026-01-01",
        },
        "evaluation": {
            "queries_total": len(rows),
            "paired_query_count": sum(row["paired"] for row in rows),
            "baseline_record_count": sum(row["paired"] for row in rows),
            "refined_record_count": sum(row["paired"] for row in rows),
            "baseline_failures": 0,
            "refined_failures": 0,
        },
        "per_query": rows,
        "metrics": {"names": [], "baseline": {}, "refined": {}},
        "environment": {"python_version": "3.12"},
        "interpretation": "fixture",
    }
    path = tmp_path / "ragas-results.json"
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (tmp_path / "ragas-results.json.sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    return path


def _fixture_rows() -> list[dict]:
    testset = load_testset()
    corpus_ids = [document.id for document in testset.corpus]
    rows: list[dict] = []
    for pair in testset.qa_pairs:
        pool = corpus_ids[:5]
        rows.append(
            {
                "qa_id": pair.id,
                "question": pair.question,
                "reference_answer": pair.reference_answer,
                "reference_context_ids": list(pair.reference_context_ids),
                "pool_ids": pool,
                "pool_sha256": "d" * 64,
                "baseline_context_ids": pool[:3],
                "refined_context_ids": [pool[1], pool[0], pool[2]],
                "refinement_trace": {
                    "stages": [],
                    "duration_ms": 0.0,
                    "config_fingerprint": "f" * 64,
                },
                "baseline_prompt": "Question: fixture",
                "refined_prompt": "Question: fixture",
                "baseline_answer": "fixture baseline answer",
                "refined_answer": "fixture refined answer",
                "baseline_failure": None,
                "refined_failure": None,
                "paired": True,
                "retrieval_latency_seconds": 0.0,
                "refinement_latency_seconds": 0.0,
                "baseline_generation_latency_seconds": 0.0,
                "refined_generation_latency_seconds": 0.0,
            }
        )
    return rows


def _score(
    tmp_path: Path,
    rows: list[dict],
    *,
    name: str = "scores",
    scorer: _FakeScorer | None = None,
    embeddings: object | None = None,
    testset_sha256: str | None = None,
) -> tuple[dict[str, object], Path]:
    testset = load_testset()
    artifact_path = _write_artifact(
        tmp_path, rows, testset_sha256=testset_sha256 or testset.sha256
    )
    result = run(
        tmp_path / name,
        artifact_path=artifact_path,
        api_key="not-used",
        model="deepseek-chat",
        scorer=scorer or _FakeScorer(),
        embeddings=embeddings,
    )
    return result, artifact_path


def test_score_persists_per_sample_aggregate_and_judge_reasons(
    tmp_path: Path,
) -> None:
    rows = _fixture_rows()
    scorer = _FakeScorer()
    result, artifact_path = _score(tmp_path, rows, scorer=scorer)

    assert result["schema_version"] == "1.0"
    assert result["configuration"]["metrics"] == list(CORE_METRICS)
    assert result["configuration"]["omitted_metrics"] == {
        "answer_relevancy": "no pinned embedding provider configured"
    }
    assert result["configuration"]["paired_rows_scored"] == 13
    assert result["configuration"]["unpaired_rows_skipped"] == 0
    assert (
        result["input_artifact"]["sha256"]
        == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    )
    assert result["testset"]["sha256"] == load_testset().sha256

    assert set(scorer.asked) == {
        (arm, metric) for arm in ARMS for metric in CORE_METRICS
    }
    for arm in ARMS:
        for metric in CORE_METRICS:
            block = result["scores"][arm][metric]
            assert block["aggregate"] == 0.75
            assert len(block["per_sample"]) == 13
            assert block["failures"] == []
            assert len(block["judge_reasons"]) == 13
            reason = block["judge_reasons"][next(iter(block["judge_reasons"]))]
            assert reason["judge_prompt"]["output"]["reason"] == "fixture"

    scores_path = tmp_path / "scores" / "ragas-scores.json"
    assert json.loads(scores_path.read_text()) == result
    sidecar = scores_path.with_suffix(".json.sha256")
    assert (
        sidecar.read_text(encoding="utf-8").strip()
        == hashlib.sha256(scores_path.read_bytes()).hexdigest()
    )


def test_score_includes_answer_relevancy_only_with_embeddings(tmp_path: Path) -> None:
    rows = _fixture_rows()
    scorer = _FakeScorer()
    result, _ = _score(tmp_path, rows, scorer=scorer, embeddings=object())

    assert "answer_relevancy" in result["configuration"]["metrics"]
    assert result["configuration"]["omitted_metrics"] == {}
    assert len(scorer.asked) == 2 * 5


def test_score_skips_unpaired_rows(tmp_path: Path) -> None:
    rows = _fixture_rows()
    rows[0]["paired"] = False
    result, _ = _score(tmp_path, rows)

    assert result["configuration"]["paired_rows_scored"] == 12
    assert result["configuration"]["unpaired_rows_skipped"] == 1
    for arm in ARMS:
        per_sample = result["scores"][arm]["faithfulness"]["per_sample"]
        assert len(per_sample) == 12
        assert rows[0]["qa_id"] not in per_sample


def test_metric_failure_is_persisted_never_a_score(tmp_path: Path) -> None:
    rows = _fixture_rows()
    failure = {
        "qa_id": None,
        "failure_type": "RuntimeError",
        "reason": "simulated judge failure",
    }
    scorer = _FakeScorer(failures={("baseline", "faithfulness"): failure})
    result, _ = _score(tmp_path, rows, scorer=scorer)

    block = result["scores"]["baseline"]["faithfulness"]
    assert block["aggregate"] is None
    assert block["per_sample"] == {}
    assert block["failures"] == [failure]
    # Other metrics and the refined arm are unaffected.
    assert result["scores"]["baseline"]["context_precision"]["aggregate"] == 0.75
    assert result["scores"]["refined"]["faithfulness"]["aggregate"] == 0.75


def test_score_rejects_unknown_context_ids(tmp_path: Path) -> None:
    rows = _fixture_rows()
    rows[0]["baseline_context_ids"][0] = "ghost-document-id"
    with pytest.raises(ValueError, match="unknown corpus IDs"):
        _score(tmp_path, rows)


def test_score_rejects_incomplete_paired_row(tmp_path: Path) -> None:
    rows = _fixture_rows()
    rows[0]["refined_answer"] = None
    with pytest.raises(ValueError, match="non-empty answers"):
        _score(tmp_path, rows)


def test_score_rejects_testset_mismatch(tmp_path: Path) -> None:
    rows = _fixture_rows()
    with pytest.raises(ValueError, match="does not match the frozen test set"):
        _score(tmp_path, rows, testset_sha256="0" * 64)


def test_score_refuses_existing_output(tmp_path: Path) -> None:
    rows = _fixture_rows()
    _score(tmp_path, rows)
    with pytest.raises(FileExistsError):
        _score(tmp_path, rows, name="scores")


class TestRagasMetricScorerIsolation:
    """Real scorer failure isolation without Ragas or DeepSeek."""

    def _scorer(self, evaluate_fn) -> _RagasMetricScorer:
        return _RagasMetricScorer(
            api_key="not-used",
            model="deepseek-chat",
            evaluate_fn=evaluate_fn,
            metric_factory=_dummy_factory,
        )

    def _sample(self) -> list:
        from benchmarks.haystack.ragas_score import ScoreSample

        return [
            ScoreSample(
                qa_id="q-01",
                question="question",
                retrieved_contexts=("context",),
                retrieved_context_ids=("id-1",),
                reference_context_ids=("id-1",),
                reference="reference",
                answer="answer",
            )
        ]

    def test_success_extracts_scores_and_judge_reasons(self) -> None:
        def evaluate_fn(**kwargs) -> _FakeResult:
            return _FakeResult(
                scores=[{"faithfulness": 0.9}],
                traces=[
                    {
                        "faithfulness": {
                            "judge_prompt": {
                                "input": {},
                                "output": {"verdict": 1, "reason": "grounded"},
                            }
                        }
                    }
                ],
            )

        outcome = self._scorer(evaluate_fn).score(
            "faithfulness", self._sample(), arm="baseline"
        )
        assert outcome.per_sample == {"q-01": 0.9}
        assert outcome.aggregate == 0.9
        assert outcome.failures == ()
        assert outcome.judge_reasons == {
            "q-01": {"judge_prompt": {"output": {"verdict": 1, "reason": "grounded"}}}
        }

    def test_evaluate_failure_is_preserved_not_a_score(self) -> None:
        def evaluate_fn(**kwargs) -> _FakeResult:
            raise RuntimeError("simulated judge failure")

        outcome = self._scorer(evaluate_fn).score(
            "faithfulness", self._sample(), arm="baseline"
        )
        assert outcome.per_sample == {}
        assert outcome.aggregate is None
        assert outcome.failures == (
            {
                "qa_id": None,
                "failure_type": "RuntimeError",
                "reason": "simulated judge failure",
            },
        )

    def test_non_finite_value_is_a_failure_not_a_score(self) -> None:
        def evaluate_fn(**kwargs) -> _FakeResult:
            return _FakeResult(scores=[{"faithfulness": float("nan")}], traces=[{}])

        outcome = self._scorer(evaluate_fn).score(
            "faithfulness", self._sample(), arm="baseline"
        )
        assert outcome.per_sample == {}
        assert outcome.aggregate is None
        assert outcome.failures[0]["failure_type"] == "non_finite_metric_value"

    def test_unsupported_metric_is_rejected(self) -> None:
        scorer = _RagasMetricScorer(
            api_key="not-used",
            model="deepseek-chat",
            evaluate_fn=lambda **kwargs: _FakeResult(scores=[{}], traces=[{}]),
            metric_factory=_dummy_factory,
        )
        with pytest.raises(ValueError, match="unsupported Ragas metric"):
            scorer.score("not_a_metric", self._sample(), arm="baseline")


def test_verify_passes_for_valid_scores(tmp_path: Path) -> None:
    rows = _fixture_rows()
    _score(tmp_path, rows)
    report = verify(
        artifact_path=tmp_path / "ragas-results.json",
        scores_path=tmp_path / "scores" / "ragas-scores.json",
    )
    assert report["ok"] is True
    assert report["failures"] == []
    assert {
        "testset_checksum",
        "input_artifact_structure",
        "input_testset_link",
        "input_artifact_checksum",
        "scores_checksum",
        "scores_schema",
        "scores_input_link",
        "scores_coverage",
    } <= set(report["checks"])


def test_verify_without_scores_validates_input_only(tmp_path: Path) -> None:
    rows = _fixture_rows()
    _score(tmp_path, rows)
    report = verify(artifact_path=tmp_path / "ragas-results.json")
    assert report["ok"] is True
    assert "scores_checksum" not in report["checks"]


def test_verify_detects_scores_checksum_corruption(tmp_path: Path) -> None:
    rows = _fixture_rows()
    _score(tmp_path, rows)
    scores_path = tmp_path / "scores" / "ragas-scores.json"
    scores_path.write_text(
        scores_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    report = verify(
        artifact_path=tmp_path / "ragas-results.json",
        scores_path=scores_path,
    )
    assert report["ok"] is False
    assert "scores_checksum" in {item["check"] for item in report["failures"]}


def test_verify_detects_input_corruption(tmp_path: Path) -> None:
    rows = _fixture_rows()
    _score(tmp_path, rows)
    artifact_path = tmp_path / "ragas-results.json"
    artifact_path.write_text(
        artifact_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    report = verify(artifact_path=artifact_path)
    assert report["ok"] is False
    assert "input_artifact_checksum" in {item["check"] for item in report["failures"]}


def test_verify_detects_scores_input_link_mismatch(tmp_path: Path) -> None:
    rows = _fixture_rows()
    _score(tmp_path, rows)
    artifact_path = tmp_path / "ragas-results.json"
    # Rewrite the input with a fresh sidecar so only the input->scores link breaks.
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    data["interpretation"] = "modified after scoring"
    artifact_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (tmp_path / "ragas-results.json.sha256").write_text(
        hashlib.sha256(artifact_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
    report = verify(
        artifact_path=artifact_path,
        scores_path=tmp_path / "scores" / "ragas-scores.json",
    )
    assert report["ok"] is False
    assert "scores_input_link" in {item["check"] for item in report["failures"]}
