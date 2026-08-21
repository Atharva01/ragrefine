"""Ragas evaluation harness tests using injected generator/evaluator seams."""

import hashlib
import json
from pathlib import Path

import pytest

haystack = pytest.importorskip("haystack")
ChatMessage = haystack.dataclasses.ChatMessage

from benchmarks.haystack.ragas_eval import _default_model, run  # noqa: E402


class _FakeGenerator:
    def run(self, messages: list[ChatMessage]) -> dict[str, list[ChatMessage]]:
        assert messages[0].text
        return {"replies": [ChatMessage.from_assistant("fixture answer")]}


class _FlakyGenerator:
    """Fails the nth generator call to exercise per-arm failure persistence."""

    def __init__(self, fail_on_call: int) -> None:
        self._calls = 0
        self._fail_on_call = fail_on_call

    def run(self, messages: list[ChatMessage]) -> dict[str, list[ChatMessage]]:
        self._calls += 1
        if self._calls == self._fail_on_call:
            raise RuntimeError("simulated generation failure")
        return {"replies": [ChatMessage.from_assistant("fixture answer")]}


class _FakeEvaluator:
    METRICS = ["faithfulness", "context_recall", "factual_correctness"]

    def evaluate(self, records: list) -> dict[str, object]:
        assert records, "evaluator must receive at least one record"
        return {
            "metrics": self.METRICS,
            "aggregate": {name: 1.0 for name in self.METRICS},
            "per_record": [{name: 1.0 for name in self.METRICS} for _ in records],
        }


def _run(
    tmp_path: Path,
    name: str = "eval",
    generator: object | None = None,
) -> dict[str, object]:
    return run(
        tmp_path / name,
        api_key="not-used",
        model="deepseek-chat",
        generator=generator or _FakeGenerator(),
        evaluator=_FakeEvaluator(),
    )


def test_ragas_eval_records_shared_pools_and_metrics(tmp_path: Path) -> None:
    result = _run(tmp_path)

    assert result["schema_version"] == "1.1"
    assert result["configuration"]["pool_reused"] is True
    assert result["configuration"]["top_k"] == 3
    assert result["metrics"]["names"] == _FakeEvaluator.METRICS
    assert set(result["metrics"]["baseline"]) == set(_FakeEvaluator.METRICS)
    assert result["testset"]["qa_pair_count"] == 13
    assert result["configuration"]["max_tokens"] is None
    assert result["configuration"]["prompt_template"] == "ragrefine-haystack-prompt-v1"
    assert result["review"]["status"] == "human-reviewed"
    assert result["evaluation"]["queries_total"] == 13
    assert result["evaluation"]["paired_query_count"] == 13
    assert result["evaluation"]["baseline_failures"] == 0
    assert result["evaluation"]["refined_failures"] == 0
    assert result["generation"]["model"] == "deepseek-chat"
    assert result["generation"]["api_base_url"].endswith("/v1")
    assert result["generation"]["temperature"] == 0

    assert len(result["per_query"]) == 13
    for row in result["per_query"]:
        pool_ids = row["pool_ids"]
        assert row["baseline_context_ids"] == pool_ids[:3]
        assert set(row["refined_context_ids"]) <= set(pool_ids)
        assert row["pool_sha256"]
        assert row["paired"] is True
        assert row["baseline_failure"] is None
        assert row["refined_failure"] is None
        assert row["baseline_answer"] == "fixture answer"
        assert row["refined_answer"] == "fixture answer"
        assert row["refinement_trace"]["stages"]
        assert row["refinement_trace"]["config_fingerprint"]
        assert row["retrieval_latency_seconds"] >= 0
        assert row["refinement_latency_seconds"] >= 0
        assert set(row["baseline_scores"]) == set(_FakeEvaluator.METRICS)
        assert set(row["refined_scores"]) == set(_FakeEvaluator.METRICS)

    persisted = json.loads((tmp_path / "eval" / "ragas-results.json").read_text())
    assert persisted == result
    artifact_path = tmp_path / "eval" / "ragas-results.json"
    sidecar = artifact_path.with_suffix(".json.sha256")
    assert (
        sidecar.read_text(encoding="utf-8").strip()
        == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    )


def test_ragas_eval_persists_exact_prompts_for_both_arms(tmp_path: Path) -> None:
    result = _run(tmp_path)

    for row in result["per_query"]:
        assert row["baseline_prompt"].startswith("Question:")
        assert row["refined_prompt"].startswith("Question:")
        for document_id in row["baseline_context_ids"]:
            assert f"[{document_id}]" in row["baseline_prompt"]
        for document_id in row["refined_context_ids"]:
            assert f"[{document_id}]" in row["refined_prompt"]
        # The refinement trace records the exact stage names with timing.
        stage_names = [stage["name"] for stage in row["refinement_trace"]["stages"]]
        assert "lexical_ranking" in stage_names
        assert "context_selection" in stage_names


def test_ragas_eval_is_deterministic_except_latency(tmp_path: Path) -> None:
    first = _run(tmp_path, "a")
    second = _run(tmp_path, "b")

    def _stable(result: dict[str, object]) -> dict[str, object]:
        per_query: list[dict[str, object]] = []
        for row in result["per_query"]:
            stable = {
                key: value
                for key, value in row.items()
                if "latency" not in key and key != "refinement_trace"
            }
            stable["trace_stages"] = [
                stage["name"] for stage in row["refinement_trace"]["stages"]
            ]
            per_query.append(stable)
        return {"metrics": result["metrics"], "per_query": per_query}

    assert _stable(first) == _stable(second)


def test_ragas_eval_persists_per_arm_failures_and_keeps_pairing(
    tmp_path: Path,
) -> None:
    # Call 2 is the refined arm of the first query.
    result = _run(tmp_path, "flaky", generator=_FlakyGenerator(fail_on_call=2))

    assert result["evaluation"]["queries_total"] == 13
    assert result["evaluation"]["paired_query_count"] == 12
    assert result["evaluation"]["baseline_failures"] == 0
    assert result["evaluation"]["refined_failures"] == 1

    failed_row = result["per_query"][0]
    assert failed_row["paired"] is False
    assert failed_row["baseline_answer"] == "fixture answer"
    assert failed_row["refined_answer"] is None
    assert failed_row["baseline_failure"] is None
    assert failed_row["refined_failure"] == {
        "failure_type": "RuntimeError",
        "reason": "simulated generation failure",
    }
    # The prompt is still persisted for the failed arm.
    assert failed_row["refined_prompt"].startswith("Question:")
    assert "refined_scores" not in failed_row

    for row in result["per_query"][1:]:
        assert row["paired"] is True
        assert "baseline_scores" in row
        assert "refined_scores" in row


def test_ragas_eval_rejects_existing_output(tmp_path: Path) -> None:
    _run(tmp_path)
    with pytest.raises(FileExistsError):
        _run(tmp_path)


def test_ragas_eval_rejects_invalid_top_k(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run(
            tmp_path / "bad",
            api_key="x",
            model="m",
            generator=_FakeGenerator(),
            evaluator=_FakeEvaluator(),
            top_k=6,
        )


def test_ragas_eval_rejects_changed_frozen_settings(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen evaluation contract"):
        run(
            tmp_path / "changed-model",
            api_key="x",
            model="different-model",
            generator=_FakeGenerator(),
            evaluator=_FakeEvaluator(),
        )
    with pytest.raises(ValueError, match="Top-N/Top-K"):
        run(
            tmp_path / "changed-top-n",
            api_key="x",
            model="deepseek-chat",
            generator=_FakeGenerator(),
            evaluator=_FakeEvaluator(),
            top_n=4,
        )


def test_default_model_is_environment_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-coder")
    assert _default_model() == "deepseek-coder"
    monkeypatch.delenv("DEEPSEEK_MODEL")
    assert _default_model() == "deepseek-chat"
