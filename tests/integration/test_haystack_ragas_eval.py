"""Ragas evaluation harness tests using injected generator/evaluator seams."""

import json
from pathlib import Path

import pytest

haystack = pytest.importorskip("haystack")
ChatMessage = haystack.dataclasses.ChatMessage

from benchmarks.haystack.ragas_eval import run  # noqa: E402


class _FakeGenerator:
    def run(self, messages: list[ChatMessage]) -> dict[str, list[ChatMessage]]:
        assert messages[0].text
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


def _run(tmp_path: Path, name: str = "eval") -> dict[str, object]:
    return run(
        tmp_path / name,
        api_key="not-used",
        model="deepseek-chat",
        generator=_FakeGenerator(),
        evaluator=_FakeEvaluator(),
    )


def test_ragas_eval_records_shared_pools_and_metrics(tmp_path: Path) -> None:
    result = _run(tmp_path)

    assert result["configuration"]["pool_reused"] is True
    assert result["configuration"]["top_k"] == 3
    assert result["metrics"]["names"] == _FakeEvaluator.METRICS
    assert set(result["metrics"]["baseline"]) == set(_FakeEvaluator.METRICS)
    assert result["testset"]["qa_pair_count"] == 13
    assert result["configuration"]["max_tokens"] is None
    assert result["configuration"]["prompt_template"] == "ragrefine-haystack-prompt-v1"
    assert result["review"]["status"] == "human-reviewed"

    assert len(result["per_query"]) == 13
    for row in result["per_query"]:
        pool_ids = row["pool_ids"]
        assert row["baseline_context_ids"] == pool_ids[:3]
        assert set(row["refined_context_ids"]) <= set(pool_ids)
        assert row["refinement_trace_stages"]
        assert set(row["baseline_scores"]) == set(_FakeEvaluator.METRICS)
        assert set(row["refined_scores"]) == set(_FakeEvaluator.METRICS)

    persisted = json.loads((tmp_path / "eval" / "ragas-results.json").read_text())
    assert persisted == result


def test_ragas_eval_is_deterministic_except_latency(tmp_path: Path) -> None:
    first = _run(tmp_path, "a")
    second = _run(tmp_path, "b")

    def _stable(result: dict[str, object]) -> dict[str, object]:
        return {
            "metrics": result["metrics"],
            "per_query": [
                {key: value for key, value in row.items() if "latency" not in key}
                for row in result["per_query"]
            ],
        }

    assert _stable(first) == _stable(second)


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
