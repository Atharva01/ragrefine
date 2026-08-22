"""Offline Ragas scoring and verification of persisted paired Haystack artifacts.

Consumes the artifact persisted by ``benchmarks.haystack.ragas_eval`` (schema
1.1) plus the frozen human-reviewed test set, and scores both arms with Ragas
without regenerating retrieval, context, or answers.

Metrics:

* ``context_precision`` — Context Precision with Reference (LLM);
* ``id_based_context_recall`` — ID-based Context Recall (non-LLM);
* ``faithfulness`` — Faithfulness (LLM);
* ``factual_correctness`` — Factual Correctness (LLM);
* ``answer_relevancy`` — added only when an explicit, pinned embedding
  provider is configured through ``RAGAS_EMBEDDING_MODEL``,
  ``RAGAS_EMBEDDING_BASE_URL``, and ``RAGAS_EMBEDDING_API_KEY``.

DeepSeek is configured only through ``DEEPSEEK_API_KEY`` and
``DEEPSEEK_MODEL`` against its OpenAI-compatible endpoint. Metric errors are
persisted as failures and are never converted into scores. A query is scored
only when both arms generated an answer in the consumed artifact.

The ``verify`` subcommand validates checksums and artifact consistency without
re-scoring. This is an evaluation harness only; it does not, by itself,
establish that refinement improves generation or context quality.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from benchmarks.haystack.testset import DEFAULT_TESTSET_PATH, load_testset

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
SCHEMA_VERSION = "1.0"
INPUT_SCHEMA_VERSION = "1.1"
CORE_METRICS = (
    "context_precision",
    "id_based_context_recall",
    "faithfulness",
    "factual_correctness",
)
OPTIONAL_METRICS = ("answer_relevancy",)
ARMS = ("baseline", "refined")


@dataclass(frozen=True, slots=True)
class ScoreSample:
    """One arm's Ragas sample derived from the persisted paired artifact."""

    qa_id: str
    question: str
    retrieved_contexts: tuple[str, ...]
    retrieved_context_ids: tuple[str, ...]
    reference_context_ids: tuple[str, ...]
    reference: str
    answer: str


@dataclass(frozen=True, slots=True)
class MetricOutcome:
    """One metric's result for one arm: scores, failures, and judge reasons.

    ``per_sample`` maps QA IDs to finite scores only. ``failures`` records
    metric-level (``qa_id`` is None) and per-sample failures; an error is
    never converted into a score.
    """

    metric: str
    aggregate: float | None
    per_sample: Mapping[str, float]
    failures: tuple[dict[str, object], ...]
    judge_reasons: Mapping[str, object]


class MetricScorer(Protocol):
    """Scoring seam: score one metric over one arm's samples."""

    def score(
        self,
        metric: str,
        samples: Sequence[ScoreSample],
        *,
        arm: str,
    ) -> MetricOutcome: ...


@dataclass(frozen=True, slots=True)
class ScoreRow:
    """One paired query from the persisted artifact that can be scored."""

    qa_id: str
    question: str
    reference_answer: str
    reference_context_ids: tuple[str, ...]
    baseline_context_ids: tuple[str, ...]
    refined_context_ids: tuple[str, ...]
    baseline_answer: str
    refined_answer: str


class _RagasMetricScorer:
    """Ragas scorer with per-metric failure isolation.

    Evaluates one metric per arm with ``raise_exceptions=True`` so a failed
    sample aborts that metric rather than leaving an exception inside a score.
    ``evaluate_fn`` and ``metric_factory`` are injectable seams for tests; the
    defaults load Ragas lazily so this module never requires Ragas at import.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        embeddings: object | None = None,
        evaluate_fn: Callable[..., object] | None = None,
        metric_factory: Callable[[str], object] | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._embeddings = embeddings
        self._evaluate_fn = evaluate_fn
        self._metric_factory = metric_factory or self._default_metric_factory
        self._llm = None

    def score(
        self,
        metric: str,
        samples: Sequence[ScoreSample],
        *,
        arm: str,
    ) -> MetricOutcome:
        _validate_metric(metric)
        from ragas import EvaluationDataset, evaluate  # type: ignore[import-not-found]

        metric_obj = self._metric_factory(metric)
        dataset = EvaluationDataset.from_list(
            [
                {
                    "user_input": sample.question,
                    "retrieved_contexts": list(sample.retrieved_contexts),
                    "retrieved_context_ids": list(sample.retrieved_context_ids),
                    "reference_context_ids": list(sample.reference_context_ids),
                    "response": sample.answer,
                    "reference": sample.reference,
                }
                for sample in samples
            ]
        )
        evaluate_fn = self._evaluate_fn or evaluate
        llm = None if self._evaluate_fn is not None else self._llm_wrapper()
        try:
            result = evaluate_fn(
                dataset=dataset,
                metrics=[metric_obj],
                llm=llm,
                embeddings=self._embeddings,
                raise_exceptions=True,
            )
        except Exception as error:
            return MetricOutcome(
                metric=metric,
                aggregate=None,
                per_sample={},
                failures=(
                    {
                        "qa_id": None,
                        "failure_type": type(error).__name__,
                        "reason": str(error),
                    },
                ),
                judge_reasons={},
            )
        return self._extract(metric, metric_obj, samples, result)

    @staticmethod
    def _extract(
        metric: str,
        metric_obj: object,
        samples: Sequence[ScoreSample],
        result: object,
    ) -> MetricOutcome:
        score_key = _score_key(metric_obj)
        trace_name = getattr(metric_obj, "name", metric)
        per_sample: dict[str, float] = {}
        failures: list[dict[str, object]] = []
        judge_reasons: dict[str, object] = {}
        for index, sample in enumerate(samples):
            value = _score_value(result, index, score_key)
            if value is None or not _is_finite(value):
                failures.append(
                    {
                        "qa_id": sample.qa_id,
                        "failure_type": "non_finite_metric_value",
                        "reason": repr(value),
                    }
                )
            else:
                per_sample[sample.qa_id] = float(value)
            outputs = _judge_outputs(result, index, trace_name)
            if outputs:
                judge_reasons[sample.qa_id] = outputs
        aggregate = sum(per_sample.values()) / len(per_sample) if per_sample else None
        return MetricOutcome(
            metric=metric,
            aggregate=aggregate,
            per_sample=per_sample,
            failures=tuple(failures),
            judge_reasons=judge_reasons,
        )

    def _llm_wrapper(self) -> object:
        if self._llm is None:
            from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]
            from ragas.llms import (  # type: ignore[import-not-found]
                LangchainLLMWrapper,
            )

            llm = ChatOpenAI(
                model=self._model,
                api_key=self._api_key,
                base_url=DEEPSEEK_BASE_URL,
                temperature=0,
            )
            self._llm = LangchainLLMWrapper(llm)
        return self._llm

    @staticmethod
    def _default_metric_factory(metric: str) -> object:
        _validate_metric(metric)
        from ragas.metrics import (  # type: ignore[import-not-found]
            AnswerRelevancy,
            ContextPrecision,
            FactualCorrectness,
            Faithfulness,
            IDBasedContextRecall,
        )

        factories: dict[str, Callable[[], object]] = {
            "context_precision": ContextPrecision,
            "id_based_context_recall": IDBasedContextRecall,
            "faithfulness": Faithfulness,
            "factual_correctness": FactualCorrectness,
            "answer_relevancy": AnswerRelevancy,
        }
        return factories[metric]()


def run(
    output_dir: Path,
    *,
    artifact_path: Path,
    api_key: str,
    model: str,
    scorer: MetricScorer | None = None,
    testset_path: Path = DEFAULT_TESTSET_PATH,
    embeddings: object | None = None,
) -> dict[str, object]:
    """Score persisted paired outputs for both arms and persist the artifact."""
    if output_dir.exists():
        raise FileExistsError(f"Ragas score output already exists: {output_dir}")
    testset = load_testset(testset_path)
    artifact, rows, artifact_sha256, skipped = _load_artifact(
        artifact_path, {document.id for document in testset.corpus}
    )
    if artifact["testset"]["sha256"] != testset.sha256:
        raise ValueError(
            "artifact test-set checksum does not match the frozen test set"
        )
    corpus_by_id = {document.id: document.content for document in testset.corpus}

    metrics: list[str] = list(CORE_METRICS)
    omitted: dict[str, str] = {}
    if embeddings is not None:
        metrics.extend(OPTIONAL_METRICS)
    else:
        omitted["answer_relevancy"] = "no pinned embedding provider configured"

    active_scorer = scorer or _RagasMetricScorer(
        api_key=api_key, model=model, embeddings=embeddings
    )
    samples = {arm: _samples_for_arm(rows, corpus_by_id, arm) for arm in ARMS}
    scores: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        arm_block: dict[str, object] = {}
        for metric in metrics:
            outcome = active_scorer.score(metric, samples[arm], arm=arm)
            arm_block[metric] = _metric_block(outcome)
        scores[arm] = arm_block

    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "input_artifact": {
            "path": str(artifact_path),
            "sha256": artifact_sha256,
            "schema_version": artifact["schema_version"],
        },
        "testset": {
            "path": str(testset_path),
            "sha256": testset.sha256,
            "corpus_size": len(testset.corpus),
        },
        "model": model,
        "configuration": {
            "metrics": metrics,
            "omitted_metrics": omitted,
            "paired_rows_scored": len(rows),
            "unpaired_rows_skipped": skipped,
            "input_pool_reused": True,
        },
        "scores": scores,
        "environment": _environment(model, embeddings),
        "interpretation": (
            "Ragas scoring of the persisted paired artifact only; a refined-"
            "branch advantage is reported only when the recorded aggregate and "
            "per-query metrics support it. Metric errors are persisted as "
            "failures and are never converted into scores."
        ),
    }
    output_dir.mkdir(parents=True)
    output_path = output_dir / "ragas-scores.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "ragas-scores.json.sha256").write_text(
        hashlib.sha256(output_path.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    return result


def verify(
    *,
    artifact_path: Path,
    scores_path: Path | None = None,
    testset_path: Path = DEFAULT_TESTSET_PATH,
) -> dict[str, object]:
    """Validate checksums and artifact consistency without re-scoring."""
    failures: list[dict[str, object]] = []
    checks: list[str] = []

    try:
        testset = load_testset(testset_path)
    except (OSError, ValueError) as error:
        failures.append(
            {
                "check": "testset_checksum",
                "failure_type": type(error).__name__,
                "reason": str(error),
            }
        )
        return {"ok": False, "checks": checks, "failures": failures}
    checks.append("testset_checksum")

    try:
        artifact, rows, artifact_sha256, skipped = _load_artifact(
            artifact_path, {document.id for document in testset.corpus}
        )
    except (OSError, ValueError) as error:
        failures.append(
            {
                "check": "input_artifact_structure",
                "failure_type": type(error).__name__,
                "reason": str(error),
            }
        )
        return {"ok": False, "checks": checks, "failures": failures}
    checks.append("input_artifact_structure")

    if artifact["testset"]["sha256"] != testset.sha256:
        failures.append(
            {
                "check": "input_testset_link",
                "failure_type": "checksum_mismatch",
                "reason": (
                    "artifact test-set checksum does not match the frozen test set"
                ),
            }
        )
    else:
        checks.append("input_testset_link")

    sidecar = artifact_path.with_suffix(artifact_path.suffix + ".sha256")
    if not sidecar.exists():
        failures.append(
            {
                "check": "input_artifact_checksum",
                "failure_type": "sidecar_missing",
                "reason": f"checksum sidecar not found: {sidecar}",
            }
        )
    elif sidecar.read_text(encoding="utf-8").strip() != artifact_sha256:
        failures.append(
            {
                "check": "input_artifact_checksum",
                "failure_type": "checksum_mismatch",
                "reason": "input artifact bytes do not match its SHA-256 sidecar",
            }
        )
    else:
        checks.append("input_artifact_checksum")

    if scores_path is not None:
        _verify_scores(scores_path, artifact_sha256, rows, failures, checks)

    return {"ok": not failures, "checks": checks, "failures": failures}


def _verify_scores(
    scores_path: Path,
    artifact_sha256: str,
    rows: Sequence[ScoreRow],
    failures: list[dict[str, object]],
    checks: list[str],
) -> None:
    sidecar = scores_path.with_suffix(scores_path.suffix + ".sha256")
    if not sidecar.exists():
        failures.append(
            {
                "check": "scores_checksum",
                "failure_type": "sidecar_missing",
                "reason": f"checksum sidecar not found: {sidecar}",
            }
        )
        return
    try:
        payload = scores_path.read_bytes()
        scores = json.loads(payload.decode("utf-8"))
    except (OSError, ValueError) as error:
        failures.append(
            {
                "check": "scores_readable",
                "failure_type": type(error).__name__,
                "reason": str(error),
            }
        )
        return
    if (
        sidecar.read_text(encoding="utf-8").strip()
        != hashlib.sha256(payload).hexdigest()
    ):
        failures.append(
            {
                "check": "scores_checksum",
                "failure_type": "checksum_mismatch",
                "reason": "scores artifact bytes do not match their SHA-256 sidecar",
            }
        )
    else:
        checks.append("scores_checksum")
    if not isinstance(scores, dict) or scores.get("schema_version") != SCHEMA_VERSION:
        failures.append(
            {
                "check": "scores_schema",
                "failure_type": "schema_mismatch",
                "reason": f"scores artifact must use schema version {SCHEMA_VERSION}",
            }
        )
        return
    checks.append("scores_schema")

    recorded = scores.get("input_artifact")
    if not isinstance(recorded, dict) or recorded.get("sha256") != artifact_sha256:
        failures.append(
            {
                "check": "scores_input_link",
                "failure_type": "sha256_mismatch",
                "reason": "scores artifact was not produced from this input artifact",
            }
        )
    else:
        checks.append("scores_input_link")

    configuration = scores.get("configuration")
    if not isinstance(configuration, dict):
        failures.append(
            {
                "check": "scores_coverage",
                "failure_type": "missing_configuration",
                "reason": "scores artifact has no configuration block",
            }
        )
        return
    paired_ids = {row.qa_id for row in rows}
    if configuration.get("paired_rows_scored") != len(rows):
        failures.append(
            {
                "check": "scores_coverage",
                "failure_type": "query_count_mismatch",
                "reason": "scored query count does not match the input artifact",
            }
        )
    arm_blocks = scores.get("scores")
    if not isinstance(arm_blocks, dict) or not set(arm_blocks) >= set(ARMS):
        failures.append(
            {
                "check": "scores_coverage",
                "failure_type": "missing_arm",
                "reason": "scores artifact must cover both baseline and refined arms",
            }
        )
        return
    metrics = configuration.get("metrics")
    if not isinstance(metrics, list):
        failures.append(
            {
                "check": "scores_coverage",
                "failure_type": "missing_metrics",
                "reason": "scores artifact has no metric list",
            }
        )
        return
    for arm in ARMS:
        arm_block = arm_blocks[arm]
        if not isinstance(arm_block, dict):
            failures.append(
                {
                    "check": "scores_coverage",
                    "failure_type": "malformed_arm",
                    "reason": f"{arm} arm has no metric blocks",
                }
            )
            continue
        for metric in metrics:
            block = arm_block.get(metric)
            if not isinstance(block, dict):
                failures.append(
                    {
                        "check": "scores_coverage",
                        "failure_type": "missing_metric",
                        "reason": f"{arm}/{metric} has no score block",
                    }
                )
                continue
            covered = set(block.get("per_sample", {}))
            covered.update(
                item["qa_id"]
                for item in block.get("failures", [])
                if isinstance(item, dict) and item.get("qa_id") is not None
            )
            if covered != paired_ids:
                failures.append(
                    {
                        "check": "scores_coverage",
                        "failure_type": "qa_id_mismatch",
                        "reason": (
                            f"{arm}/{metric} does not cover exactly the paired "
                            "queries of the input artifact"
                        ),
                    }
                )
    if not any(item["check"] == "scores_coverage" for item in failures):
        checks.append("scores_coverage")


def _load_artifact(
    artifact_path: Path, corpus_ids: set[str]
) -> tuple[dict[str, object], tuple[ScoreRow, ...], str, int]:
    """Validate the persisted paired artifact and return scorable rows."""
    if not artifact_path.is_file():
        raise ValueError(f"artifact not found: {artifact_path}")
    artifact: object = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError("artifact must be a JSON object")
    if artifact.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported artifact schema version: {artifact.get('schema_version')!r}"
        )
    testset_block = artifact.get("testset")
    if (
        not isinstance(testset_block, dict)
        or not isinstance(testset_block.get("sha256"), str)
        or not testset_block["sha256"]
    ):
        raise ValueError("artifact must record its test-set SHA-256")
    rows_data = artifact.get("per_query")
    if not isinstance(rows_data, list) or not rows_data:
        raise ValueError("artifact must contain a non-empty per_query list")

    rows: list[ScoreRow] = []
    skipped = 0
    seen_ids: set[str] = set()
    for row in rows_data:
        if not isinstance(row, dict):
            raise ValueError("each per_query entry must be an object")
        qa_id = row.get("qa_id")
        if not isinstance(qa_id, str) or not qa_id:
            raise ValueError("per_query qa_id must be a non-empty string")
        if qa_id in seen_ids:
            raise ValueError(f"duplicate per_query qa_id: {qa_id!r}")
        seen_ids.add(qa_id)
        _require_strings(row, "question")
        _require_strings(row, "reference_answer")
        reference_ids = _require_id_list(row, "reference_context_ids", corpus_ids)
        baseline_ids = _require_id_list(row, "baseline_context_ids", corpus_ids)
        refined_ids = _require_id_list(row, "refined_context_ids", corpus_ids)
        if row.get("paired") is not True:
            skipped += 1
            continue
        baseline_answer = row.get("baseline_answer")
        refined_answer = row.get("refined_answer")
        if (
            not isinstance(baseline_answer, str)
            or not baseline_answer
            or not isinstance(refined_answer, str)
            or not refined_answer
        ):
            raise ValueError(
                f"paired query {qa_id!r} must have non-empty answers for both arms"
            )
        rows.append(
            ScoreRow(
                qa_id=qa_id,
                question=row["question"],
                reference_answer=row["reference_answer"],
                reference_context_ids=reference_ids,
                baseline_context_ids=baseline_ids,
                refined_context_ids=refined_ids,
                baseline_answer=baseline_answer,
                refined_answer=refined_answer,
            )
        )
    return artifact, tuple(rows), _file_sha256(artifact_path), skipped


def _require_strings(row: Mapping[str, object], key: str) -> None:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"per_query {key} must be a non-empty string")


def _require_id_list(
    row: Mapping[str, object], key: str, corpus_ids: set[str]
) -> tuple[str, ...]:
    value = row.get(key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError(f"per_query {key} must be a non-empty list of strings")
    unknown = [item for item in value if item not in corpus_ids]
    if unknown:
        raise ValueError(f"per_query {key} references unknown corpus IDs: {unknown!r}")
    return tuple(value)


def _samples_for_arm(
    rows: Sequence[ScoreRow],
    corpus_by_id: Mapping[str, str],
    arm: str,
) -> tuple[ScoreSample, ...]:
    return tuple(
        ScoreSample(
            qa_id=row.qa_id,
            question=row.question,
            retrieved_context_ids=_arm_ids(row, arm),
            retrieved_contexts=tuple(
                corpus_by_id[document_id] for document_id in _arm_ids(row, arm)
            ),
            reference_context_ids=row.reference_context_ids,
            reference=row.reference_answer,
            answer=_arm_answer(row, arm),
        )
        for row in rows
    )


def _arm_ids(row: ScoreRow, arm: str) -> tuple[str, ...]:
    return row.baseline_context_ids if arm == "baseline" else row.refined_context_ids


def _arm_answer(row: ScoreRow, arm: str) -> str:
    return row.baseline_answer if arm == "baseline" else row.refined_answer


def _metric_block(outcome: MetricOutcome) -> dict[str, object]:
    return {
        "aggregate": outcome.aggregate,
        "per_sample": dict(outcome.per_sample),
        "failures": [dict(item) for item in outcome.failures],
        "judge_reasons": outcome.judge_reasons,
    }


def _validate_metric(metric: str) -> None:
    if metric not in (*CORE_METRICS, *OPTIONAL_METRICS):
        raise ValueError(f"unsupported Ragas metric: {metric!r}")


def _score_key(metric_obj: object) -> str:
    """Return the Ragas score-column key for one metric object.

    Ragas keys mode-parameterised metrics (e.g. ``FactualCorrectness``) as
    ``"factual_correctness(mode=f1)"``; everything else uses the plain name.
    """
    name = getattr(metric_obj, "name", "")
    mode = getattr(metric_obj, "mode", None)
    return f"{name}(mode={mode})" if mode is not None else name


def _score_value(result: object, index: int, metric_name: str) -> object:
    scores = getattr(result, "scores", None)
    if not isinstance(scores, list) or index >= len(scores):
        return None
    row = scores[index]
    if not isinstance(row, dict):
        return None
    return row.get(metric_name)


def _judge_outputs(result: object, index: int, metric_name: str) -> dict[str, object]:
    traces = getattr(result, "traces", None)
    if not isinstance(traces, list) or index >= len(traces):
        return {}
    row_trace = traces[index]
    metric_trace = row_trace.get(metric_name, {}) if isinstance(row_trace, dict) else {}
    outputs: dict[str, object] = {}
    for prompt_name, prompt_data in metric_trace.items():
        if not isinstance(prompt_data, dict):
            continue
        outputs[prompt_name] = {"output": _json_safe(prompt_data.get("output"))}
    return outputs


def _is_finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _json_safe(value: object) -> object:
    """Convert Ragas trace objects into JSON round-trip stable structures."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    return str(value)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _default_model() -> str:
    """Return the DeepSeek model name configured only through the environment."""
    return os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")


def _embeddings_from_env() -> object | None:
    """Build the pinned embedding provider, or None when not fully configured."""
    model = os.environ.get("RAGAS_EMBEDDING_MODEL")
    base_url = os.environ.get("RAGAS_EMBEDDING_BASE_URL")
    api_key = os.environ.get("RAGAS_EMBEDDING_API_KEY")
    if not (model and base_url and api_key):
        return None
    from langchain_openai import OpenAIEmbeddings  # type: ignore[import-not-found]

    return OpenAIEmbeddings(model=model, api_key=api_key, base_url=base_url)


def _environment(model: str, embeddings: object | None) -> dict[str, object]:
    """Record reproducible provenance without ever exposing credentials."""
    return {
        "python_version": platform.python_version(),
        "git_commit": _git_commit(),
        "model": model,
        "api_base_url": DEEPSEEK_BASE_URL,
        "inference_backend": "OpenAI-compatible HTTP",
        "embedding_provider_configured": embeddings is not None,
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[2],
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser(
        "score", help="score persisted paired outputs with Ragas"
    )
    score_parser.add_argument("--input", required=True, type=Path)
    score_parser.add_argument("--output-dir", required=True, type=Path)
    score_parser.add_argument("--model", default=_default_model())

    verify_parser = subparsers.add_parser(
        "verify", help="validate checksums and artifact consistency"
    )
    verify_parser.add_argument("--input", required=True, type=Path)
    verify_parser.add_argument("--scores", type=Path)
    verify_parser.add_argument("--testset", default=DEFAULT_TESTSET_PATH, type=Path)

    args = parser.parse_args()
    if args.command == "score":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            parser.error("DEEPSEEK_API_KEY must be set externally")
        result = run(
            args.output_dir,
            artifact_path=args.input,
            api_key=api_key,
            model=args.model,
            embeddings=_embeddings_from_env(),
        )
        print(json.dumps(result, sort_keys=True))
    else:
        report = verify(
            artifact_path=args.input,
            scores_path=args.scores,
            testset_path=args.testset,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["ok"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
