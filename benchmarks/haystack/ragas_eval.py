"""End-to-end Ragas evaluation of the Haystack adaptation of ragrefine.

The harness compares two branches over the identical retrieved Top-N pool for
every question in a fixed human-reviewed test set:

* ``baseline`` (B0): the original Haystack retrieval order, truncated to Top-K;
* ``refined``: the ``RagRefineComponent``-selected context from the same pool.

Answers are generated with DeepSeek through its OpenAI-compatible endpoint,
configured only through environment variables (``DEEPSEEK_API_KEY`` and
``DEEPSEEK_MODEL``), and scored with Ragas LLM-only metrics (``faithfulness``,
``context_recall``, ``factual_correctness``). For every query the artifact
persists the exact prompts, context IDs and pool digest, answers, the full
refinement trace, retrieval/refinement/generation timing, model configuration,
and per-arm generation failures. A query contributes to the Ragas comparison
only when both arms generated an answer, so aggregate metrics stay paired.
This is an evaluation harness only; it does not, by itself, establish that
refinement improves generation or context quality.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

from haystack import Document
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret

from benchmarks.haystack.testset import DEFAULT_TESTSET_PATH, load_testset
from integrations.haystack.component import RagRefineComponent
from integrations.haystack.prompt import prompt_builder
from ragrefine import Refiner, RefinerConfig
from ragrefine.config import ChannelConfig
from ragrefine.ranking.lexical import LexicalRanker

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
SCHEMA_VERSION = "1.1"


class ChatGenerator(Protocol):
    """Test seam matching Haystack's ``OpenAIChatGenerator`` interface."""

    def run(self, messages: list[ChatMessage]) -> dict[str, list[ChatMessage]]: ...


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """One question, its retrieved context, generated answer, and reference."""

    question: str
    contexts: tuple[str, ...]
    answer: str
    reference: str


@dataclass(frozen=True, slots=True)
class GenerationOutcome:
    """One arm's generation result, or a persisted per-arm failure.

    A failed arm keeps its built prompt (when available) so the artifact can
    reconstruct exactly what was attempted; ``failure`` records the typed error
    without ever exposing credentials or request internals.
    """

    prompt: str | None
    answer: str | None
    latency_seconds: float | None
    failure: dict[str, object] | None


class Evaluator(Protocol):
    """Scoring seam: returns aggregate and per-record metric scores."""

    def evaluate(self, records: list[EvaluationRecord]) -> dict[str, object]: ...


class _RagasEvaluator:
    """Ragas scorer built against DeepSeek's OpenAI-compatible endpoint.

    ``evaluate_fn`` is a test seam; when injected, no LLM client is built and
    the caller must supply a fake result object with a ``scores`` attribute.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        evaluate_fn: Callable[..., object] | None = None,
    ) -> None:
        # Imported lazily so the harness and its tests do not require Ragas.
        from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]
        from ragas.llms import LangchainLLMWrapper  # type: ignore[import-not-found]
        from ragas.metrics import (  # type: ignore[import-not-found]
            FactualCorrectness,
            Faithfulness,
            LLMContextRecall,
        )

        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0,
        )
        self._evaluator_llm = LangchainLLMWrapper(llm)
        self._metrics = [Faithfulness(), LLMContextRecall(), FactualCorrectness()]
        self._evaluate_fn = evaluate_fn

    def evaluate(self, records: list[EvaluationRecord]) -> dict[str, object]:
        """Score one branch and return aggregate plus per-record metrics.

        ``result.scores`` in Ragas 0.4.x is a list of per-sample dicts; a
        non-finite or failed sample is recorded as None, never as a score.
        Mode-parameterised metrics (``factual_correctness``) are keyed by
        Ragas as ``name(mode=...)``.
        """
        from ragas import EvaluationDataset, evaluate  # type: ignore[import-not-found]

        dataset = EvaluationDataset.from_list(
            [
                {
                    "user_input": record.question,
                    "retrieved_contexts": list(record.contexts),
                    "response": record.answer,
                    "reference": record.reference,
                }
                for record in records
            ]
        )
        evaluate_fn = self._evaluate_fn or evaluate
        result = evaluate_fn(
            dataset=dataset,
            metrics=self._metrics,
            llm=None if self._evaluate_fn is not None else self._evaluator_llm,
            raise_exceptions=True,
        )
        metric_names = [metric.name for metric in self._metrics]
        score_keys = [_score_key(metric) for metric in self._metrics]
        rows = result.scores
        per_record = [
            {
                name: _finite_or_none(row.get(key))
                for name, key in zip(metric_names, score_keys)
            }
            for row in rows
        ]
        return {
            "metrics": metric_names,
            "aggregate": {
                name: _mean_or_none([row.get(key) for row in rows])
                for name, key in zip(metric_names, score_keys)
            },
            "per_record": per_record,
        }


def run(
    output_dir: Path,
    *,
    api_key: str,
    model: str,
    generator: ChatGenerator | None = None,
    evaluator: Evaluator | None = None,
    refiner: Refiner | None = None,
    top_n: int = 5,
    top_k: int = 3,
    testset_path: Path = DEFAULT_TESTSET_PATH,
    max_queries: int | None = None,
) -> dict[str, object]:
    """Run the paired baseline/refined Ragas evaluation and persist artifacts.

    ``max_queries`` slices the frozen test set to the first N questions for a
    cost/behaviour pilot only; the declared frozen run leaves it unset so every
    question is evaluated.
    """
    if output_dir.exists():
        raise FileExistsError(f"Ragas evaluation output already exists: {output_dir}")
    if top_n < 1 or top_k < 1 or top_k > top_n:
        raise ValueError("top_k must satisfy 1 <= top_k <= top_n")
    if max_queries is not None and max_queries < 1:
        raise ValueError("max_queries must be a positive integer or None")

    testset = load_testset(testset_path)
    contract = testset.contract
    _validate_run_contract(contract, model=model, top_n=top_n, top_k=top_k)
    active_generator = generator or OpenAIChatGenerator(
        api_key=Secret.from_token(api_key),
        model=model,
        api_base_url=contract.api_base_url,
        generation_kwargs={
            "temperature": contract.temperature,
            "max_tokens": contract.generation_max_tokens,
        },
        max_retries=0,
    )
    active_evaluator = evaluator or _RagasEvaluator(api_key=api_key, model=model)
    active_refiner = refiner or _default_lexical_refiner()

    store = InMemoryDocumentStore(bm25_algorithm="BM25L")
    store.write_documents(
        [
            Document(id=document.id, content=document.content, meta=document.meta)
            for document in testset.corpus
        ]
    )
    retriever = InMemoryBM25Retriever(document_store=store)
    component = RagRefineComponent(
        refiner=active_refiner,
        top_k=top_k,
        max_tokens=contract.max_tokens,
    )

    baseline_records: list[EvaluationRecord] = []
    refined_records: list[EvaluationRecord] = []
    rows: list[dict[str, object]] = []
    paired_row_indexes: list[int] = []

    for pair in testset.qa_pairs[:max_queries]:
        retrieval_started = perf_counter()
        pool = tuple(retriever.run(query=pair.question, top_k=top_n)["documents"])
        retrieval_latency = perf_counter() - retrieval_started
        baseline_docs = list(pool)[:top_k]
        refinement_started = perf_counter()
        refined_output = component.run(pair.question, list(pool))
        refinement_latency = perf_counter() - refinement_started
        refined_docs = list(refined_output["documents"])

        baseline_outcome = _generate(active_generator, pair.question, baseline_docs)
        refined_outcome = _generate(active_generator, pair.question, refined_docs)

        paired = baseline_outcome.failure is None and refined_outcome.failure is None
        if paired:
            baseline_records.append(
                EvaluationRecord(
                    question=pair.question,
                    contexts=tuple(document.content for document in baseline_docs),
                    answer=baseline_outcome.answer or "",
                    reference=pair.reference_answer,
                )
            )
            refined_records.append(
                EvaluationRecord(
                    question=pair.question,
                    contexts=tuple(document.content for document in refined_docs),
                    answer=refined_outcome.answer or "",
                    reference=pair.reference_answer,
                )
            )
            paired_row_indexes.append(len(rows))

        rows.append(
            {
                "qa_id": pair.id,
                "question": pair.question,
                "reference_answer": pair.reference_answer,
                "reference_context_ids": list(pair.reference_context_ids),
                "pool_ids": [str(document.id) for document in pool],
                "pool_sha256": _pool_digest(pool),
                "baseline_context_ids": [
                    str(document.id) for document in baseline_docs
                ],
                "refined_context_ids": [str(document.id) for document in refined_docs],
                "refinement_trace": _json_safe(refined_output["trace"]),
                "baseline_prompt": baseline_outcome.prompt,
                "refined_prompt": refined_outcome.prompt,
                "baseline_answer": baseline_outcome.answer,
                "refined_answer": refined_outcome.answer,
                "baseline_failure": baseline_outcome.failure,
                "refined_failure": refined_outcome.failure,
                "paired": paired,
                "retrieval_latency_seconds": retrieval_latency,
                "refinement_latency_seconds": refinement_latency,
                "baseline_generation_latency_seconds": baseline_outcome.latency_seconds,
                "refined_generation_latency_seconds": refined_outcome.latency_seconds,
            }
        )

    if paired_row_indexes:
        baseline_scores = active_evaluator.evaluate(baseline_records)
        refined_scores = active_evaluator.evaluate(refined_records)
        if tuple(baseline_scores["metrics"]) != contract.metrics:
            raise RuntimeError("evaluator did not return the fixed evaluation metrics")
        if tuple(refined_scores["metrics"]) != contract.metrics:
            raise RuntimeError("evaluator did not return the fixed evaluation metrics")
    else:
        baseline_scores = {
            "metrics": list(contract.metrics),
            "aggregate": {},
            "per_record": [],
        }
        refined_scores = {
            "metrics": list(contract.metrics),
            "aggregate": {},
            "per_record": [],
        }

    for row_index, scores in zip(
        paired_row_indexes, baseline_scores["per_record"], strict=True
    ):
        rows[row_index]["baseline_scores"] = scores
    for row_index, scores in zip(
        paired_row_indexes, refined_scores["per_record"], strict=True
    ):
        rows[row_index]["refined_scores"] = scores

    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "testset": {
            "path": str(testset_path),
            "sha256": testset.sha256,
            "corpus_size": len(testset.corpus),
            "qa_pair_count": len(testset.qa_pairs),
        },
        "model": model,
        "api_base_url": contract.api_base_url,
        "profile": contract.refinement_profile,
        "generation": {
            "model": model,
            "api_base_url": contract.api_base_url,
            "temperature": contract.temperature,
            "max_tokens": contract.generation_max_tokens,
            "prompt_template": contract.prompt_template,
        },
        "configuration": {
            "top_n": top_n,
            "top_k": top_k,
            "max_tokens": contract.max_tokens,
            "prompt_template": contract.prompt_template,
            "pool_reused": True,
        },
        "review": {
            "status": testset.review.status,
            "reviewer_role": testset.review.reviewer_role,
            "reviewed_on": testset.review.reviewed_on,
        },
        "evaluation": {
            "queries_total": len(rows),
            "paired_query_count": len(paired_row_indexes),
            "baseline_record_count": len(baseline_records),
            "refined_record_count": len(refined_records),
            "baseline_failures": sum(
                1 for row in rows if row["baseline_failure"] is not None
            ),
            "refined_failures": sum(
                1 for row in rows if row["refined_failure"] is not None
            ),
        },
        "per_query": rows,
        "metrics": {
            "names": baseline_scores["metrics"],
            "baseline": baseline_scores["aggregate"],
            "refined": refined_scores["aggregate"],
        },
        "environment": _environment(model),
        "interpretation": (
            "Ragas evaluation only; a refined-branch advantage is reported only "
            "when the recorded aggregate and per-query metrics support it. "
            "Queries are scored only when both arms generated an answer; "
            "per-arm failures are persisted per query."
        ),
    }

    output_dir.mkdir(parents=True)
    output_path = output_dir / "ragas-results.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "ragas-results.json.sha256").write_text(
        hashlib.sha256(output_path.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    return result


def _default_lexical_refiner() -> Refiner:
    """Return the retained lightweight lexical (B2-L) refinement profile."""
    return Refiner(
        lexical_ranker=LexicalRanker(),
        config=RefinerConfig(
            original=ChannelConfig(enabled=False),
            lexical=ChannelConfig(enabled=True),
        ),
    )


def _generate(
    generator: ChatGenerator, query: str, documents: list[Document]
) -> GenerationOutcome:
    """Generate one arm's answer, capturing per-arm failures instead of aborting."""
    try:
        prompt = prompt_builder().run(query=query, documents=documents)["prompt"]
    except Exception as error:
        return GenerationOutcome(
            prompt=None,
            answer=None,
            latency_seconds=None,
            failure=_failure(error),
        )
    started = perf_counter()
    try:
        reply = generator.run(messages=[ChatMessage.from_user(prompt)])["replies"][0]
        return GenerationOutcome(
            prompt=prompt,
            answer=reply.text,
            latency_seconds=perf_counter() - started,
            failure=None,
        )
    except Exception as error:
        return GenerationOutcome(
            prompt=prompt,
            answer=None,
            latency_seconds=perf_counter() - started,
            failure=_failure(error),
        )


def _json_safe(value: object) -> object:
    """Convert dataclass-derived tuples into JSON round-trip stable lists."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _failure(error: Exception) -> dict[str, object]:
    """Return a credential-safe persisted failure record for one arm."""
    return {"failure_type": type(error).__name__, "reason": str(error)}


def _finite_or_none(value: object) -> float | None:
    """Convert a metric value to a finite float, or None for a failure."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return float(value)


def _mean_or_none(values: Sequence[object]) -> float | None:
    """Mean of finite per-sample values; failures are never converted to scores."""
    finite = [
        value
        for value in (_finite_or_none(item) for item in values)
        if value is not None
    ]
    return sum(finite) / len(finite) if finite else None


def _score_key(metric_obj: object) -> str:
    """Return the Ragas score-column key for one metric object.

    Ragas keys mode-parameterised metrics (e.g. ``FactualCorrectness``) as
    ``"factual_correctness(mode=f1)"``; everything else uses the plain name.
    """
    name = getattr(metric_obj, "name", "")
    mode = getattr(metric_obj, "mode", None)
    return f"{name}(mode={mode})" if mode is not None else name


def _pool_digest(documents: Sequence[Document]) -> str:
    """Hash ordered pool IDs and text to prove both arms shared the same pool."""
    payload = json.dumps(
        [(str(document.id), document.content) for document in documents],
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _default_model() -> str:
    """Return the DeepSeek model name configured only through the environment."""
    return os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")


def _validate_run_contract(
    contract: object, *, model: str, top_n: int, top_k: int
) -> None:
    """Reject changed retrieval or generation settings for a frozen test set."""
    from benchmarks.haystack.testset import EvaluationContract

    if not isinstance(contract, EvaluationContract):
        raise TypeError("test-set contract must be an EvaluationContract")
    if model != contract.model:
        raise ValueError("model does not match the frozen evaluation contract")
    if top_n != contract.top_n or top_k != contract.top_k:
        raise ValueError("Top-N/Top-K do not match the frozen evaluation contract")


def _environment(model: str) -> dict[str, object]:
    """Record reproducible provenance for the persisted artifact."""
    return {
        "python_version": platform.python_version(),
        "git_commit": _git_commit(),
        "model": model,
        "api_base_url": DEEPSEEK_BASE_URL,
        "inference_backend": "OpenAI-compatible HTTP",
        "generation_settings": {"temperature": 0},
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
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="pilot hook: evaluate only the first N frozen questions",
    )
    args = parser.parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        parser.error("DEEPSEEK_API_KEY must be set externally")
    print(
        json.dumps(
            run(
                args.output_dir,
                api_key=api_key,
                model=_default_model(),
                top_n=args.top_n,
                top_k=args.top_k,
                max_queries=args.max_queries,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
