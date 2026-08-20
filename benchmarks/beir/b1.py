"""Run B1 CrossEncoder reranking against an existing frozen B0 snapshot."""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from benchmarks.beir.metrics import evaluate_snapshot
from benchmarks.beir.snapshot import (
    load_verified_snapshot,
    snapshot_checksum,
    write_snapshot,
)
from ragrefine.models import Candidate
from ragrefine.rerank import SentenceTransformersReranker

LOGGER = logging.getLogger(__name__)

EXPECTED_B0_CHECKSUM = (
    "fc08cf7b496c8cd7c9020a81560d08edc10f389b27682f2b6e722ccfef0793dc"
)


class _ModelInfo(Protocol):
    sha: str


class _ModelInfoClient(Protocol):
    def model_info(self, repo_id: str, *, revision: str) -> _ModelInfo: ...


def resolve_model_revision(
    model: str,
    revision: str | None,
    *,
    client: _ModelInfoClient | None = None,
) -> str:
    """Validate a model revision and return its immutable Hugging Face commit SHA."""
    if not model.strip():
        raise ValueError("model must not be empty")
    requested_revision = revision or "main"
    if client is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise RuntimeError(
                "install the optional extra: ragrefine[rerank]"
            ) from error
        client = HfApi()
    try:
        info = client.model_info(model, revision=requested_revision)
    except Exception as error:
        raise ValueError(
            f"model revision is unavailable: model={model!r}, "
            f"revision={requested_revision!r}"
        ) from error
    if not info.sha:
        raise ValueError(f"model revision did not resolve to a commit SHA: {model!r}")
    return info.sha


def wall_clock_measurement(
    *,
    device: str,
    started_at: datetime,
    completed_at: datetime,
    elapsed_ms: float,
) -> dict[str, object]:
    """Build validated, serializable elapsed-time metadata for a B1 run phase."""
    if elapsed_ms < 0:
        raise ValueError("elapsed_ms must be non-negative")
    if completed_at < started_at:
        raise ValueError("completed_at must not precede started_at")
    return {
        "device": device,
        "started_at_utc": started_at.astimezone(UTC).isoformat(),
        "completed_at_utc": completed_at.astimezone(UTC).isoformat(),
        "elapsed_ms": elapsed_ms,
    }


def run(
    snapshot_path: Path,
    output_dir: Path,
    *,
    model: str,
    revision: str | None = None,
    batch_size: int = 128,
    queries_per_batch: int = 8,
    device: str = "cpu",
) -> dict[str, float]:
    """Persist B1 ranking and metrics without regenerating first-stage retrieval."""
    run_started_at = datetime.now(UTC)
    run_started = perf_counter()
    print(f"Loading frozen B0 snapshot: {snapshot_path}", flush=True)
    snapshot = load_verified_snapshot(snapshot_path)
    checksum = snapshot_checksum(snapshot)
    resolved_revision = resolve_model_revision(model, revision)
    query_count = len(snapshot["queries"])
    LOGGER.info("Verified B0 snapshot checksum: %s", checksum)
    print(
        f"B1 setup: {query_count} queries, Top-{snapshot['retriever']['top_n']}, "
        f"model={model}, revision={resolved_revision}, device={device}, "
        f"batch_size={batch_size}, "
        f"queries_per_batch={queries_per_batch}",
        flush=True,
    )
    reranker = SentenceTransformersReranker(
        model, revision=resolved_revision, device=device, batch_size=batch_size
    )
    reranked: dict[str, Any] = {
        **snapshot,
        "b1": {"model": model, "revision": resolved_revision},
    }
    timings: list[dict[str, object]] = []
    queries: list[dict[str, Any]] = []
    prepared = [
        (
            query,
            tuple(
                Candidate(
                    id=item["id"],
                    text=item["text"],
                    retrieval_score=item["score"],
                    retrieval_rank=item["rank"],
                )
                for item in query["candidates"]
            ),
        )
        for query in snapshot["queries"]
    ]
    reranking_started_at = datetime.now(UTC)
    reranking_started = perf_counter()
    for batch_start in range(0, query_count, queries_per_batch):
        batch = prepared[batch_start : batch_start + queries_per_batch]
        started = perf_counter()
        scored_batches = reranker.rank_many(
            tuple((query["text"], candidates) for query, candidates in batch)
        )
        duration_ms = (perf_counter() - started) * 1000
        timings.append(
            {
                "query_ids": [query["id"] for query, _ in batch],
                "duration_ms": duration_ms,
                "per_query_estimate_ms": duration_ms / len(batch),
            }
        )
        for (query, _), scored in zip(batch, scored_batches, strict=True):
            original = {item["id"]: item for item in query["candidates"]}
            queries.append(
                {
                    "id": query["id"],
                    "text": query["text"],
                    "candidates": [
                        {
                            **original[item.candidate.id],
                            "rank": item.rank,
                            "score": item.score,
                        }
                        for item in scored
                    ],
                }
            )
        completed = min(batch_start + len(batch), query_count)
        print(f"Reranked {completed}/{query_count} queries", flush=True)
    reranking_completed_at = datetime.now(UTC)
    reranking_elapsed_ms = (perf_counter() - reranking_started) * 1_000
    reranked["queries"] = queries
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = output_dir / "b1-ranking.json"
    write_snapshot(reranked, ranking_path)
    LOGGER.info("Persisted B1 ranking artifact: %s", ranking_path)
    metrics = evaluate_snapshot(reranked)
    (output_dir / "b1-metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "b1-latency.json").write_text(json.dumps(timings, indent=2) + "\n")
    (output_dir / "b1-environment.json").write_text(
        json.dumps(
            {
                "b0_snapshot_checksum": checksum,
                "model": model,
                "revision": resolved_revision,
                "backend": "sentence-transformers",
                "device": device,
                "batch_size": batch_size,
                "queries_per_batch": queries_per_batch,
                "ranking_checksum": snapshot_checksum(reranked),
                "reranking_wall_clock": wall_clock_measurement(
                    device=device,
                    started_at=reranking_started_at,
                    completed_at=reranking_completed_at,
                    elapsed_ms=reranking_elapsed_ms,
                ),
                "run_wall_clock": wall_clock_measurement(
                    device=device,
                    started_at=run_started_at,
                    completed_at=datetime.now(UTC),
                    elapsed_ms=(perf_counter() - run_started) * 1_000,
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"B1 complete; artifacts written to: {output_dir}", flush=True)
    return metrics


def main() -> None:
    """Run B1 against one pre-existing frozen B0 snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--queries-per-batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(
        json.dumps(
            run(
                args.snapshot,
                args.output_dir,
                model=args.model,
                revision=args.revision,
                batch_size=args.batch_size,
                queries_per_batch=args.queries_per_batch,
                device=args.device,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
