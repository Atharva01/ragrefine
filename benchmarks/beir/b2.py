"""Evaluate independent B2 lexical and pattern signals on frozen inputs."""

import argparse
import json
import platform
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from benchmarks.beir.metrics import evaluate_snapshot
from benchmarks.beir.snapshot import load_snapshot, snapshot_checksum, write_snapshot
from benchmarks.hard_set.validate import load_and_validate
from ragrefine.models import Candidate
from ragrefine.query.patterns import PatternRegistry, PatternRule
from ragrefine.ranking.lexical import LexicalRanker
from ragrefine.ranking.patterns import PatternRanker

EXPECTED_B0_CHECKSUM = (
    "fc08cf7b496c8cd7c9020a81560d08edc10f389b27682f2b6e722ccfef0793dc"
)


def _compact(value: str) -> str:
    """Normalize equivalent technical separator variants."""
    return value.casefold().replace("_", "-").replace(" ", "-")


def _patterns() -> PatternRanker:
    return PatternRanker(
        PatternRegistry(
            (
                PatternRule(
                    "version",
                    r"(?i)\b(?:model[-_ ]?v\d+|v\d+(?:\.\d+){0,3}|\d+\.\d+\.\d+)\b",
                    _compact,
                ),
                PatternRule("identifier", r"(?i)\b(?:inc|rq|alrt)-\d+\b"),
                PatternRule("date", r"\b\d{4}-\d{2}-\d{2}\b"),
                PatternRule("percentage", r"\b\d+(?:\.\d+)?\s*%"),
                PatternRule("number", r"\b\d+(?:\.\d+)?\b"),
            )
        )
    )


def _candidates(query: dict[str, Any]) -> tuple[Candidate, ...]:
    return tuple(
        Candidate(
            id=item["id"],
            text=item["text"],
            retrieval_rank=item["rank"],
            retrieval_score=item["score"],
        )
        for item in query["candidates"]
    )


def _percentile(values: list[float], fraction: float) -> float:
    return (
        sorted(values)[max(0, min(len(values) - 1, int((len(values) - 1) * fraction)))]
        if values
        else 0.0
    )


def _rank_snapshot(
    snapshot: dict[str, Any], signal: str
) -> tuple[dict[str, Any], list[float]]:
    ranker: Any = LexicalRanker() if signal == "lexical" else _patterns()
    ranked = {**snapshot, "b2": {"signal": signal, "signal_version": "1"}}
    queries: list[dict[str, Any]] = []
    timings: list[float] = []
    for query in snapshot["queries"]:
        started = perf_counter()
        result = ranker.rank(query["text"], _candidates(query))
        timings.append((perf_counter() - started) * 1000)
        original = {item["id"]: item for item in query["candidates"]}
        queries.append(
            {
                "id": query["id"],
                "text": query["text"],
                "candidates": [
                    {**original[item.candidate.id], "rank": item.rank}
                    for item in result
                ],
            }
        )
    ranked["queries"] = queries
    return ranked, timings


def _hard_metrics(path: Path, signal: str) -> dict[str, object]:
    data = load_and_validate(path)
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in data["cases"]:
        groups[case["query_id"]].append(case)
    ranker: Any = LexicalRanker() if signal == "lexical" else _patterns()
    category_ranks: defaultdict[str, list[int]] = defaultdict(list)
    per_query: list[dict[str, object]] = []
    for cases in groups.values():
        candidates = tuple(
            Candidate(
                id=item["candidate_id"],
                text=item["candidate_text"],
                retrieval_rank=index,
            )
            for index, item in enumerate(cases, start=1)
        )
        ranked = ranker.rank(cases[0]["query_text"], candidates)
        positive = next(
            item["candidate_id"] for item in cases if item["relevance"] == 1
        )
        positive_rank = next(
            item.rank for item in ranked if item.candidate.id == positive
        )
        category_ranks[cases[0]["category"]].append(positive_rank)
        per_query.append(
            {
                "query_id": cases[0]["query_id"],
                "category": cases[0]["category"],
                "relevant_rank": positive_rank,
            }
        )
    by_category = {
        category: {
            "top_1_accuracy": sum(rank == 1 for rank in ranks) / len(ranks),
            "mrr": sum(1 / rank for rank in ranks) / len(ranks),
            "relevant_rank": ranks,
        }
        for category, ranks in sorted(category_ranks.items())
    }
    all_ranks = [rank for ranks in category_ranks.values() for rank in ranks]
    return {
        "dataset_checksum": path.with_suffix(path.suffix + ".sha256")
        .read_text()
        .strip(),
        "query_count": len(groups),
        "top_1_accuracy": sum(rank == 1 for rank in all_ranks) / len(all_ranks),
        "mrr": sum(1 / rank for rank in all_ranks) / len(all_ranks),
        "by_category": by_category,
        "per_query": per_query,
    }


def run(
    snapshot_path: Path, hard_set_path: Path, output_dir: Path, signal: str
) -> dict[str, float]:
    """Run exactly one B2 signal against frozen inputs and persist artifacts."""
    snapshot = load_snapshot(snapshot_path)
    checksum = snapshot_checksum(snapshot)
    if checksum != EXPECTED_B0_CHECKSUM:
        raise ValueError("B2 requires the documented SciFact B0 snapshot checksum")
    started = perf_counter()
    ranked, timings = _rank_snapshot(snapshot, signal)
    metrics = evaluate_snapshot(ranked)
    baseline_metrics = evaluate_snapshot(snapshot)
    hard_metrics = _hard_metrics(hard_set_path, signal)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = output_dir / f"b2-{signal}-ranking.json"
    write_snapshot(ranked, ranking_path)
    decision = (
        "retain"
        if hard_metrics["top_1_accuracy"] == 1.0
        and all(metrics[name] >= baseline_metrics[name] for name in metrics)
        and any(metrics[name] > baseline_metrics[name] for name in metrics)
        else "modify"
    )
    (output_dir / f"b2-{signal}-metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / f"b2-{signal}-diagnostics.json").write_text(
        json.dumps(hard_metrics, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / f"b2-{signal}-config.json").write_text(
        json.dumps(
            {
                "signal": signal,
                "signal_version": "1",
                "device": "cpu",
                "backend": "stdlib",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output_dir / f"b2-{signal}-environment.json").write_text(
        json.dumps(
            {
                "signal": signal,
                "signal_version": "1",
                "device": "cpu",
                "backend": "stdlib",
                "b0_snapshot_checksum": checksum,
                "hard_set_checksum": hard_metrics["dataset_checksum"],
                "ranking_checksum": snapshot_checksum(ranked),
                "python": platform.python_version(),
                "git_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip(),
                "runtime": {
                    "total_ms": (perf_counter() - started) * 1000,
                    "p50_query_ms": median(timings),
                    "p95_query_ms": _percentile(timings, 0.95),
                    "queries": len(timings),
                    "candidates": sum(
                        len(query["candidates"]) for query in snapshot["queries"]
                    ),
                    "candidates_per_second": sum(
                        len(query["candidates"]) for query in snapshot["queries"]
                    )
                    / (sum(timings) / 1000),
                },
                "decision": decision,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument(
        "--hard-set",
        type=Path,
        default=Path("benchmarks/hard_set/structured-hard-negatives-v1.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--signal", choices=("lexical", "patterns"))
    parser.add_argument("--reproduce-ranking", type=Path)
    args = parser.parse_args()
    if args.reproduce_ranking:
        first = evaluate_snapshot(load_snapshot(args.reproduce_ranking))
        second = evaluate_snapshot(load_snapshot(args.reproduce_ranking))
        if first != second:
            raise RuntimeError("persisted B2 ranking metrics were not reproducible")
        print(json.dumps(first, sort_keys=True))
        return
    if not args.snapshot or not args.output_dir or not args.signal:
        parser.error("--snapshot, --output-dir, and --signal are required")
    print(
        json.dumps(
            run(args.snapshot, args.hard_set, args.output_dir, args.signal),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
