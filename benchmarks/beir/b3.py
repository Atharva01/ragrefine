"""Fuse persisted B0/B1/B2 rankings without regenerating SciFact retrieval."""

import argparse
import json
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

from benchmarks.beir.b2 import EXPECTED_B0_CHECKSUM
from benchmarks.beir.metrics import evaluate_snapshot
from benchmarks.beir.snapshot import load_snapshot, snapshot_checksum, write_snapshot
from benchmarks.hard_set.validate import load_and_validate
from ragrefine.models import Candidate
from ragrefine.ranking.lexical import LexicalRanker
from ragrefine.ranking.patterns import PatternRanker
from ragrefine.ranking.rrf import ReciprocalRankFusion
from ragrefine.ranking.types import RankingChannel
from ragrefine.rerank import SentenceTransformersReranker
from ragrefine.rerank.base import NeuralReranker

PROFILES = {
    "original-lexical": ("original", "lexical"),
    "original-pattern": ("original", "pattern"),
    "b3-light": ("original", "lexical", "pattern"),
    "b3-reference": ("original", "neural", "lexical", "pattern"),
}

FINAL_K = 60


def _new_directory(path: Path) -> None:
    """Refuse to overwrite an existing experimental result directory."""
    if path.exists():
        raise FileExistsError(f"B3 output directory already exists: {path}")
    path.mkdir(parents=True)


def _load_required_snapshot(path: Path) -> dict[str, Any]:
    """Load a persisted artifact only when its SHA-256 sidecar is present."""
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists():
        raise ValueError(f"missing required checksum sidecar: {sidecar}")
    return load_snapshot(path)


def _hard_snapshot(path: Path) -> dict[str, Any]:
    """Convert the validated hard set to the frozen snapshot schema."""
    data = load_and_validate(path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in data["cases"]:
        grouped.setdefault(case["query_id"], []).append(case)
    queries: list[dict[str, Any]] = []
    qrels: dict[str, dict[str, int]] = {}
    for query_id, cases in grouped.items():
        queries.append(
            {
                "id": query_id,
                "text": cases[0]["query_text"],
                "category": cases[0]["category"],
                "expected_constraints": cases[0]["expected_constraints"],
                "candidates": [
                    {
                        "id": item["candidate_id"],
                        "text": item["candidate_text"],
                        "rank": rank,
                        "retrieval_rank": rank,
                        "score": 0.0,
                    }
                    for rank, item in enumerate(cases, start=1)
                ],
            }
        )
        qrels[query_id] = {item["candidate_id"]: item["relevance"] for item in cases}
    return {
        "schema_version": "1.0",
        "dataset": {"id": data["dataset_id"], "version": data["dataset_version"]},
        "retriever": {"top_n": len(queries[0]["candidates"])},
        "qrels": qrels,
        "queries": queries,
    }


def _rank_hard(
    snapshot: dict[str, Any], channel: str, reranker: NeuralReranker | None = None
) -> dict[str, Any]:
    """Create one complete hard-set upstream ranking without changing its pool."""
    from benchmarks.beir.b2 import _patterns

    ranker: LexicalRanker | PatternRanker | None
    ranker = (
        LexicalRanker()
        if channel == "lexical"
        else _patterns()
        if channel == "pattern"
        else None
    )
    queries: list[dict[str, Any]] = []
    for query in snapshot["queries"]:
        candidates = tuple(
            Candidate(id=item["id"], text=item["text"], retrieval_rank=item["rank"])
            for item in query["candidates"]
        )
        if channel == "original":
            ordered = candidates
        elif channel == "neural":
            if reranker is None:
                raise ValueError("hard neural ranking requires a reranker")
            ordered = tuple(
                item.candidate for item in reranker.rank(query["text"], candidates)
            )
        else:
            if ranker is None:
                raise ValueError(f"unknown hard channel: {channel}")
            ordered = tuple(
                item.candidate for item in ranker.rank(query["text"], candidates)
            )
        original = {item["id"]: item for item in query["candidates"]}
        queries.append(
            {
                **query,
                "candidates": [
                    {**original[item.id], "rank": rank}
                    for rank, item in enumerate(ordered, start=1)
                ],
            }
        )
    return {**snapshot, "queries": queries, "b3_upstream": {"channel": channel}}


def prepare_hard(
    hard_set: Path, output_dir: Path, *, model: str, revision: str, device: str
) -> dict[str, str]:
    """Persist original, lexical, pattern, and one-time neural hard rankings."""
    _new_directory(output_dir)
    source = _hard_snapshot(hard_set)
    reranker = SentenceTransformersReranker(
        model, revision=revision, device=device, batch_size=512
    )
    checksums: dict[str, str] = {}
    for channel in ("original", "lexical", "pattern", "neural"):
        ranked = _rank_hard(source, channel, reranker if channel == "neural" else None)
        checksums[channel] = write_snapshot(ranked, output_dir / f"{channel}.json")
    (output_dir / "environment.json").write_text(
        json.dumps(
            {
                "hard_set_checksum": hard_set.with_suffix(hard_set.suffix + ".sha256")
                .read_text()
                .strip(),
                "model": model,
                "revision": revision,
                "device": device,
                "backend": "sentence-transformers",
                "ranking_checksums": checksums,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return checksums


def _hard_diagnostics(snapshot: Mapping[str, object]) -> dict[str, object]:
    """Report deterministic structured-hard-negative ranking outcomes."""
    categories: dict[str, list[dict[str, object]]] = {}
    rows: list[dict[str, object]] = []
    for query in snapshot["queries"]:
        query_id = str(query["id"])
        relevant = next(
            doc_id for doc_id, value in snapshot["qrels"][query_id].items() if value > 0
        )
        rank = next(
            index
            for index, item in enumerate(query["candidates"], start=1)
            if item["id"] == relevant
        )
        original_rank = next(
            item["retrieval_rank"]
            for item in query["candidates"]
            if item["id"] == relevant
        )
        row = {
            "query_id": query_id,
            "category": query["category"],
            "top_1_correct": rank == 1,
            "mrr": 1 / rank,
            "relevant_rank": rank,
            "rank_movement": original_rank - rank,
        }
        rows.append(row)
        categories.setdefault(str(query["category"]), []).append(row)
    return {
        "query_count": len(rows),
        "top_1_accuracy": sum(row["top_1_correct"] for row in rows) / len(rows),
        "mrr": sum(row["mrr"] for row in rows) / len(rows),
        "by_category": {
            name: {
                "top_1_accuracy": sum(row["top_1_correct"] for row in values)
                / len(values),
                "mrr": sum(row["mrr"] for row in values) / len(values),
                "mean_rank_movement": sum(row["rank_movement"] for row in values)
                / len(values),
            }
            for name, values in sorted(categories.items())
        },
        "per_query": rows,
    }


def _percentile(values: list[float], percentile: float) -> float:
    """Return a deterministic nearest-rank percentile for recorded timings."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _environment_for(path: Path) -> dict[str, Any] | None:
    """Load the provenance sidecar associated with a persisted ranking, if any."""
    candidates = [
        path.parent / "environment.json",
        *path.parent.glob("*-environment.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def _candidate_map(
    snapshot: Mapping[str, object],
) -> dict[str, tuple[tuple[str, str], ...]]:
    """Return complete candidate identities, retaining multiplicity for validation."""
    return {
        str(query["id"]): tuple(
            sorted((str(item["id"]), str(item["text"])) for item in query["candidates"])
        )
        for query in snapshot["queries"]
    }


def _validate_upstream(
    baseline: Mapping[str, object], upstream: Mapping[str, Mapping[str, object]]
) -> None:
    expected = _candidate_map(baseline)
    for name, snapshot in upstream.items():
        if _candidate_map(snapshot) != expected:
            raise ValueError(f"{name} ranking does not preserve the frozen B0 pool")


def _validate_environment(
    path: Path, snapshot: Mapping[str, object], b0_checksum: str
) -> None:
    """Require upstream benchmark provenance to reference the same frozen B0."""
    data = _environment_for(path)
    if data is None:
        raise ValueError(f"missing upstream environment next to {path}")
    if data.get("b0_snapshot_checksum") != b0_checksum:
        raise ValueError(f"upstream environment does not reference frozen B0: {path}")
    if data.get("ranking_checksum") != snapshot_checksum(snapshot):
        raise ValueError(f"upstream environment checksum mismatch: {path}")


def _fuse_query(
    query_id: str,
    snapshots: Mapping[str, Mapping[str, object]],
    channels: tuple[str, ...],
    *,
    k: int,
) -> list[dict[str, object]]:
    query_maps = {
        name: {str(query["id"]): query for query in snapshot["queries"]}
        for name, snapshot in snapshots.items()
    }
    original = query_maps["original"][query_id]
    source = {str(item["id"]): item for item in original["candidates"]}
    rankings = []
    for channel in channels:
        ranked = query_maps[channel][query_id]["candidates"]
        rankings.append(
            RankingChannel(
                channel,
                tuple(
                    Candidate(
                        id=str(item["id"]),
                        text=str(item["text"]),
                        retrieval_rank=int(source[str(item["id"])]["rank"]),
                        retrieval_score=float(source[str(item["id"])]["score"]),
                    )
                    for item in ranked
                ),
            )
        )
    fused = ReciprocalRankFusion(k=k).fuse(rankings)
    return [
        {
            **source[item.candidate.id],
            "rank": item.rank,
            "rrf_score": item.score,
            "channel_ranks": item.channel_ranks,
            "contributions": item.contributions,
        }
        for item in fused
    ]


def run(
    baseline_path: Path,
    b1_path: Path | None,
    lexical_path: Path | None,
    pattern_path: Path | None,
    output_dir: Path,
    profile: str,
    *,
    hard: bool = False,
) -> dict[str, float]:
    """Persist one B3 profile using only frozen, already-ranked artifacts."""
    if profile not in PROFILES:
        raise ValueError(f"unknown B3 profile: {profile}")
    baseline = _load_required_snapshot(baseline_path)
    if not hard and snapshot_checksum(baseline) != EXPECTED_B0_CHECKSUM:
        raise ValueError("B3 requires the documented SciFact B0 snapshot checksum")
    channels = PROFILES[profile]
    paths = {"neural": b1_path, "lexical": lexical_path, "pattern": pattern_path}
    snapshots: dict[str, Mapping[str, object]] = {"original": baseline}
    for name in channels:
        if name != "original":
            path = paths[name]
            if path is None:
                raise ValueError(f"profile {profile!r} requires {name!r} ranking")
            upstream = _load_required_snapshot(path)
            if not hard:
                _validate_environment(path, upstream, snapshot_checksum(baseline))
            snapshots[name] = upstream
    _validate_upstream(baseline, snapshots)
    started = perf_counter()
    per_query_timings: list[dict[str, object]] = []
    fused_queries: list[dict[str, Any]] = []
    for query in baseline["queries"]:
        query_started = perf_counter()
        fused_queries.append(
            {
                **query,
                "candidates": _fuse_query(
                    str(query["id"]), snapshots, channels, k=FINAL_K
                ),
            }
        )
        per_query_timings.append(
            {
                "query_id": query["id"],
                "fusion_ms": (perf_counter() - query_started) * 1000,
            }
        )
    ranking: dict[str, Any] = {
        **baseline,
        "b3": {
            "profile": profile,
            "channels": channels,
            "rrf_k": FINAL_K,
            "weights": {name: 1.0 for name in channels},
        },
        "queries": fused_queries,
    }
    _new_directory(output_dir)
    ranking_path = output_dir / "b3-ranking.json"
    ranking_checksum = write_snapshot(ranking, ranking_path)
    metrics = evaluate_snapshot(ranking)
    (output_dir / "b3-metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "b3-config.json").write_text(
        json.dumps(ranking["b3"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    per_query = [
        {
            "query_id": query["id"],
            "metrics": evaluate_snapshot({**ranking, "queries": [query]}),
            "candidates": query["candidates"],
        }
        for query in ranking["queries"]
    ]
    (output_dir / "b3-per-query.json").write_text(
        json.dumps(per_query, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "b3-latency.json").write_text(
        json.dumps(per_query_timings, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if hard:
        (output_dir / "b3-diagnostics.json").write_text(
            json.dumps(_hard_diagnostics(ranking), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elapsed_ms = (perf_counter() - started) * 1000
    timing_values = [float(item["fusion_ms"]) for item in per_query_timings]
    artifact_paths = {"original": baseline_path}
    artifact_paths.update(
        {name: path for name, path in paths.items() if name in channels and path}
    )
    (output_dir / "b3-environment.json").write_text(
        json.dumps(
            {
                "b0_snapshot_checksum": snapshot_checksum(baseline),
                "upstream_ranking_checksums": {
                    name: snapshot_checksum(snapshot)
                    for name, snapshot in snapshots.items()
                    if name in channels
                },
                "upstream_artifacts": {
                    name: {
                        "path": str(path),
                        "checksum": snapshot_checksum(snapshots[name]),
                        "environment": _environment_for(path),
                    }
                    for name, path in artifact_paths.items()
                },
                "ranking_checksum": ranking_checksum,
                "python": platform.python_version(),
                "git_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip(),
                "runtime": {
                    "total_ms": elapsed_ms,
                    "p50_query_fusion_ms": _percentile(timing_values, 0.50),
                    "p95_query_fusion_ms": _percentile(timing_values, 0.95),
                    "queries": len(ranking["queries"]),
                    "candidates": sum(
                        len(query["candidates"]) for query in ranking["queries"]
                    ),
                    "candidates_per_second": sum(
                        len(query["candidates"]) for query in ranking["queries"]
                    )
                    / (elapsed_ms / 1000),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics


def reproduce(ranking_path: Path) -> dict[str, float]:
    """Re-evaluate one persisted B3 ranking without ranking or retrieval work."""
    ranking = _load_required_snapshot(ranking_path)
    metrics = evaluate_snapshot(ranking)
    output_dir = ranking_path.parent
    persisted_metrics = json.loads(
        (output_dir / "b3-metrics.json").read_text(encoding="utf-8")
    )
    if metrics != persisted_metrics:
        raise RuntimeError("persisted B3 ranking metrics do not match its first run")
    diagnostics_path = output_dir / "b3-diagnostics.json"
    if diagnostics_path.exists():
        persisted_diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if _hard_diagnostics(ranking) != persisted_diagnostics:
            raise RuntimeError("persisted B3 hard-set diagnostics do not match")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--b1-ranking", type=Path)
    parser.add_argument("--lexical-ranking", type=Path)
    parser.add_argument("--pattern-ranking", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile", choices=tuple(PROFILES))
    parser.add_argument("--reproduce-ranking", type=Path)
    parser.add_argument("--prepare-hard", action="store_true")
    parser.add_argument("--hard-set", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hard", action="store_true")
    args = parser.parse_args()
    if args.reproduce_ranking:
        print(json.dumps(reproduce(args.reproduce_ranking), sort_keys=True))
        return
    if args.prepare_hard:
        if not all((args.hard_set, args.output_dir, args.model, args.revision)):
            parser.error(
                "--prepare-hard requires hard-set, output-dir, model, and revision"
            )
        print(
            json.dumps(
                prepare_hard(
                    args.hard_set,
                    args.output_dir,
                    model=args.model,
                    revision=args.revision,
                    device=args.device,
                ),
                sort_keys=True,
            )
        )
        return
    if not all((args.baseline, args.output_dir, args.profile)):
        parser.error("--baseline, --output-dir, and --profile are required")
    print(
        json.dumps(
            run(
                args.baseline,
                args.b1_ranking,
                args.lexical_ranking,
                args.pattern_ranking,
                args.output_dir,
                args.profile,
                hard=args.hard,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
