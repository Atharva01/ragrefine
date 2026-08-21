"""Analyze retained profiles on one qualified secondary frozen dataset."""

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.beir.metrics import evaluate_snapshot
from benchmarks.beir.snapshot import load_verified_snapshot, snapshot_checksum

PRIMARY_METRICS = ("ndcg@5", "mrr@5", "precision@5", "recall@5")
B1_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
B1_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
QUALIFIED_SNAPSHOTS = {
    "nfcorpus": "92f56832566e361abd723e5519414a6ba6b821813b3e0bdb9611ee42692b9dbe",
    "fiqa": "a2fe59001a24c97f8213151cc66dbf2c285b98a57fbb4a8cccded1762d814e71",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _dataset_name(snapshot: Mapping[str, object]) -> str:
    dataset = snapshot.get("dataset")
    if not isinstance(dataset, Mapping) or not isinstance(dataset.get("name"), str):
        raise ValueError("snapshot lacks a dataset name")
    return dataset["name"]


def _query_map(snapshot: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    queries = snapshot.get("queries")
    if not isinstance(queries, Sequence):
        raise ValueError("snapshot queries must be a sequence")
    result: dict[str, Mapping[str, object]] = {}
    for query in queries:
        if not isinstance(query, Mapping):
            raise ValueError("snapshot query must be an object")
        query_id = str(query["id"])
        if query_id in result:
            raise ValueError(f"duplicate query ID: {query_id}")
        result[query_id] = query
    return result


def _candidate_population(
    snapshot: Mapping[str, object],
) -> dict[str, tuple[tuple[str, str], ...]]:
    population: dict[str, tuple[tuple[str, str], ...]] = {}
    for query_id, query in _query_map(snapshot).items():
        candidates = query.get("candidates")
        if not isinstance(candidates, Sequence):
            raise ValueError(f"query {query_id} candidates must be a sequence")
        values = tuple(
            sorted(
                (str(candidate["id"]), str(candidate["text"]))
                for candidate in candidates
                if isinstance(candidate, Mapping)
            )
        )
        if len(values) != len(candidates):
            raise ValueError(f"query {query_id} candidate must be an object")
        if len({candidate_id for candidate_id, _ in values}) != len(values):
            raise ValueError(f"query {query_id} contains duplicate candidate IDs")
        population[query_id] = values
    return population


def validate_artifacts(
    baseline: Mapping[str, object],
    b1: Mapping[str, object],
    b2: Mapping[str, object],
    b1_environment: Mapping[str, object],
    b2_environment: Mapping[str, object],
    *,
    qualified_snapshots: Mapping[str, str] = QUALIFIED_SNAPSHOTS,
) -> str:
    """Validate profile identity, provenance, and frozen candidate integrity."""
    dataset = _dataset_name(baseline)
    baseline_checksum = snapshot_checksum(baseline)
    if qualified_snapshots.get(dataset) != baseline_checksum:
        raise ValueError(f"unqualified secondary snapshot: {dataset}")
    expected_population = _candidate_population(baseline)
    expected_qrels = baseline.get("qrels")
    for name, ranking in (("B1-reference", b1), ("B2-L", b2)):
        if _dataset_name(ranking) != dataset:
            raise ValueError(f"{name} uses a different dataset")
        if ranking.get("qrels") != expected_qrels:
            raise ValueError(f"{name} uses different relevance labels")
        if _candidate_population(ranking) != expected_population:
            raise ValueError(f"{name} does not preserve the frozen candidate pool")
    if b1_environment.get("b0_snapshot_checksum") != baseline_checksum:
        raise ValueError("B1-reference does not reference the supplied B0 snapshot")
    if b2_environment.get("b0_snapshot_checksum") != baseline_checksum:
        raise ValueError("B2-L does not reference the supplied B0 snapshot")
    if b1_environment.get("ranking_checksum") != snapshot_checksum(b1):
        raise ValueError("B1-reference ranking checksum mismatch")
    if b2_environment.get("ranking_checksum") != snapshot_checksum(b2):
        raise ValueError("B2-L ranking checksum mismatch")
    if (
        b1_environment.get("model") != B1_MODEL
        or b1_environment.get("revision") != B1_REVISION
        or b1_environment.get("backend") != "sentence-transformers"
        or b1_environment.get("device") != "cuda"
    ):
        raise ValueError("B1 artifact is not the retained CUDA reference profile")
    b1_config = b1.get("b1")
    if not isinstance(b1_config, Mapping) or (
        b1_config.get("model") != B1_MODEL or b1_config.get("revision") != B1_REVISION
    ):
        raise ValueError("B1 ranking lacks retained model provenance")
    b2_config = b2.get("b2")
    if not isinstance(b2_config, Mapping) or b2_config.get("signal") != "lexical":
        raise ValueError("B2 ranking is not the retained lexical profile")
    if (
        b2_environment.get("signal") != "lexical"
        or b2_environment.get("backend") != "stdlib"
        or b2_environment.get("device") != "cpu"
    ):
        raise ValueError("B2 artifact is not the retained lexical CPU profile")
    return dataset


def _query_metrics(
    snapshot: Mapping[str, object], query: Mapping[str, object]
) -> dict[str, float]:
    return evaluate_snapshot({**snapshot, "queries": [dict(query)]})


def analyze(
    baseline: Mapping[str, object],
    b1: Mapping[str, object],
    b2: Mapping[str, object],
    b1_environment: Mapping[str, object],
    b2_environment: Mapping[str, object],
    *,
    qualified_snapshots: Mapping[str, str] = QUALIFIED_SNAPSHOTS,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Build aggregate and per-query quality evidence from persisted rankings."""
    dataset = validate_artifacts(
        baseline,
        b1,
        b2,
        b1_environment,
        b2_environment,
        qualified_snapshots=qualified_snapshots,
    )
    snapshots = {"B0": baseline, "B1-reference": b1, "B2-L": b2}
    metrics = {name: evaluate_snapshot(value) for name, value in snapshots.items()}
    profile_deltas = {
        name: {
            metric: value - metrics["B0"][metric]
            for metric, value in profile_metrics.items()
        }
        for name, profile_metrics in metrics.items()
        if name != "B0"
    }
    outcomes = {
        profile: {
            metric: {"wins": 0, "losses": 0, "unchanged": 0}
            for metric in PRIMARY_METRICS
        }
        for profile in ("B1-reference", "B2-L")
    }
    query_maps = {name: _query_map(value) for name, value in snapshots.items()}
    per_query: list[dict[str, object]] = []
    for query_id, baseline_query in query_maps["B0"].items():
        query_metrics = {
            name: _query_metrics(snapshots[name], query_maps[name][query_id])
            for name in snapshots
        }
        deltas: dict[str, dict[str, float]] = {}
        query_outcomes: dict[str, dict[str, str]] = {}
        for profile in ("B1-reference", "B2-L"):
            deltas[profile] = {
                metric: query_metrics[profile][metric] - query_metrics["B0"][metric]
                for metric in PRIMARY_METRICS
            }
            query_outcomes[profile] = {}
            for metric, delta in deltas[profile].items():
                outcome = "win" if delta > 0 else "loss" if delta < 0 else "unchanged"
                query_outcomes[profile][metric] = outcome
                outcome_key = {
                    "win": "wins",
                    "loss": "losses",
                    "unchanged": "unchanged",
                }[outcome]
                outcomes[profile][metric][outcome_key] += 1
        per_query.append(
            {
                "query_id": query_id,
                "query_text": baseline_query["text"],
                "metrics": query_metrics,
                "delta_vs_b0": deltas,
                "outcome_vs_b0": query_outcomes,
            }
        )
    regressions = {
        profile: [
            {"metric": metric, "absolute_delta": delta}
            for metric, delta in deltas.items()
            if metric in PRIMARY_METRICS and delta < 0
        ]
        for profile, deltas in profile_deltas.items()
    }
    summary: dict[str, object] = {
        "schema_version": "1.0",
        "dataset": dataset,
        "query_count": len(per_query),
        "frozen_b0_snapshot_checksum": snapshot_checksum(baseline),
        "retrieval_quality": {
            "metrics": metrics,
            "absolute_delta_vs_b0": profile_deltas,
            "per_query_outcomes_vs_b0": outcomes,
            "dataset_regressions": regressions,
        },
        "runtime_observations": {
            "interpretation": (
                "Deployment observations only; runtime is not retrieval-quality "
                "evidence and B1 CUDA is not comparable to B2-L CPU as a pure "
                "backend benchmark."
            ),
            "B1-reference": {
                "model": b1_environment["model"],
                "revision": b1_environment["revision"],
                "backend": b1_environment["backend"],
                "device": b1_environment["device"],
                "batch_size": b1_environment.get("batch_size"),
                "queries_per_batch": b1_environment.get("queries_per_batch"),
                "reranking_wall_clock": b1_environment.get("reranking_wall_clock"),
                "run_wall_clock": b1_environment.get("run_wall_clock"),
            },
            "B2-L": {
                "backend": b2_environment["backend"],
                "device": b2_environment["device"],
                "runtime": b2_environment.get("runtime"),
            },
        },
    }
    return summary, per_query


def run(
    baseline_path: Path,
    b1_ranking_path: Path,
    b1_environment_path: Path,
    b2_ranking_path: Path,
    b2_environment_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Analyze one qualified dataset and persist complete comparison artifacts."""
    if output_dir.exists():
        raise FileExistsError(
            f"secondary result directory already exists: {output_dir}"
        )
    baseline = load_verified_snapshot(baseline_path)
    b1 = load_verified_snapshot(b1_ranking_path)
    b2 = load_verified_snapshot(b2_ranking_path)
    b1_environment = _read_json(b1_environment_path)
    b2_environment = _read_json(b2_environment_path)
    summary, per_query = analyze(baseline, b1, b2, b1_environment, b2_environment)
    manifest = {
        "baseline": {
            "path": str(baseline_path),
            "checksum": snapshot_checksum(baseline),
        },
        "B1-reference": {
            "ranking_path": str(b1_ranking_path),
            "ranking_checksum": snapshot_checksum(b1),
            "environment_path": str(b1_environment_path),
        },
        "B2-L": {
            "ranking_path": str(b2_ranking_path),
            "ranking_checksum": snapshot_checksum(b2),
            "environment_path": str(b2_environment_path),
        },
    }
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "per-query.json", per_query)
    _write_json(output_dir / "input-manifest.json", manifest)
    return summary


def reproduce(output_dir: Path) -> dict[str, object]:
    """Recompute the comparison from persisted rankings without model execution."""
    reproduction_path = output_dir / "reproduction.json"
    if reproduction_path.exists():
        raise FileExistsError(
            f"secondary reproduction already exists: {reproduction_path}"
        )
    manifest = _read_json(output_dir / "input-manifest.json")
    baseline_path = Path(manifest["baseline"]["path"])
    b1_data = manifest["B1-reference"]
    b2_data = manifest["B2-L"]
    baseline = load_verified_snapshot(baseline_path)
    b1 = load_verified_snapshot(Path(b1_data["ranking_path"]))
    b2 = load_verified_snapshot(Path(b2_data["ranking_path"]))
    b1_environment = _read_json(Path(b1_data["environment_path"]))
    b2_environment = _read_json(Path(b2_data["environment_path"]))
    summary, per_query = analyze(baseline, b1, b2, b1_environment, b2_environment)
    matches = summary == _read_json(
        output_dir / "summary.json"
    ) and per_query == json.loads(
        (output_dir / "per-query.json").read_text(encoding="utf-8")
    )
    if not matches:
        raise RuntimeError("secondary-dataset comparison did not reproduce")
    reproduction = {
        "created_at": datetime.now(UTC).isoformat(),
        "logical_results_match": True,
        "retrieval_or_reranking_executed": False,
    }
    _write_json(reproduction_path, reproduction)
    return reproduction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "reproduce"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--b1-ranking", type=Path)
    parser.add_argument("--b1-environment", type=Path)
    parser.add_argument("--b2-ranking", type=Path)
    parser.add_argument("--b2-environment", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "reproduce":
        print(json.dumps(reproduce(args.output_dir), sort_keys=True))
        return
    required = (
        args.baseline,
        args.b1_ranking,
        args.b1_environment,
        args.b2_ranking,
        args.b2_environment,
    )
    if any(path is None for path in required):
        parser.error("run requires baseline, B1, and B2 ranking/environment paths")
    summary = run(
        args.baseline,
        args.b1_ranking,
        args.b1_environment,
        args.b2_ranking,
        args.b2_environment,
        args.output_dir,
    )
    print(json.dumps(summary["retrieval_quality"], sort_keys=True))


if __name__ == "__main__":
    main()
