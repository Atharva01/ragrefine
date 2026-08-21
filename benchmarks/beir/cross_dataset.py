"""Compare retained B0, B1-reference, and B2-L profiles across BEIR datasets."""

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.beir import secondary
from benchmarks.beir.artifacts import verify as verify_artifact_bundle
from benchmarks.beir.snapshot import load_verified_snapshot, snapshot_checksum

PRIMARY_METRICS = secondary.PRIMARY_METRICS
PROFILE_ORDER = ("B0", "B1-reference", "B2-L")
QUALIFIED_SNAPSHOTS = {
    "scifact": "fc08cf7b496c8cd7c9020a81560d08edc10f389b27682f2b6e722ccfef0793dc",
    **secondary.QUALIFIED_SNAPSHOTS,
}


@dataclass(frozen=True, slots=True)
class DatasetPaths:
    """One immutable set of checked artifacts for a qualified dataset."""

    baseline: Path
    b1_ranking: Path
    b1_environment: Path
    b2_ranking: Path
    b2_environment: Path


DEFAULT_INPUTS = {
    "scifact": DatasetPaths(
        Path("benchmarks/results/scifact-b0/snapshot.json"),
        Path("benchmarks/results/scifact-b1-batched/b1-ranking.json"),
        Path("benchmarks/results/scifact-b1-batched/b1-environment.json"),
        Path("benchmarks/results/scifact-b2-lexical/b2-lexical-ranking.json"),
        Path("benchmarks/results/scifact-b2-lexical/b2-lexical-environment.json"),
    ),
    "nfcorpus": DatasetPaths(
        Path("benchmarks/results/nfcorpus-b0-v1/snapshot.json"),
        Path("benchmarks/results/nfcorpus-b1-reference-v1/b1-ranking.json"),
        Path("benchmarks/results/nfcorpus-b1-reference-v1/b1-environment.json"),
        Path("benchmarks/results/nfcorpus-b2-lexical-v1/b2-lexical-ranking.json"),
        Path("benchmarks/results/nfcorpus-b2-lexical-v1/b2-lexical-environment.json"),
    ),
    "fiqa": DatasetPaths(
        Path("benchmarks/results/fiqa-b0-v2/snapshot.json"),
        Path("benchmarks/results/fiqa-b1-reference-v1/b1-ranking.json"),
        Path("benchmarks/results/fiqa-b1-reference-v1/b1-environment.json"),
        Path("benchmarks/results/fiqa-b2-lexical-v1/b2-lexical-ranking.json"),
        Path("benchmarks/results/fiqa-b2-lexical-v1/b2-lexical-environment.json"),
    ),
}
RETAINED_ARTIFACT_MANIFEST = Path("benchmarks/artifacts/retained-v1.json")
RETAINED_ARTIFACT_ROOT = Path("benchmarks")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_artifacts(
    paths: DatasetPaths,
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    return (
        load_verified_snapshot(paths.baseline),
        load_verified_snapshot(paths.b1_ranking),
        load_verified_snapshot(paths.b2_ranking),
        _read_json(paths.b1_environment),
        _read_json(paths.b2_environment),
    )


def _query_map(snapshot: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    queries = snapshot.get("queries")
    if not isinstance(queries, Sequence):
        raise ValueError("snapshot queries must be a sequence")
    return {str(query["id"]): query for query in queries if isinstance(query, Mapping)}


def _top_candidates(
    snapshot: Mapping[str, object], query_id: str
) -> list[dict[str, object]]:
    query = _query_map(snapshot)[query_id]
    candidates = query.get("candidates")
    if not isinstance(candidates, Sequence):
        raise ValueError(f"query {query_id} lacks candidates")
    qrels = snapshot.get("qrels")
    if not isinstance(qrels, Mapping):
        raise ValueError("snapshot lacks qrels")
    relevance = qrels.get(query_id, {})
    if not isinstance(relevance, Mapping):
        raise ValueError(f"query {query_id} relevance labels must be an object")
    ranked = sorted(
        (candidate for candidate in candidates if isinstance(candidate, Mapping)),
        key=lambda candidate: (int(candidate["rank"]), str(candidate["id"])),
    )
    if len(ranked) != len(candidates):
        raise ValueError(f"query {query_id} candidate must be an object")
    return [
        {
            "id": str(candidate["id"]),
            "rank": int(candidate["rank"]),
            "relevance": int(relevance.get(str(candidate["id"]), 0)),
            "text_preview": " ".join(str(candidate["text"]).split())[:240],
        }
        for candidate in ranked[:5]
    ]


def _representative_cases(
    dataset: str,
    per_query: Sequence[Mapping[str, object]],
    snapshots: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for profile in ("B1-reference", "B2-L"):
        positive = sorted(
            (
                row
                for row in per_query
                if float(row["delta_vs_b0"][profile]["ndcg@5"]) > 0
            ),
            key=lambda row: float(row["delta_vs_b0"][profile]["ndcg@5"]),
            reverse=True,
        )
        negative = sorted(
            (
                row
                for row in per_query
                if float(row["delta_vs_b0"][profile]["ndcg@5"]) < 0
            ),
            key=lambda row: float(row["delta_vs_b0"][profile]["ndcg@5"]),
        )
        for direction, rows in (("improvement", positive), ("regression", negative)):
            if not rows:
                continue
            row = rows[0]
            query_id = str(row["query_id"])
            cases.append(
                {
                    "dataset": dataset,
                    "profile": profile,
                    "direction": direction,
                    "metric": "ndcg@5",
                    "query_id": query_id,
                    "query_text": row["query_text"],
                    "metrics": {
                        "B0": row["metrics"]["B0"],
                        profile: row["metrics"][profile],
                    },
                    "delta_vs_b0": row["delta_vs_b0"][profile],
                    "top_5": {
                        "B0": _top_candidates(snapshots["B0"], query_id),
                        profile: _top_candidates(snapshots[profile], query_id),
                    },
                }
            )
    return cases


def _profile_orderings(
    metrics: Mapping[str, Mapping[str, float]],
) -> dict[str, list[dict[str, float | str]]]:
    return {
        metric: [
            {"profile": profile, "value": metrics[profile][metric]}
            for profile in sorted(
                PROFILE_ORDER,
                key=lambda profile: (
                    -metrics[profile][metric],
                    PROFILE_ORDER.index(profile),
                ),
            )
        ]
        for metric in PRIMARY_METRICS
    }


def _classification(
    dataset_summaries: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    metrics_by_dataset = {
        dataset: summary["retrieval_quality"]["absolute_delta_vs_b0"]
        for dataset, summary in dataset_summaries.items()
    }
    b1_consistent = all(
        all(float(values["B1-reference"][metric]) > 0 for metric in PRIMARY_METRICS)
        for values in metrics_by_dataset.values()
    )
    b2_improving = sorted(
        dataset
        for dataset, values in metrics_by_dataset.items()
        if all(float(values["B2-L"][metric]) > 0 for metric in PRIMARY_METRICS)
    )
    b2_regressing = sorted(
        dataset
        for dataset, values in metrics_by_dataset.items()
        if any(float(values["B2-L"][metric]) < 0 for metric in PRIMARY_METRICS)
    )
    return {
        "B1-reference": {
            "classification": (
                "consistent aggregate improvement"
                if b1_consistent
                else "mixed aggregate effect"
            ),
            "all_primary_metrics_improve_on": sorted(
                dataset
                for dataset, values in metrics_by_dataset.items()
                if all(
                    float(values["B1-reference"][metric]) > 0
                    for metric in PRIMARY_METRICS
                )
            ),
        },
        "B2-L": {
            "classification": (
                "dataset-specific aggregate effect"
                if b2_improving and b2_regressing
                else "consistent aggregate improvement"
                if b2_improving
                else "aggregate regression"
                if b2_regressing
                else "no aggregate change"
            ),
            "all_primary_metrics_improve_on": b2_improving,
            "one_or_more_primary_metrics_regress_on": b2_regressing,
        },
    }


def _suite_recommendations(
    dataset_summaries: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, str]]:
    recommendations: dict[str, dict[str, str]] = {}
    for dataset, summary in dataset_summaries.items():
        metrics = summary["retrieval_quality"]["metrics"]
        candidate_key = next(
            key for key in metrics["B0"] if key.startswith("candidate_pool_recall")
        )
        ceiling = float(metrics["B0"][candidate_key])
        if dataset == "nfcorpus":
            rationale = (
                "Retain: both profiles improve aggregate Top-5 metrics, while the "
                "low frozen-pool recall ceiling makes candidate-generation limits "
                "visible."
            )
        elif dataset == "fiqa":
            rationale = (
                "Retain: B1-reference improves, while B2-L regresses, providing a "
                "necessary domain-diverse guard against overgeneralizing lexical gains."
            )
        else:
            rationale = (
                "Retain: high candidate-pool availability and gains from both retained "
                "profiles make it a useful controlled scientific-domain reference."
            )
        recommendations[dataset] = {
            "decision": "retain",
            "candidate_pool_recall": ceiling,
            "rationale": rationale,
        }
    return recommendations


def analyze_all(
    inputs: Mapping[str, DatasetPaths],
    *,
    qualified_snapshots: Mapping[str, str] = QUALIFIED_SNAPSHOTS,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    """Analyze qualified frozen datasets without retrieval or neural inference."""
    dataset_summaries: dict[str, Mapping[str, object]] = {}
    representatives: list[dict[str, object]] = []
    manifest: dict[str, object] = {"datasets": {}}
    for expected_dataset, paths in inputs.items():
        baseline, b1, b2, b1_environment, b2_environment = _load_artifacts(paths)
        summary, per_query = secondary.analyze(
            baseline,
            b1,
            b2,
            b1_environment,
            b2_environment,
            qualified_snapshots=qualified_snapshots,
        )
        dataset = str(summary["dataset"])
        if dataset != expected_dataset:
            raise ValueError(
                f"input name {expected_dataset!r} does not match dataset {dataset!r}"
            )
        snapshots = {"B0": baseline, "B1-reference": b1, "B2-L": b2}
        dataset_summaries[dataset] = summary
        representatives.extend(_representative_cases(dataset, per_query, snapshots))
        manifest["datasets"][dataset] = {
            "qualified_b0_snapshot_checksum": qualified_snapshots.get(dataset),
            "baseline": {
                "path": str(paths.baseline),
                "checksum": snapshot_checksum(baseline),
            },
            "B1-reference": {
                "ranking_path": str(paths.b1_ranking),
                "ranking_checksum": snapshot_checksum(b1),
                "environment_path": str(paths.b1_environment),
            },
            "B2-L": {
                "ranking_path": str(paths.b2_ranking),
                "ranking_checksum": snapshot_checksum(b2),
                "environment_path": str(paths.b2_environment),
            },
        }
    ordered_datasets = {
        dataset: {
            "metrics": summary["retrieval_quality"]["metrics"],
            "absolute_delta_vs_b0": summary["retrieval_quality"][
                "absolute_delta_vs_b0"
            ],
            "metric_orderings": _profile_orderings(
                summary["retrieval_quality"]["metrics"]
            ),
            "per_query_outcomes_vs_b0": summary["retrieval_quality"][
                "per_query_outcomes_vs_b0"
            ],
        }
        for dataset, summary in sorted(dataset_summaries.items())
    }
    summary: dict[str, object] = {
        "schema_version": "1.0",
        "datasets": ordered_datasets,
        "stability": _classification(dataset_summaries),
        "benchmark_suite_recommendations": _suite_recommendations(dataset_summaries),
        "scope": {
            "retrieval_or_reranking_executed": False,
            "rejected_profiles_executed": False,
            "runtime_interpretation": (
                "Runtime is intentionally excluded from cross-dataset quality "
                "ordering because B1-reference/CUDA and B2-L/CPU differ in both "
                "model and device."
            ),
        },
    }
    representatives.sort(
        key=lambda item: (
            str(item["dataset"]),
            str(item["profile"]),
            str(item["direction"]),
            str(item["query_id"]),
        )
    )
    return summary, representatives, manifest


def run(
    output_dir: Path, *, inputs: Mapping[str, DatasetPaths] = DEFAULT_INPUTS
) -> dict[str, object]:
    """Write a cross-dataset stability report from existing checked artifacts."""
    if output_dir.exists():
        raise FileExistsError(
            f"cross-dataset output directory already exists: {output_dir}"
        )
    if inputs is DEFAULT_INPUTS:
        verify_artifact_bundle(RETAINED_ARTIFACT_MANIFEST, RETAINED_ARTIFACT_ROOT)
    summary, representatives, manifest = analyze_all(inputs)
    output_dir.mkdir(parents=True)
    _write_json(
        output_dir / "config.json",
        {
            "profiles": list(PROFILE_ORDER),
            "primary_metrics": list(PRIMARY_METRICS),
            "qualified_datasets": sorted(inputs),
            "case_selection": (
                "largest positive and negative nDCG@5 delta per dataset/profile"
            ),
        },
    )
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "representative-cases.json", representatives)
    _write_json(output_dir / "input-manifest.json", manifest)
    return summary


def _inputs_from_manifest(manifest: Mapping[str, object]) -> dict[str, DatasetPaths]:
    datasets = manifest.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError("cross-dataset manifest lacks datasets")
    result: dict[str, DatasetPaths] = {}
    for dataset, value in datasets.items():
        if not isinstance(value, Mapping):
            raise ValueError("cross-dataset manifest dataset entry must be an object")
        baseline = value.get("baseline")
        b1 = value.get("B1-reference")
        b2 = value.get("B2-L")
        if not all(isinstance(item, Mapping) for item in (baseline, b1, b2)):
            raise ValueError("cross-dataset manifest profile entry must be an object")
        result[str(dataset)] = DatasetPaths(
            Path(str(baseline["path"])),
            Path(str(b1["ranking_path"])),
            Path(str(b1["environment_path"])),
            Path(str(b2["ranking_path"])),
            Path(str(b2["environment_path"])),
        )
    return result


def _qualified_snapshots_from_manifest(
    manifest: Mapping[str, object],
) -> dict[str, str]:
    datasets = manifest.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError("cross-dataset manifest lacks datasets")
    qualified: dict[str, str] = {}
    for dataset, value in datasets.items():
        if not isinstance(value, Mapping):
            raise ValueError("cross-dataset manifest dataset entry must be an object")
        checksum = value.get("qualified_b0_snapshot_checksum")
        if not isinstance(checksum, str):
            raise ValueError("cross-dataset manifest lacks a qualified B0 checksum")
        qualified[str(dataset)] = checksum
    return qualified


def reproduce(output_dir: Path) -> dict[str, object]:
    """Recreate the cross-dataset report from exact persisted input artifacts."""
    reproduction_path = output_dir / "reproduction.json"
    if reproduction_path.exists():
        raise FileExistsError(
            f"cross-dataset reproduction already exists: {reproduction_path}"
        )
    manifest = _read_json(output_dir / "input-manifest.json")
    inputs = _inputs_from_manifest(manifest)
    summary, representatives, _ = analyze_all(
        inputs, qualified_snapshots=_qualified_snapshots_from_manifest(manifest)
    )
    matches = summary == _read_json(
        output_dir / "summary.json"
    ) and representatives == json.loads(
        (output_dir / "representative-cases.json").read_text(encoding="utf-8")
    )
    if not matches:
        raise RuntimeError("cross-dataset analysis did not reproduce")
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
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "reproduce":
        print(json.dumps(reproduce(args.output_dir), sort_keys=True))
        return
    summary = run(args.output_dir)
    print(json.dumps(summary["stability"], sort_keys=True))


if __name__ == "__main__":
    main()
