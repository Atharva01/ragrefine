"""Create a reproducible B0/B1/B2/B3 comparison from frozen artifacts."""

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchmarks.beir.metrics import evaluate_snapshot
from benchmarks.beir.snapshot import load_snapshot

PRIMARY_METRICS = ("ndcg@5", "mrr", "precision@5", "recall@5")

SCIFACT_EXPERIMENTS = {
    "B0": Path("benchmarks/results/scifact-b0/snapshot.json"),
    "B1-reference": Path("benchmarks/results/scifact-b1-batched/b1-ranking.json"),
    "B1-light": Path("benchmarks/results/scifact-b1-tinybert/b1-ranking.json"),
    "B2-lexical": Path("benchmarks/results/scifact-b2-lexical/b2-lexical-ranking.json"),
    "B2-pattern": Path(
        "benchmarks/results/scifact-b2-patterns/b2-patterns-ranking.json"
    ),
    "B3-original-lexical": Path(
        "benchmarks/results/scifact-b3-final-v4-original-lexical/b3-ranking.json"
    ),
    "B3-original-pattern": Path(
        "benchmarks/results/scifact-b3-final-v4-original-pattern/b3-ranking.json"
    ),
    "B3-light": Path("benchmarks/results/scifact-b3-final-v4-light/b3-ranking.json"),
    "B3-reference": Path(
        "benchmarks/results/scifact-b3-final-v4-reference/b3-ranking.json"
    ),
}

HARD_EXPERIMENTS = {
    "original": Path("benchmarks/results/hard-b3-upstream-final/original.json"),
    "neural": Path("benchmarks/results/hard-b3-upstream-final/neural.json"),
    "lexical": Path("benchmarks/results/hard-b3-upstream-final/lexical.json"),
    "pattern": Path("benchmarks/results/hard-b3-upstream-final/pattern.json"),
    "B3-original-lexical": Path(
        "benchmarks/results/hard-b3-final-v4-original-lexical/b3-ranking.json"
    ),
    "B3-original-pattern": Path(
        "benchmarks/results/hard-b3-final-v4-original-pattern/b3-ranking.json"
    ),
    "B3-light": Path("benchmarks/results/hard-b3-final-v4-light/b3-ranking.json"),
    "B3-reference": Path(
        "benchmarks/results/hard-b3-final-v4-reference/b3-ranking.json"
    ),
}


def _require_new_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"comparison output directory already exists: {path}")
    path.mkdir(parents=True)


def _per_query_metrics(snapshot: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    return {
        str(query["id"]): evaluate_snapshot({**snapshot, "queries": [query]})
        for query in snapshot["queries"]
    }


def _outcomes(
    baseline: Mapping[str, Mapping[str, float]],
    candidate: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, list[str]]]:
    """Classify every frozen query so aggregate scores cannot hide regressions."""
    result: dict[str, dict[str, list[str]]] = {}
    for metric in PRIMARY_METRICS:
        outcomes = {"wins": [], "losses": [], "unchanged": []}
        for query_id in sorted(baseline):
            baseline_metrics = baseline[query_id]
            delta = candidate[query_id][metric] - baseline_metrics[metric]
            if delta > 0:
                outcomes["wins"].append(query_id)
            elif delta < 0:
                outcomes["losses"].append(query_id)
            else:
                outcomes["unchanged"].append(query_id)
        result[metric] = outcomes
    return result


def _deltas(
    baseline: Mapping[str, float], candidate: Mapping[str, float]
) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "absolute": candidate[metric] - baseline[metric],
            "relative_percent": (
                (candidate[metric] - baseline[metric]) / baseline[metric] * 100
                if baseline[metric]
                else 0.0
            ),
        }
        for metric in PRIMARY_METRICS
    }


def _hard_diagnostics(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in snapshot["queries"]:
        query_id = str(query["id"])
        relevant = next(
            document_id
            for document_id, relevance in snapshot["qrels"][query_id].items()
            if relevance > 0
        )
        rank = next(
            index
            for index, candidate in enumerate(query["candidates"], start=1)
            if candidate["id"] == relevant
        )
        categories[str(query["category"])].append(
            {
                "query_id": query_id,
                "relevant_rank": rank,
                "top_1_correct": rank == 1,
                "mrr": 1 / rank,
            }
        )
    return {
        "query_count": sum(len(rows) for rows in categories.values()),
        "by_category": {
            category: {
                "top_1_accuracy": sum(row["top_1_correct"] for row in rows) / len(rows),
                "mrr": sum(row["mrr"] for row in rows) / len(rows),
                "mean_relevant_rank": sum(row["relevant_rank"] for row in rows)
                / len(rows),
            }
            for category, rows in sorted(categories.items())
        },
    }


def _hard_deltas(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    return {
        category: {
            metric: candidate["by_category"][category][metric]
            - baseline["by_category"][category][metric]
            for metric in ("top_1_accuracy", "mrr", "mean_relevant_rank")
        }
        for category in baseline["by_category"]
    }


def _representatives(
    neural: Mapping[str, Mapping[str, float]],
    fused: Mapping[str, Mapping[str, float]],
    queries: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    """Choose deterministic examples of fusion help, harm, and no change."""

    def example(predicate: Any) -> dict[str, Any] | None:
        for query_id in sorted(queries, key=int):
            if predicate(query_id):
                return {
                    "query_id": query_id,
                    "query_text": queries[query_id]["text"],
                    "neural": neural[query_id],
                    "b3_reference": fused[query_id],
                }
        return None

    return {
        "fusion_fixes_neural_mistake": example(
            lambda query_id: fused[query_id]["ndcg@5"] > neural[query_id]["ndcg@5"]
        ),
        "fusion_overrides_correct_neural_ranking": example(
            lambda query_id: (
                neural[query_id]["mrr"] == 1.0 and fused[query_id]["mrr"] < 1.0
            )
        ),
        "fusion_adds_no_measurable_value": example(
            lambda query_id: neural[query_id] == fused[query_id]
        ),
    }


def _runtime_profiles() -> dict[str, dict[str, Any]]:
    lexical = json.loads(
        Path(
            "benchmarks/results/scifact-b2-lexical/b2-lexical-environment.json"
        ).read_text(encoding="utf-8")
    )["runtime"]
    pattern = json.loads(
        Path(
            "benchmarks/results/scifact-b2-patterns/b2-patterns-environment.json"
        ).read_text(encoding="utf-8")
    )["runtime"]
    b1 = json.loads(
        Path("benchmarks/results/scifact-b1-batched/b1-analysis.json").read_text(
            encoding="utf-8"
        )
    )["reranking_cost"]
    b3_light = json.loads(
        Path(
            "benchmarks/results/scifact-b3-final-v4-light/b3-environment.json"
        ).read_text(encoding="utf-8")
    )["runtime"]
    b3_reference = json.loads(
        Path(
            "benchmarks/results/scifact-b3-final-v4-reference/b3-environment.json"
        ).read_text(encoding="utf-8")
    )["runtime"]

    def combined(*parts: Mapping[str, float]) -> dict[str, float]:
        total_ms = sum(part["total_ms"] for part in parts)
        candidates = int(parts[0]["candidates"])
        return {
            "total_ms": total_ms,
            "p50_query_ms": sum(
                part.get("p50_query_ms", part.get("p50_query_fusion_ms", 0.0))
                for part in parts
            ),
            "p95_query_ms": sum(
                part.get("p95_query_ms", part.get("p95_query_fusion_ms", 0.0))
                for part in parts
            ),
            "candidates": candidates,
            "candidates_per_second": candidates / (total_ms / 1000),
        }

    neural_runtime = {
        "total_ms": b1["total_reranking_seconds"] * 1000,
        "p50_query_ms": b1["p50_query_latency_ms"],
        "p95_query_ms": b1["p95_query_latency_ms"],
        "candidates": b1["candidates_processed"],
        "candidates_per_second": b1["query_document_pairs_per_second"],
    }
    return {
        "B3-light": {
            "profile": "CPU / stdlib lexical + pattern + RRF",
            "timing_kind": "additive component estimate",
            **combined(lexical, pattern, b3_light),
        },
        "B3-reference": {
            "profile": "CUDA SentenceTransformers neural + CPU stdlib signals + RRF",
            "timing_kind": "additive component estimate",
            **combined(neural_runtime, lexical, pattern, b3_reference),
        },
    }


def _decisions(metrics: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, str]]:
    comparisons = {
        "B3-original-lexical": "B2-lexical",
        "B3-original-pattern": "B2-lexical",
        "B3-light": "B2-lexical",
        "B3-reference": "B1-reference",
    }
    for b3_profile, comparator in comparisons.items():
        if not all(
            metrics[b3_profile][metric] < metrics[comparator][metric]
            for metric in ("ndcg@5", "mrr", "recall@5")
        ):
            raise ValueError(
                f"measured decision no longer supports rejecting {b3_profile} "
                f"against {comparator}"
            )
    return {
        "B3-original-lexical": {
            "decision": "reject",
            "reason": (
                "SciFact nDCG@5, MRR, and Recall@5 are lower than B2-lexical, "
                "which uses fewer stages."
            ),
        },
        "B3-original-pattern": {
            "decision": "reject",
            "reason": (
                "Its small B0 gain adds no hard-set separation and is dominated by "
                "retained B2-lexical on SciFact."
            ),
        },
        "B3-light": {
            "decision": "reject",
            "reason": (
                "SciFact nDCG@5, MRR, and Recall@5 are lower than B2-lexical; "
                "it also adds pattern plus fusion cost."
            ),
        },
        "B3-reference": {
            "decision": "reject",
            "reason": (
                "It regresses every primary SciFact metric relative to B1-reference "
                "and adds CPU signal cost; the hard set is saturated."
            ),
        },
        "recommended_default": {
            "decision": "retain B1-reference when CUDA latency is acceptable",
            "reason": (
                "B1-reference has the strongest measured SciFact metrics. Use "
                "B2-lexical for a CPU-only low-cost deployment."
            ),
        },
    }


def _markdown(report: Mapping[str, Any]) -> str:
    rows = []
    for name, result in report["scifact"]["experiments"].items():
        metrics = result["metrics"]
        rows.append(
            f"| {name} | {metrics['ndcg@5']:.4f} | {metrics['mrr']:.4f} | "
            f"{metrics['precision@5']:.4f} | {metrics['recall@5']:.4f} |"
        )
    decision_rows = [
        f"| {name} | {value['decision']} | {value['reason']} |"
        for name, value in report["decisions"].items()
        if name != "recommended_default"
    ]
    return "\n".join(
        (
            "# B3 Consolidated Experiment Analysis",
            "",
            "Machine-readable artifacts are the source of truth. This report is "
            "generated from frozen rankings; it does not rerun retrieval or neural "
            "inference.",
            "",
            "## SciFact comparison",
            "",
            "| Experiment | nDCG@5 | MRR | Precision@5 | Recall@5 |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            "## B3 decision",
            "",
            "| Profile | Decision | Evidence |",
            "|---|---|---|",
            *decision_rows,
            "",
            "Recommended default: "
            f"**{report['decisions']['recommended_default']['decision']}** — "
            f"{report['decisions']['recommended_default']['reason']}",
            "",
            "## Hard-negative interpretation",
            "",
            "All frozen hard-negative categories are saturated at Top-1/MRR 1.0 "
            "for the original and evaluated B3 rankings; they provide no measured "
            "evidence that fusion adds robustness for this set.",
            "",
            "## Runtime",
            "",
            "The B3 profile times are additive component estimates. They are "
            "deployment observations, not retrieval-quality evidence.",
            "",
        )
    )


def run(output_dir: Path) -> dict[str, Any]:
    """Build the RRF-49 comparison without modifying any experiment input."""
    _require_new_directory(output_dir)
    snapshots = {
        name: load_snapshot(path) for name, path in SCIFACT_EXPERIMENTS.items()
    }
    metrics = {
        name: evaluate_snapshot(snapshot) for name, snapshot in snapshots.items()
    }
    per_query = {
        name: _per_query_metrics(snapshot) for name, snapshot in snapshots.items()
    }
    baseline = metrics["B0"]
    hard = {
        name: _hard_diagnostics(load_snapshot(path))
        for name, path in HARD_EXPERIMENTS.items()
    }
    report = {
        "schema_version": "1.0",
        "scifact": {
            "frozen_b0_checksum": (
                "fc08cf7b496c8cd7c9020a81560d08edc10f389b27682f2b6e722ccfef0793dc"
            ),
            "experiments": {
                name: {
                    "artifact": str(SCIFACT_EXPERIMENTS[name]),
                    "metrics": value,
                    "delta_vs_b0": _deltas(baseline, value),
                    "per_query_vs_b0": _outcomes(per_query["B0"], per_query[name]),
                }
                for name, value in metrics.items()
            },
            "representative_b3_reference_cases": _representatives(
                per_query["B1-reference"],
                per_query["B3-reference"],
                {
                    str(query["id"]): query
                    for query in snapshots["B3-reference"]["queries"]
                },
            ),
        },
        "hard_negative": {
            "experiments": hard,
            "delta_vs_original": {
                name: _hard_deltas(hard["original"], diagnostics)
                for name, diagnostics in hard.items()
                if name != "original"
            },
        },
        "runtime": _runtime_profiles(),
        "decisions": _decisions(metrics),
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir)["decisions"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
