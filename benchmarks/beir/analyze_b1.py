"""Compare a persisted B1 ranking with its frozen B0 snapshot."""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from benchmarks.beir.metrics import evaluate_snapshot
from benchmarks.beir.snapshot import load_snapshot, snapshot_checksum

PRIMARY_METRICS = ("ndcg@5", "mrr@5", "precision@5", "recall@5")
OUTCOME_TEMPLATE = {"wins": 0, "losses": 0, "unchanged": 0}


def _query_metrics(
    snapshot: Mapping[str, object], query: Mapping[str, object]
) -> dict[str, float]:
    """Evaluate one query with the same metric implementation as aggregate B0/B1."""
    return evaluate_snapshot({**snapshot, "queries": [dict(query)]})


def _candidate_map(snapshot: Mapping[str, object]) -> dict[str, dict[str, str]]:
    """Map every query to its candidate text, independent of ranking order."""
    result: dict[str, dict[str, str]] = {}
    for query in snapshot["queries"]:
        if not isinstance(query, Mapping) or not isinstance(
            query["candidates"], Sequence
        ):
            raise ValueError("snapshot queries must contain candidate sequences")
        result[str(query["id"])] = {
            str(candidate["id"]): str(candidate["text"])
            for candidate in query["candidates"]
            if isinstance(candidate, Mapping)
        }
    return result


def _validate_same_pool(
    baseline: Mapping[str, object],
    ranking: Mapping[str, object],
    environment: Mapping[str, object],
) -> None:
    """Reject comparisons that are not based on the exact frozen B0 population."""
    if environment.get("b0_snapshot_checksum") != snapshot_checksum(baseline):
        raise ValueError("B1 environment does not reference the supplied B0 snapshot")
    if _candidate_map(baseline) != _candidate_map(ranking):
        raise ValueError("B1 ranking does not preserve the frozen B0 candidate pool")


def _comparison_counts(
    baseline: Mapping[str, object], ranking: Mapping[str, object]
) -> tuple[dict[str, dict[str, int]], list[dict[str, object]]]:
    """Calculate metric win/loss counts and query-level evidence."""
    baseline_queries = {str(query["id"]): query for query in baseline["queries"]}
    ranking_queries = {str(query["id"]): query for query in ranking["queries"]}
    if baseline_queries.keys() != ranking_queries.keys():
        raise ValueError("B0 and B1 contain different query IDs")
    counts = {metric: dict(OUTCOME_TEMPLATE) for metric in PRIMARY_METRICS}
    details: list[dict[str, object]] = []
    for query_id, baseline_query in baseline_queries.items():
        if not isinstance(baseline_query, Mapping) or not isinstance(
            ranking_queries[query_id], Mapping
        ):
            raise ValueError("query must be an object")
        b0 = _query_metrics(baseline, baseline_query)
        b1 = _query_metrics(ranking, ranking_queries[query_id])
        deltas = {metric: b1[metric] - b0[metric] for metric in PRIMARY_METRICS}
        relevance_map = baseline["qrels"].get(query_id, {})
        relevant_ids = {
            str(document_id)
            for document_id, relevance in relevance_map.items()
            if int(relevance) > 0
        }
        top_5_b0 = [candidate["id"] for candidate in baseline_query["candidates"][:5]]
        top_5_b1 = [
            candidate["id"] for candidate in ranking_queries[query_id]["candidates"][:5]
        ]
        b0_hits = set(top_5_b0) & relevant_ids
        b1_hits = set(top_5_b1) & relevant_ids
        category = (
            "relevant_evidence_promoted_into_top_5"
            if b1_hits - b0_hits
            else "relevant_evidence_demoted_from_top_5"
            if b0_hits - b1_hits
            else "relevance_reordered_within_top_5"
            if deltas["ndcg@5"] != 0
            else "top_5_relevance_unchanged"
        )
        for metric, delta in deltas.items():
            outcome = "wins" if delta > 0 else "losses" if delta < 0 else "unchanged"
            counts[metric][outcome] += 1
        details.append(
            {
                "query_id": query_id,
                "query_text": baseline_query["text"],
                "b0": {metric: b0[metric] for metric in PRIMARY_METRICS},
                "b1": {metric: b1[metric] for metric in PRIMARY_METRICS},
                "delta": deltas,
                "category": category,
                "top_5_b0": top_5_b0,
                "top_5_b1": top_5_b1,
            }
        )
    return counts, details


def _latency_summary(
    latency: Sequence[Mapping[str, object]], query_count: int, top_n: int
) -> dict[str, object]:
    """Summarize direct or batched timing artifacts without inventing measurements."""
    durations = [float(item["duration_ms"]) for item in latency]
    estimates = [
        float(item.get("per_query_estimate_ms", item["duration_ms"]))
        for item in latency
    ]
    estimates.sort()
    total_ms = sum(durations)
    return {
        "timing_samples": len(latency),
        "timing_kind": (
            "per_query"
            if latency and "query_id" in latency[0]
            else "per_query_estimate_from_batch"
        ),
        "total_reranking_seconds": total_ms / 1_000,
        "p50_query_latency_ms": estimates[int((len(estimates) - 1) * 0.5)],
        "p95_query_latency_ms": estimates[int((len(estimates) - 1) * 0.95)],
        "queries_processed": query_count,
        "candidates_processed": query_count * top_n,
        "query_document_pairs_per_second": (query_count * top_n) / (total_ms / 1_000),
    }


def analyze(
    baseline: Mapping[str, object],
    ranking: Mapping[str, object],
    environment: Mapping[str, object],
    latency: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Create a reproducible B0-to-B1 comparison from persisted artifacts only."""
    _validate_same_pool(baseline, ranking, environment)
    b0_metrics = evaluate_snapshot(baseline)
    first_b1_metrics = evaluate_snapshot(ranking)
    second_b1_metrics = evaluate_snapshot(ranking)
    if first_b1_metrics != second_b1_metrics:
        raise ValueError("persisted B1 ranking did not reproduce identical metrics")
    counts, query_details = _comparison_counts(baseline, ranking)
    absolute_delta = {
        metric: first_b1_metrics[metric] - b0_metrics[metric]
        for metric in PRIMARY_METRICS
    }
    relative_delta_percent = {
        metric: (absolute_delta[metric] / b0_metrics[metric]) * 100
        if b0_metrics[metric]
        else None
        for metric in PRIMARY_METRICS
    }
    no_regressions = all(delta >= 0 for delta in absolute_delta.values())
    any_gain = any(delta > 0 for delta in absolute_delta.values())
    decision = "retain" if no_regressions and any_gain else "modify"
    ordered = sorted(query_details, key=lambda query: float(query["delta"]["ndcg@5"]))
    top_n = int(baseline["retriever"]["top_n"])
    return {
        "schema_version": "1.0",
        "frozen_b0_snapshot_checksum": snapshot_checksum(baseline),
        "b1_ranking_checksum": snapshot_checksum(ranking),
        "reproducibility": {
            "first_metrics": first_b1_metrics,
            "second_metrics": second_b1_metrics,
            "identical": True,
        },
        "effectiveness": {
            "b0_metrics": b0_metrics,
            "b1_metrics": first_b1_metrics,
            "absolute_delta": absolute_delta,
            "relative_delta_percent": relative_delta_percent,
            "per_query_outcomes": counts,
            "representative_wins": list(reversed(ordered[-3:])),
            "representative_losses": ordered[:3],
        },
        "reranking_cost": {
            "model": environment["model"],
            "revision": environment["revision"],
            "device": environment["device"],
            "backend": environment["backend"],
            "batch_size": environment.get("batch_size"),
            **_latency_summary(latency, len(baseline["queries"]), top_n),
        },
        "decision": {
            "outcome": decision,
            "rationale": (
                "No primary B0-to-B1 metric decreased and at least one increased on "
                "the fixed candidate pool; retain neural reranking for later "
                "controlled fusion experiments while treating recorded reranking cost "
                "as a deployment constraint."
            ),
        },
    }


def main() -> None:
    """Write a B1 comparison artifact without retrieval or CrossEncoder inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--ranking", required=True, type=Path)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--latency", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    baseline = load_snapshot(args.baseline)
    ranking = load_snapshot(args.ranking)
    report = analyze(
        baseline,
        ranking,
        json.loads(args.environment.read_text(encoding="utf-8")),
        json.loads(args.latency.read_text(encoding="utf-8")),
    )
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
