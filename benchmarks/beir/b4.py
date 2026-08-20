"""Evaluate frozen-ranking B4 context-selection ablations."""

import argparse
import json
import platform
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from benchmarks.beir.redundancy import WORD_TOKENIZATION, word_token_count
from benchmarks.beir.snapshot import load_snapshot, snapshot_checksum
from ragrefine import Candidate, RankingSignal, RefinedCandidate
from ragrefine.context import RankPreservingContextSelector, SelectionRecord
from ragrefine.filtering import CandidateDeduplicator, DeduplicationConfig

TOP_K = 5
TOKEN_BUDGET = 1_000
NEAR_DUPLICATE_THRESHOLD = 0.90
SHINGLE_SIZE = 5
MAX_EVIDENCE_RETENTION_DROP = 0.02


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    """One predeclared B4 selection ablation."""

    name: str
    description: str
    deduplication: str | None = None
    max_tokens: int | None = None


POLICIES = (
    SelectionPolicy("S0", "ranking-only control"),
    SelectionPolicy("S1", "exact normalized-content deduplication", "exact"),
    SelectionPolicy("S2", "exact and near token-shingle deduplication", "exact+near"),
    SelectionPolicy("S3", "word-token budget", max_tokens=TOKEN_BUDGET),
    SelectionPolicy(
        "S4",
        "exact and near deduplication plus word-token budget",
        "exact+near",
        TOKEN_BUDGET,
    ),
)


class UnicodeWordTokenCounter:
    """Dependency-free counter used only for this controlled B4 experiment."""

    name = WORD_TOKENIZATION

    def count(self, text: str) -> int:
        """Count Unicode word-like tokens under the frozen audit definition."""
        return word_token_count(text)


def _load_required_snapshot(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists():
        raise ValueError(f"missing required checksum sidecar: {sidecar}")
    return load_snapshot(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _query_map(snapshot: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    return {str(query["id"]): query for query in snapshot["queries"]}


def _candidate_population(
    snapshot: Mapping[str, object],
) -> dict[str, tuple[tuple[str, str], ...]]:
    return {
        query_id: tuple(
            sorted(
                (str(candidate["id"]), str(candidate["text"]))
                for candidate in query["candidates"]
            )
        )
        for query_id, query in _query_map(snapshot).items()
    }


def validate_profiles(
    profiles: Mapping[str, Mapping[str, object]],
    environments: Mapping[str, Mapping[str, object]],
) -> str:
    """Require retained profiles to reference one identical frozen population."""
    if set(profiles) != set(environments):
        raise ValueError("each retained profile requires environment metadata")
    first_name = next(iter(profiles))
    first = profiles[first_name]
    expected_dataset = first.get("dataset")
    expected_qrels = first.get("qrels")
    expected_population = _candidate_population(first)
    b0_checksums: set[str] = set()
    for name, snapshot in profiles.items():
        if snapshot.get("dataset") != expected_dataset:
            raise ValueError(f"{name} uses a different dataset")
        if snapshot.get("qrels") != expected_qrels:
            raise ValueError(f"{name} uses different relevance labels")
        if _candidate_population(snapshot) != expected_population:
            raise ValueError(f"{name} does not preserve the frozen candidate pool")
        environment = environments[name]
        if environment.get("ranking_checksum") != snapshot_checksum(snapshot):
            raise ValueError(f"{name} environment ranking checksum mismatch")
        b0_checksum = environment.get("b0_snapshot_checksum")
        if not isinstance(b0_checksum, str) or not b0_checksum:
            raise ValueError(f"{name} environment lacks a B0 snapshot checksum")
        b0_checksums.add(b0_checksum)
    if len(b0_checksums) != 1:
        raise ValueError("retained profiles do not reference the same frozen B0")
    return b0_checksums.pop()


def _ranked_candidates(query: Mapping[str, object]) -> tuple[RefinedCandidate, ...]:
    candidates = sorted(
        query["candidates"], key=lambda item: (int(item["rank"]), str(item["id"]))
    )
    return tuple(
        RefinedCandidate(
            candidate=Candidate(
                id=str(item["id"]),
                text=str(item["text"]),
                retrieval_rank=int(item["rank"]),
                retrieval_score=float(item["score"]),
            ),
            original_rank=int(item["rank"]),
            final_rank=int(item["rank"]),
            final_score=float(item["score"]),
            signals={
                "retained_profile": RankingSignal(
                    rank=int(item["rank"]), score=float(item["score"])
                )
            },
        )
        for item in candidates
    )


def _apply_policy(
    candidates: tuple[RefinedCandidate, ...], policy: SelectionPolicy
) -> tuple[tuple[RefinedCandidate, ...], tuple[SelectionRecord, ...], int, int]:
    suppressed = ()
    available = candidates
    if policy.deduplication is not None:
        threshold = (
            None if policy.deduplication == "exact" else NEAR_DUPLICATE_THRESHOLD
        )
        deduplicated = CandidateDeduplicator(
            DeduplicationConfig(
                near_duplicate_threshold=threshold,
                shingle_size=SHINGLE_SIZE,
            )
        ).deduplicate(candidates)
        available, suppressed = deduplicated.candidates, deduplicated.suppressed
    selection = RankPreservingContextSelector(UnicodeWordTokenCounter()).select(
        available,
        top_k=TOP_K,
        max_tokens=policy.max_tokens,
        suppressed=suppressed,
    )
    exact = sum(item.reason == "exact_normalized_content" for item in suppressed)
    near = sum(item.reason == "near_token_shingle_jaccard" for item in suppressed)
    return selection.candidates, selection.records, exact, near


def evaluate_profile(
    snapshot: Mapping[str, object],
    *,
    clock: Callable[[], float] = perf_counter,
) -> tuple[dict[str, object], dict[str, object]]:
    """Evaluate every frozen B4 policy for one retained ranking profile."""
    qrels = snapshot["qrels"]
    policy_rows: dict[str, list[dict[str, object]]] = {
        policy.name: [] for policy in POLICIES
    }
    per_query: dict[str, object] = {}
    for query in snapshot["queries"]:
        query_id = str(query["id"])
        candidates = _ranked_candidates(query)
        candidate_ids = {item.candidate.id for item in candidates}
        relevance = qrels.get(query_id, {})
        pool_relevant_ids = {
            str(candidate_id)
            for candidate_id, value in relevance.items()
            if int(value) > 0 and str(candidate_id) in candidate_ids
        }
        input_tokens = sum(word_token_count(item.candidate.text) for item in candidates)
        query_policies: dict[str, object] = {}
        for policy in POLICIES:
            started = clock()
            selected, records, exact, near = _apply_policy(candidates, policy)
            duration_ms = (clock() - started) * 1_000
            selected_ids = tuple(item.candidate.id for item in selected)
            selected_relevant = pool_relevant_ids & set(selected_ids)
            selected_tokens = sum(
                word_token_count(item.candidate.text) for item in selected
            )
            retention = (
                len(selected_relevant) / len(pool_relevant_ids)
                if pool_relevant_ids
                else None
            )
            row = {
                "query_id": query_id,
                "input_candidates": len(candidates),
                "selected_candidates": len(selected),
                "input_word_tokens": input_tokens,
                "selected_word_tokens": selected_tokens,
                "pool_relevant_count": len(pool_relevant_ids),
                "selected_relevant_count": len(selected_relevant),
                "evidence_retention": retention,
                "exact_duplicates_suppressed": exact,
                "near_duplicates_suppressed": near,
                "duration_ms": duration_ms,
            }
            policy_rows[policy.name].append(row)
            query_policies[policy.name] = {
                **row,
                "pool_relevant_ids": sorted(pool_relevant_ids),
                "selected_relevant_ids": sorted(selected_relevant),
                "selected_ids": list(selected_ids),
                "exclusions": [
                    asdict(record) for record in records if record.status != "selected"
                ],
            }
        per_query[query_id] = query_policies
    summaries = {
        policy.name: _summarize(policy_rows[policy.name]) for policy in POLICIES
    }
    _add_control_deltas(summaries)
    return summaries, per_query


def _summarize(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    input_candidates = sum(int(row["input_candidates"]) for row in rows)
    selected_candidates = sum(int(row["selected_candidates"]) for row in rows)
    input_tokens = sum(int(row["input_word_tokens"]) for row in rows)
    selected_tokens = sum(int(row["selected_word_tokens"]) for row in rows)
    pool_relevant = sum(int(row["pool_relevant_count"]) for row in rows)
    selected_relevant = sum(int(row["selected_relevant_count"]) for row in rows)
    retention_values = [
        float(row["evidence_retention"])
        for row in rows
        if row["evidence_retention"] is not None
    ]
    timings = [float(row["duration_ms"]) for row in rows]
    total_ms = sum(timings)
    return {
        "evidence": {
            "eligible_queries": len(retention_values),
            "pool_relevant_instances": pool_relevant,
            "selected_relevant_instances": selected_relevant,
            "micro_retention": _share(selected_relevant, pool_relevant),
            "mean_query_retention": mean(retention_values) if retention_values else 0.0,
        },
        "candidates": {
            "input": input_candidates,
            "selected": selected_candidates,
            "reduction_share": 1 - _share(selected_candidates, input_candidates),
        },
        "word_tokens": {
            "input": input_tokens,
            "selected": selected_tokens,
            "reduction_share": 1 - _share(selected_tokens, input_tokens),
        },
        "duplicates": {
            "exact_suppressed": sum(
                int(row["exact_duplicates_suppressed"]) for row in rows
            ),
            "near_suppressed": sum(
                int(row["near_duplicates_suppressed"]) for row in rows
            ),
        },
        "runtime": {
            "total_ms": total_ms,
            "p50_query_ms": _percentile(timings, 0.50),
            "p95_query_ms": _percentile(timings, 0.95),
            "queries": len(rows),
            "input_candidates_per_second": input_candidates / (total_ms / 1_000)
            if total_ms
            else 0.0,
        },
    }


def _add_control_deltas(summaries: Mapping[str, dict[str, object]]) -> None:
    control = summaries["S0"]
    for summary in summaries.values():
        summary["delta_vs_s0"] = {
            "selected_candidates": int(summary["candidates"]["selected"])
            - int(control["candidates"]["selected"]),
            "selected_word_tokens": int(summary["word_tokens"]["selected"])
            - int(control["word_tokens"]["selected"]),
            "mean_query_evidence_retention": float(
                summary["evidence"]["mean_query_retention"]
            )
            - float(control["evidence"]["mean_query_retention"]),
        }


def decide_policies(
    profile_summaries: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict[str, Mapping[str, object]]:
    """Apply the predeclared decision rule across both retained profiles."""
    decisions: dict[str, Mapping[str, object]] = {
        "S0": {"decision": "retain", "reason": "required ranking-only control"}
    }
    comparators = {"S1": "S0", "S2": "S1", "S3": "S0", "S4": "S3"}
    for policy_name, comparator_name in comparators.items():
        effects: list[bool] = []
        evidence_drops: list[float] = []
        for summaries in profile_summaries.values():
            policy = summaries[policy_name]
            comparator = summaries[comparator_name]
            if policy_name in {"S1", "S2", "S4"}:
                policy_duplicates = int(policy["duplicates"]["exact_suppressed"]) + int(
                    policy["duplicates"]["near_suppressed"]
                )
                comparator_duplicates = int(
                    comparator["duplicates"]["exact_suppressed"]
                ) + int(comparator["duplicates"]["near_suppressed"])
            else:
                policy_duplicates = comparator_duplicates = 0
            token_effect = int(policy["word_tokens"]["selected"]) < int(
                comparator["word_tokens"]["selected"]
            )
            effects.append(policy_duplicates > comparator_duplicates or token_effect)
            evidence_drops.append(
                float(comparator["evidence"]["mean_query_retention"])
                - float(policy["evidence"]["mean_query_retention"])
            )
        max_drop = max(evidence_drops, default=0.0)
        if not any(effects):
            decision, reason = (
                "reject",
                f"no incremental effect over {comparator_name}",
            )
        elif max_drop <= MAX_EVIDENCE_RETENTION_DROP:
            decision, reason = (
                "retain",
                "measurable reduction with evidence-retention drop within 0.02",
            )
        else:
            decision, reason = (
                "modify",
                "measurable reduction but evidence-retention drop exceeds 0.02",
            )
        decisions[policy_name] = {
            "decision": decision,
            "comparator": comparator_name,
            "maximum_mean_query_evidence_retention_drop": max_drop,
            "reason": reason,
        }
    return decisions


def run(
    profile_paths: Mapping[str, Path],
    environment_paths: Mapping[str, Path],
    output_dir: Path,
) -> dict[str, object]:
    """Run B4 once from persisted rankings and write complete artifacts."""
    if output_dir.exists():
        raise FileExistsError(f"B4 output directory already exists: {output_dir}")
    snapshots = {
        name: _load_required_snapshot(path) for name, path in profile_paths.items()
    }
    environments = {name: _read_json(path) for name, path in environment_paths.items()}
    b0_checksum = validate_profiles(snapshots, environments)
    profile_summaries: dict[str, Mapping[str, Mapping[str, object]]] = {}
    per_query: dict[str, object] = {}
    for name, snapshot in snapshots.items():
        summary, queries = evaluate_profile(snapshot)
        profile_summaries[name] = summary
        per_query[name] = queries
    result: dict[str, object] = {
        "profiles": profile_summaries,
        "decisions": decide_policies(profile_summaries),
    }
    config = {
        "top_k": TOP_K,
        "token_counter": {"name": WORD_TOKENIZATION, "budget": TOKEN_BUDGET},
        "deduplication": {
            "exact": "normalized_content_sha256",
            "near": "strict_token_shingle_jaccard",
            "near_threshold": NEAR_DUPLICATE_THRESHOLD,
            "shingle_size": SHINGLE_SIZE,
        },
        "policies": [asdict(policy) for policy in POLICIES],
        "decision_rule": {
            "maximum_evidence_retention_drop": MAX_EVIDENCE_RETENTION_DROP,
            "comparators": {"S1": "S0", "S2": "S1", "S3": "S0", "S4": "S3"},
        },
    }
    manifest = {
        "b0_snapshot_checksum": b0_checksum,
        "profiles": {
            name: {
                "ranking_path": str(profile_paths[name]),
                "ranking_checksum": snapshot_checksum(snapshots[name]),
                "environment_path": str(environment_paths[name]),
                "environment": environments[name],
            }
            for name in snapshots
        },
    }
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "input-manifest.json", manifest)
    _write_json(output_dir / "summary.json", result)
    _write_json(output_dir / "per-query.json", per_query)
    _write_json(
        output_dir / "environment.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "git_commit": _git_commit(),
            "retrieval_or_reranking_executed": False,
        },
    )
    return result


def reproduce(output_dir: Path) -> dict[str, object]:
    """Recompute logical B4 results from the same rankings, excluding timings."""
    reproduction_path = output_dir / "reproduction.json"
    if reproduction_path.exists():
        raise FileExistsError(f"B4 reproduction already exists: {reproduction_path}")
    manifest = _read_json(output_dir / "input-manifest.json")
    profile_data = manifest["profiles"]
    profile_paths = {
        name: Path(data["ranking_path"]) for name, data in profile_data.items()
    }
    environment_paths = {
        name: Path(data["environment_path"]) for name, data in profile_data.items()
    }
    snapshots = {
        name: _load_required_snapshot(path) for name, path in profile_paths.items()
    }
    environments = {name: _read_json(path) for name, path in environment_paths.items()}
    validate_profiles(snapshots, environments)
    profile_summaries: dict[str, Mapping[str, Mapping[str, object]]] = {}
    per_query: dict[str, object] = {}
    for name, snapshot in snapshots.items():
        summary, queries = evaluate_profile(snapshot)
        profile_summaries[name] = summary
        per_query[name] = queries
    repeated = {
        "profiles": profile_summaries,
        "decisions": decide_policies(profile_summaries),
    }
    persisted = _read_json(output_dir / "summary.json")
    persisted_per_query = _read_json(output_dir / "per-query.json")
    matches = _without_timings(repeated) == _without_timings(
        persisted
    ) and _without_timings(per_query) == _without_timings(persisted_per_query)
    if not matches:
        raise RuntimeError("B4 reproduction differs from persisted logical results")
    reproduction = {
        "created_at": datetime.now(UTC).isoformat(),
        "logical_results_match": True,
        "retrieval_or_reranking_executed": False,
    }
    _write_json(reproduction_path, reproduction)
    return reproduction


def _without_timings(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_timings(item)
            for key, item in value.items()
            if key not in {"runtime", "duration_ms"}
        }
    if isinstance(value, list):
        return [_without_timings(item) for item in value]
    return value


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _share(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=False, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "reproduce"))
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
        args.b1_ranking,
        args.b1_environment,
        args.b2_ranking,
        args.b2_environment,
    )
    if any(path is None for path in required):
        parser.error("run requires both ranking and environment paths for B1 and B2")
    result = run(
        {"b1-reference": args.b1_ranking, "b2-lexical": args.b2_ranking},
        {
            "b1-reference": args.b1_environment,
            "b2-lexical": args.b2_environment,
        },
        args.output_dir,
    )
    print(json.dumps(result["decisions"], sort_keys=True))


if __name__ == "__main__":
    main()
