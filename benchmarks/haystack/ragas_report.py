"""A/B analysis and decision for the paired Haystack Ragas evaluation.

Consumes the persisted paired harness artifact (``ragas-results.json``) and
the Ragas scores artifact (``ragas-scores.json``) and produces an
evidence-backed B0-versus-refined comparison:

* aggregate baseline/refined scores per metric (from the scores artifact);
* paired per-query deltas with wins/losses/ties;
* a deterministic paired bootstrap 95% confidence interval on the mean delta;
* context size changes (prompt word counts per arm);
* representative wins and regressions per metric;
* an explicit decision per metric and overall: ``supported_improvement``,
  ``regression``, or ``inconclusive`` — never a broader claim.

The report embeds the ``verify`` output as the reproduction record. This is
analysis only; it does not regenerate retrieval, context, answers, or scores.
"""

import argparse
import hashlib
import json
import platform
import random
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from benchmarks.haystack.ragas_score import SCHEMA_VERSION as SCORES_SCHEMA_VERSION
from benchmarks.haystack.ragas_score import verify

REPORT_SCHEMA_VERSION = "1.0"
INPUT_SCHEMA_VERSION = "1.1"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 97
TIE_TOLERANCE = 1e-9
REPRESENTATIVE_COUNT = 3


class Verifier(Protocol):
    """Verification seam matching ``ragas_score.verify``."""

    def __call__(
        self, *, artifact_path: Path, scores_path: Path | None
    ) -> dict[str, object]: ...


def run(
    output_dir: Path,
    *,
    artifact_path: Path,
    scores_path: Path,
    verifier: Verifier = verify,
) -> dict[str, object]:
    """Analyze persisted paired artifacts and persist the comparison report."""
    if output_dir.exists():
        raise FileExistsError(f"Ragas report output already exists: {output_dir}")
    artifact, scores = _load_artifacts(artifact_path, scores_path)
    verification = verifier(artifact_path=artifact_path, scores_path=scores_path)

    metrics = scores["configuration"]["metrics"]
    per_metric: dict[str, object] = {}
    for metric in metrics:
        per_metric[metric] = _analyze_metric(metric, scores)

    context = _context_summary(artifact)
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "input_artifact": {
            "path": str(artifact_path),
            "sha256": _file_sha256(artifact_path),
        },
        "scores_artifact": {
            "path": str(scores_path),
            "sha256": _file_sha256(scores_path),
        },
        "configuration": {
            "metrics": metrics,
            "omitted_metrics": scores["configuration"].get("omitted_metrics", {}),
            "paired_queries": scores["configuration"].get("paired_rows_scored", 0),
            "refinement_profile": artifact.get("profile"),
        },
        "context": context,
        "per_metric": per_metric,
        "decision": _overall_decision(
            [item["decision"]["outcome"] for item in per_metric.values()]
        ),
        "statistics": {
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "tie_tolerance": TIE_TOLERANCE,
        },
        "verification": verification,
        "environment": _environment(scores),
        "interpretation": (
            "Ragas A/B of the frozen human-reviewed set (B2-L lexical profile); "
            "per-query and aggregate scores are LLM-judged and non-deterministic. "
            "An outcome is classified supported_improvement or regression only "
            "when the paired delta, its bootstrap confidence interval, and the "
            "win/loss counts agree; no broader claim is made."
        ),
    }
    output_dir.mkdir(parents=True)
    output_path = output_dir / "ragas-report.json"
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "ragas-report.json.sha256").write_text(
        hashlib.sha256(output_path.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    (output_dir / "ragas-report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _load_artifacts(
    artifact_path: Path, scores_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    if not artifact_path.is_file() or not scores_path.is_file():
        raise ValueError("both input and scores artifacts must exist")
    artifact: object = json.loads(artifact_path.read_text(encoding="utf-8"))
    scores: object = json.loads(scores_path.read_text(encoding="utf-8"))
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema_version") != INPUT_SCHEMA_VERSION
    ):
        raise ValueError(
            f"unsupported input artifact schema: {artifact.get('schema_version')!r}"
        )
    if (
        not isinstance(scores, dict)
        or scores.get("schema_version") != SCORES_SCHEMA_VERSION
    ):
        raise ValueError(
            f"unsupported scores artifact schema: {scores.get('schema_version')!r}"
        )
    recorded = scores.get("input_artifact")
    if not isinstance(recorded, dict) or recorded.get("sha256") != _file_sha256(
        artifact_path
    ):
        raise ValueError("scores artifact was not produced from this input artifact")
    return artifact, scores


def _analyze_metric(metric: str, scores: dict[str, object]) -> dict[str, object]:
    baseline_block = scores["scores"]["baseline"][metric]
    refined_block = scores["scores"]["refined"][metric]
    baseline_scores = dict(baseline_block["per_sample"])
    refined_scores = dict(refined_block["per_sample"])
    paired_ids = sorted(set(baseline_scores) & set(refined_scores))

    deltas: dict[str, float] = {}
    per_query: dict[str, object] = {}
    for qa_id in paired_ids:
        baseline = baseline_scores[qa_id]
        refined = refined_scores[qa_id]
        delta = refined - baseline
        deltas[qa_id] = delta
        per_query[qa_id] = {
            "baseline": baseline,
            "refined": refined,
            "delta": delta,
        }
    outcomes = _outcomes(deltas)
    ordered = sorted(deltas, key=lambda qa_id: deltas[qa_id])
    values = list(deltas.values())
    mean_delta = sum(values) / len(values) if values else None
    ci = _bootstrap_ci(values) if values else None
    decision = _metric_decision(mean_delta, ci, outcomes, len(paired_ids), metric)
    return {
        "aggregate": {
            "baseline": baseline_block.get("aggregate"),
            "refined": refined_block.get("aggregate"),
        },
        "scored_queries": len(paired_ids),
        "mean_paired_delta": mean_delta,
        "paired_delta_ci95": ci,
        "outcomes": outcomes,
        "per_query": per_query,
        "representative_wins": _representative(ordered, deltas, reverse=True),
        "representative_regressions": _representative(ordered, deltas),
        "decision": decision,
    }


def _outcomes(deltas: Mapping[str, float]) -> dict[str, int]:
    wins = losses = ties = 0
    for delta in deltas.values():
        if delta > TIE_TOLERANCE:
            wins += 1
        elif delta < -TIE_TOLERANCE:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties}


def _bootstrap_ci(
    deltas: Sequence[float], samples: int = BOOTSTRAP_SAMPLES
) -> list[float] | None:
    """Deterministic percentile bootstrap on the mean paired delta."""
    if not deltas:
        return None
    generator = random.Random(BOOTSTRAP_SEED)
    size = len(deltas)
    means = [
        sum(deltas[index] for index in (generator.randrange(size) for _ in range(size)))
        / size
        for _ in range(samples)
    ]
    means.sort()
    lower = means[int(0.025 * (samples - 1))]
    upper = means[int(0.975 * (samples - 1))]
    return [round(lower, 6), round(upper, 6)]


def _metric_decision(
    mean_delta: float | None,
    ci: list[float] | None,
    outcomes: Mapping[str, int],
    scored_queries: int,
    metric: str,
) -> dict[str, object]:
    if scored_queries == 0 or mean_delta is None or ci is None:
        return {
            "outcome": "inconclusive",
            "rationale": f"no paired scores for {metric!r}",
        }
    wins = outcomes["wins"]
    losses = outcomes["losses"]
    if mean_delta > 0 and ci[0] > 0 and wins > losses:
        return {
            "outcome": "supported_improvement",
            "rationale": (
                f"positive mean paired delta ({mean_delta:.4f}) with a 95% "
                f"bootstrap CI [{ci[0]:.4f}, {ci[1]:.4f}] excluding zero and "
                f"more wins than losses ({wins} vs {losses})"
            ),
        }
    if mean_delta < 0 and ci[1] < 0 and losses > wins:
        return {
            "outcome": "regression",
            "rationale": (
                f"negative mean paired delta ({mean_delta:.4f}) with a 95% "
                f"bootstrap CI [{ci[0]:.4f}, {ci[1]:.4f}] excluding zero and "
                f"more losses than wins ({losses} vs {wins})"
            ),
        }
    return {
        "outcome": "inconclusive",
        "rationale": (
            f"mean paired delta {mean_delta:.4f} with CI "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] does not give a consistent signal "
            f"({wins} wins, {losses} losses, {outcomes['ties']} ties)"
        ),
    }


def _overall_decision(outcomes: Sequence[str]) -> dict[str, object]:
    improvements = outcomes.count("supported_improvement")
    regressions = outcomes.count("regression")
    if improvements and not regressions:
        return {
            "outcome": "supported_improvement",
            "rationale": (
                f"{improvements} metric(s) show supported improvement and none "
                "show regression"
            ),
        }
    if regressions and not improvements:
        return {
            "outcome": "regression",
            "rationale": (
                f"{regressions} metric(s) show regression and none show "
                "supported improvement"
            ),
        }
    return {
        "outcome": "inconclusive",
        "rationale": (
            f"mixed or absent signals ({improvements} supported improvements, "
            f"{regressions} regressions)"
        ),
    }


def _representative(
    ordered: Sequence[str],
    deltas: Mapping[str, float],
    *,
    reverse: bool = False,
) -> list[dict[str, object]]:
    if reverse:
        candidates = [
            qa_id for qa_id in reversed(ordered) if deltas[qa_id] > TIE_TOLERANCE
        ][:REPRESENTATIVE_COUNT]
    else:
        candidates = [qa_id for qa_id in ordered if deltas[qa_id] < -TIE_TOLERANCE][
            :REPRESENTATIVE_COUNT
        ]
    return [{"qa_id": qa_id, "delta": deltas[qa_id]} for qa_id in candidates]


def _context_summary(artifact: Mapping[str, object]) -> dict[str, object]:
    per_query: dict[str, object] = {}
    baseline_words = refined_words = 0
    for row in artifact["per_query"]:
        baseline_prompt = row.get("baseline_prompt")
        refined_prompt = row.get("refined_prompt")
        if not isinstance(baseline_prompt, str) or not isinstance(refined_prompt, str):
            continue
        baseline_count = len(baseline_prompt.split())
        refined_count = len(refined_prompt.split())
        baseline_words += baseline_count
        refined_words += refined_count
        per_query[str(row["qa_id"])] = {
            "baseline_words": baseline_count,
            "refined_words": refined_count,
            "delta_words": refined_count - baseline_count,
        }
    delta_words = refined_words - baseline_words
    return {
        "per_query": per_query,
        "aggregate": {
            "baseline_words": baseline_words,
            "refined_words": refined_words,
            "delta_words": delta_words,
        },
    }


def _markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Haystack Ragas A/B report",
        "",
        f"- Input artifact: `{report['input_artifact']['path']}`",
        f"- Scores artifact: `{report['scores_artifact']['path']}`",
        f"- Profile: {report['configuration']['refinement_profile']}",
        f"- Paired queries scored: {report['configuration']['paired_queries']}",
        f"- Metrics: {', '.join(report['configuration']['metrics'])}",
        "",
        "## Per-metric comparison",
        "",
        "| Metric | Baseline | Refined | Mean Δ | CI95 (Δ) | W/L/T | Decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for metric, block in sorted(report["per_metric"].items()):
        outcomes = block["outcomes"]
        lines.append(
            f"| {metric} | {_fmt(block['aggregate']['baseline'])} | "
            f"{_fmt(block['aggregate']['refined'])} | "
            f"{_fmt(block['mean_paired_delta'])} | "
            f"{_fmt_ci(block['paired_delta_ci95'])} | "
            f"{outcomes['wins']}/{outcomes['losses']}/{outcomes['ties']} | "
            f"{block['decision']['outcome']} |"
        )
    lines.extend(
        [
            "",
            f"**Overall decision: {report['decision']['outcome']}** — "
            f"{report['decision']['rationale']}",
            "",
            "## Context size",
            "",
            f"- Baseline words: {report['context']['aggregate']['baseline_words']}",
            f"- Refined words: {report['context']['aggregate']['refined_words']}",
            f"- Delta words: {report['context']['aggregate']['delta_words']}",
            "",
            "## Representative regressions (refined below baseline)",
        ]
    )
    for metric, block in sorted(report["per_metric"].items()):
        regressions = block["representative_regressions"]
        if regressions:
            details = ", ".join(
                f"{item['qa_id']} (Δ {item['delta']:+.4f})" for item in regressions
            )
            lines.append(f"- {metric}: {details}")
    lines.extend(
        [
            "",
            "## Verification",
            "",
            f"- OK: {report['verification']['ok']}",
        ]
    )
    if report["verification"]["failures"]:
        lines.append(f"- Failures: {json.dumps(report['verification']['failures'])}")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "—"


def _fmt_ci(ci: object) -> str:
    if not isinstance(ci, list) or len(ci) != 2:
        return "—"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def _environment(scores: Mapping[str, object]) -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "git_commit": _git_commit(),
        "model": scores.get("model"),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "tie_tolerance": TIE_TOLERANCE,
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = run(
        args.output_dir,
        artifact_path=args.input,
        scores_path=args.scores,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
