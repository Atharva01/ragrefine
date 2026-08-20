"""Frozen-artifact B4 selection-ablation tests."""

import json
from pathlib import Path

import pytest

from benchmarks.beir import b4
from benchmarks.beir.snapshot import snapshot_checksum, write_snapshot


def _snapshot() -> dict[str, object]:
    shared = " ".join(f"token{index}" for index in range(30))
    return {
        "schema_version": "1.0",
        "dataset": "fixture",
        "qrels": {"q": {"a": 1, "near": 1, "late": 1}},
        "queries": [
            {
                "id": "q",
                "text": "query",
                "candidates": [
                    {"id": "a", "text": f"{shared} alpha", "rank": 1, "score": 1.0},
                    {"id": "exact", "text": f"{shared} alpha", "rank": 2, "score": 0.9},
                    {"id": "near", "text": f"{shared} beta", "rank": 3, "score": 0.8},
                    {"id": "c", "text": "unrelated gamma", "rank": 4, "score": 0.7},
                    {"id": "d", "text": "unrelated delta", "rank": 5, "score": 0.6},
                    {"id": "late", "text": "late evidence", "rank": 6, "score": 0.5},
                ],
            }
        ],
    }


def _environment(snapshot: object) -> dict[str, object]:
    return {
        "b0_snapshot_checksum": "frozen-b0",
        "ranking_checksum": snapshot_checksum(snapshot),
    }


def test_b4_policies_are_predeclared_and_independent() -> None:
    """The experiment contains the five required ranking/selection ablations."""
    assert [policy.name for policy in b4.POLICIES] == ["S0", "S1", "S2", "S3", "S4"]
    assert b4.POLICIES[1].deduplication == "exact"
    assert b4.POLICIES[2].max_tokens is None
    assert b4.POLICIES[3].deduplication is None
    assert b4.POLICIES[4].max_tokens == b4.TOKEN_BUDGET


def test_b4_records_selected_ids_exclusions_and_duplicate_effects() -> None:
    """Per-query output preserves all decisions needed to audit each policy."""
    summary, per_query = b4.evaluate_profile(_snapshot())

    assert per_query["q"]["S0"]["selected_ids"] == ["a", "exact", "near", "c", "d"]
    assert per_query["q"]["S1"]["selected_ids"] == ["a", "near", "c", "d", "late"]
    assert per_query["q"]["S2"]["selected_ids"] == ["a", "c", "d", "late"]
    assert summary["S1"]["duplicates"]["exact_suppressed"] == 1
    assert summary["S1"]["duplicates"]["near_suppressed"] == 0
    assert summary["S2"]["duplicates"]["near_suppressed"] == 1
    assert any(
        exclusion["reason"] == "exact_normalized_content"
        for exclusion in per_query["q"]["S1"]["exclusions"]
    )


def test_b4_token_budget_is_applied_without_deduplication() -> None:
    """S3 semantics can constrain tokens independently of duplicate handling."""
    candidates = b4._ranked_candidates(_snapshot()["queries"][0])
    policy = b4.SelectionPolicy("budget", "test", max_tokens=2)

    selected, records, exact, near = b4._apply_policy(candidates, policy)

    assert [candidate.candidate.id for candidate in selected] == ["c"]
    assert exact == near == 0
    assert any(record.status == "budget_excluded" for record in records)


def test_b4_rejects_mismatched_frozen_candidate_pools() -> None:
    """Retained profiles must contain exactly the same frozen input population."""
    first = _snapshot()
    second = _snapshot()
    second["queries"][0]["candidates"].pop()

    with pytest.raises(ValueError, match="does not preserve"):
        b4.validate_profiles(
            {"b1-reference": first, "b2-lexical": second},
            {
                "b1-reference": _environment(first),
                "b2-lexical": _environment(second),
            },
        )


def test_b4_run_and_reproduce_use_persisted_rankings_only(tmp_path: Path) -> None:
    """A saved experiment reproduces logical results without retrieval or reranking."""
    snapshot = _snapshot()
    b1_path = tmp_path / "b1.json"
    b2_path = tmp_path / "b2.json"
    write_snapshot(snapshot, b1_path)
    write_snapshot(snapshot, b2_path)
    b1_environment = tmp_path / "b1-environment.json"
    b2_environment = tmp_path / "b2-environment.json"
    b1_environment.write_text(json.dumps(_environment(snapshot)), encoding="utf-8")
    b2_environment.write_text(json.dumps(_environment(snapshot)), encoding="utf-8")

    output_dir = tmp_path / "result"
    result = b4.run(
        {"b1-reference": b1_path, "b2-lexical": b2_path},
        {"b1-reference": b1_environment, "b2-lexical": b2_environment},
        output_dir,
    )
    reproduction = b4.reproduce(output_dir)

    assert result["profiles"].keys() == {"b1-reference", "b2-lexical"}
    assert result["decisions"].keys() == {"S0", "S1", "S2", "S3", "S4"}
    assert reproduction["logical_results_match"] is True
    assert reproduction["retrieval_or_reranking_executed"] is False
    assert (output_dir / "per-query.json").exists()
    assert (output_dir / "input-manifest.json").exists()
