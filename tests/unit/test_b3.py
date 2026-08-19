"""Fast contract tests for frozen B3 fusion experiments."""

from pathlib import Path

import pytest

from benchmarks.beir import b3
from benchmarks.beir.snapshot import snapshot_checksum, write_snapshot


def _snapshot() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "retriever": {"top_n": 2},
        "qrels": {"q": {"a": 1}},
        "queries": [
            {
                "id": "q",
                "text": "query",
                "candidates": [
                    {"id": "b", "text": "other", "rank": 1, "score": 0.2},
                    {"id": "a", "text": "evidence", "rank": 2, "score": 0.1},
                ],
            }
        ],
    }


def test_b3_rejects_missing_profile_artifact(tmp_path: Path) -> None:
    """A profile cannot silently omit one of its configured ranking channels."""
    path = tmp_path / "baseline.json"
    write_snapshot(_snapshot(), path)

    with pytest.raises(ValueError, match="requires 'lexical' ranking"):
        b3.run(path, None, None, None, tmp_path / "out", "original-lexical", hard=True)


def test_b3_requires_checksum_sidecars(tmp_path: Path) -> None:
    """Final B3 evaluation never consumes an unchecked upstream artifact."""
    path = tmp_path / "baseline.json"
    write_snapshot(_snapshot(), path)
    path.with_suffix(path.suffix + ".sha256").unlink()

    with pytest.raises(ValueError, match="missing required checksum sidecar"):
        b3.run(path, None, None, None, tmp_path / "out", "original-lexical", hard=True)


def test_b3_rejects_incompatible_pool(tmp_path: Path) -> None:
    """Fusion refuses a ranking that invents or loses source candidates."""
    baseline = _snapshot()
    baseline_path, lexical_path = tmp_path / "base.json", tmp_path / "lexical.json"
    write_snapshot(baseline, baseline_path)
    incompatible = {
        **baseline,
        "queries": [
            {
                **baseline["queries"][0],
                "candidates": [baseline["queries"][0]["candidates"][0]],
            }
        ],
    }
    write_snapshot(incompatible, lexical_path)

    with pytest.raises(ValueError, match="does not preserve"):
        b3.run(
            baseline_path,
            None,
            lexical_path,
            None,
            tmp_path / "out",
            "original-lexical",
            hard=True,
        )


def test_hard_diagnostics_aggregate_categories() -> None:
    """Hard-set diagnostics retain rank movement and category-level results."""
    snapshot = _snapshot()
    snapshot["queries"][0]["category"] = "wrong_version"
    snapshot["queries"][0]["candidates"][0]["retrieval_rank"] = 1
    snapshot["queries"][0]["candidates"][1]["retrieval_rank"] = 2
    diagnostics = b3._hard_diagnostics(snapshot)

    assert diagnostics["by_category"]["wrong_version"]["top_1_accuracy"] == 0.0
    assert diagnostics["per_query"][0]["rank_movement"] == 0


def test_b3_uses_only_fixed_final_configuration() -> None:
    """The published B3 contract freezes equal weights and k=60."""
    assert b3.FINAL_K == 60
    assert all(channels for channels in b3.PROFILES.values())
    assert snapshot_checksum(_snapshot()) == snapshot_checksum(_snapshot())


def test_reproduce_rechecks_persisted_hard_ranking(tmp_path: Path) -> None:
    """Reproduction evaluates the saved B3 ranking and its saved diagnostics."""
    snapshot = _snapshot()
    snapshot["queries"][0]["category"] = "wrong_version"
    for index, candidate in enumerate(snapshot["queries"][0]["candidates"], start=1):
        candidate["retrieval_rank"] = index
    baseline_path, lexical_path = tmp_path / "base.json", tmp_path / "lexical.json"
    write_snapshot(snapshot, baseline_path)
    write_snapshot(snapshot, lexical_path)

    output_dir = tmp_path / "out"
    first = b3.run(
        baseline_path,
        None,
        lexical_path,
        None,
        output_dir,
        "original-lexical",
        hard=True,
    )

    assert b3.reproduce(output_dir / "b3-ranking.json") == first
