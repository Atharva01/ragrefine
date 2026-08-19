"""Tests for the frozen structured hard-negative diagnostic artifact."""

import json
from pathlib import Path

import pytest

from benchmarks.hard_set.validate import (
    REQUIRED_CATEGORIES,
    checksum,
    load_and_validate,
)

_DATASET = Path("benchmarks/hard_set/structured-hard-negatives-v1.json")


def test_frozen_hard_set_is_valid_and_covers_required_categories() -> None:
    """The committed artifact has a valid sidecar and all diagnostic categories."""
    dataset = load_and_validate(_DATASET)

    assert len(dataset["cases"]) == 48
    assert REQUIRED_CATEGORIES.issubset({case["category"] for case in dataset["cases"]})


def test_hard_set_checksum_is_deterministic() -> None:
    """Canonical serialization produces a stable checksum for the frozen data."""
    data = json.loads(_DATASET.read_text(encoding="utf-8"))
    assert checksum(data) == checksum(data)


def test_hard_set_rejects_ambiguous_query_groups(tmp_path: Path) -> None:
    """A group with multiple positives fails validation after checksum verification."""
    data = json.loads(_DATASET.read_text(encoding="utf-8"))
    data["cases"][1]["relevance"] = 1
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    path.with_suffix(".json.sha256").write_text(checksum(data) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="needs one positive"):
        load_and_validate(path)
