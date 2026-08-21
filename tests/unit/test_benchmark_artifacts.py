"""Tests for external immutable benchmark-artifact verification."""

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.beir import artifacts


def _manifest(path: Path, checksum: str) -> Path:
    manifest = path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "artifacts": [{"path": "one.json", "sha256": checksum}],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_verify_accepts_exact_declared_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "one.json"
    artifact.write_bytes(b"frozen")
    manifest = _manifest(tmp_path, hashlib.sha256(b"frozen").hexdigest())

    assert artifacts.verify(manifest, tmp_path)["verified"] == ["one.json"]


def test_verify_names_missing_or_mismatched_artifacts(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, "0" * 64)

    with pytest.raises(FileNotFoundError, match="missing declared benchmark artifact"):
        artifacts.verify(manifest, tmp_path)

    (tmp_path / "one.json").write_bytes(b"different")
    with pytest.raises(ValueError, match="checksum mismatch"):
        artifacts.verify(manifest, tmp_path)
