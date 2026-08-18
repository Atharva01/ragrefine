"""Deterministic frozen-candidate snapshot serialization."""

import hashlib
import json
from pathlib import Path
from typing import Any  # noqa: I001

SNAPSHOT_SCHEMA_VERSION = "1.0"


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a benchmark artifact in a stable, hashable representation."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def snapshot_checksum(snapshot: object) -> str:
    """Return the SHA-256 checksum for a complete frozen snapshot."""
    return hashlib.sha256(canonical_json_bytes(snapshot)).hexdigest()


def write_snapshot(snapshot: dict[str, Any], path: Path) -> str:
    """Persist a snapshot and its checksum sidecar without changing its order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(snapshot)
    path.write_bytes(encoded)
    checksum = hashlib.sha256(encoded).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        checksum + "\n", encoding="utf-8"
    )
    return checksum


def load_snapshot(path: Path) -> dict[str, Any]:
    """Load and validate a frozen snapshot against its checksum sidecar when present."""
    encoded = path.read_bytes()
    expected_path = path.with_suffix(path.suffix + ".sha256")
    if (
        expected_path.exists()
        and expected_path.read_text(encoding="utf-8").strip()
        != hashlib.sha256(encoded).hexdigest()
    ):
        msg = f"snapshot checksum does not match: {path}"
        raise ValueError(msg)
    snapshot = json.loads(encoded)
    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        msg = f"unsupported snapshot schema: {snapshot.get('schema_version')}"
        raise ValueError(msg)
    return snapshot
