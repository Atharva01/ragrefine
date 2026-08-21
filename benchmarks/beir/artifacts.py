"""Verify an external immutable benchmark-artifact bundle before analysis."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_manifest(path: Path) -> dict[str, Any]:
    """Load the committed retained-artifact manifest."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "1.0":
        raise ValueError(f"unsupported artifact manifest: {path}")
    if not isinstance(value.get("artifacts"), list):
        raise ValueError(f"artifact manifest lacks an artifact list: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(manifest_path: Path, artifact_root: Path) -> dict[str, object]:
    """Fail clearly unless every declared immutable artifact matches its checksum."""
    manifest = _read_manifest(manifest_path)
    verified: list[str] = []
    for entry in manifest["artifacts"]:
        if not isinstance(entry, dict):
            raise ValueError("artifact manifest entries must be objects")
        relative_path = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected, str):
            raise ValueError("artifact manifest entry lacks path or SHA-256")
        artifact = artifact_root / relative_path
        if not artifact.is_file():
            raise FileNotFoundError(
                f"missing declared benchmark artifact: {artifact}. "
                "See benchmarks/artifacts/README.md for bundle layout."
            )
        actual = _sha256(artifact)
        if actual != expected:
            raise ValueError(
                f"benchmark artifact checksum mismatch: {artifact}; "
                f"expected {expected}, got {actual}"
            )
        verified.append(relative_path)
    return {
        "manifest": str(manifest_path),
        "artifact_root": str(artifact_root),
        "verified": verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verify", choices=("verify",))
    parser.add_argument(
        "--manifest", type=Path, default=Path("benchmarks/artifacts/retained-v1.json")
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("benchmarks"))
    args = parser.parse_args()
    print(json.dumps(verify(args.manifest, args.artifact_root), sort_keys=True))


if __name__ == "__main__":
    main()
