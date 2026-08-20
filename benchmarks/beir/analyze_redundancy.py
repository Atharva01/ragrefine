"""Persist a deterministic redundancy and context-size audit for snapshots."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.beir.redundancy import (
    NEAR_DUPLICATE_THRESHOLDS,
    SHINGLE_SIZE,
    WORD_TOKENIZATION,
    analyze_snapshot,
)
from benchmarks.beir.snapshot import load_snapshot, snapshot_checksum


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def analyze_paths(paths: list[Path]) -> dict[str, object]:
    """Analyze snapshots and retain their checksums as audit provenance."""
    datasets: dict[str, object] = {}
    for path in paths:
        snapshot = load_snapshot(path)
        analysis = analyze_snapshot(snapshot)
        dataset = str(analysis["dataset"])
        if dataset in datasets:
            raise ValueError(f"duplicate dataset in redundancy audit: {dataset}")
        datasets[dataset] = {
            "snapshot": str(path),
            "snapshot_checksum": snapshot_checksum(snapshot),
            "analysis": analysis,
        }
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "measurement": {
            "word_tokenization": WORD_TOKENIZATION,
            "shingle_size": SHINGLE_SIZE,
            "near_duplicate_thresholds": list(NEAR_DUPLICATE_THRESHOLDS),
            "relevance_labels_used": False,
        },
        "datasets": datasets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot", action="append", required=True, type=Path, help="Frozen snapshot"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing audit: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, analyze_paths(args.snapshot))


if __name__ == "__main__":
    main()
