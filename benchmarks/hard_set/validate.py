"""Validation and deterministic checksums for curated hard-negative data."""

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

HARD_SET_SCHEMA_VERSION = "1.0"
REQUIRED_CATEGORIES = frozenset(
    {"wrong_version", "wrong_identifier", "wrong_date", "wrong_numeric_value"}
)
_REQUIRED_CASE_FIELDS = frozenset(
    {
        "id",
        "query_id",
        "query_text",
        "candidate_id",
        "candidate_text",
        "relevance",
        "category",
        "expected_constraints",
    }
)


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a diagnostic artifact deterministically for checksum generation."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def checksum(value: object) -> str:
    """Return the SHA-256 checksum of canonical hard-set content."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_and_validate(path: Path) -> dict[str, Any]:
    """Load a frozen dataset, verify its sidecar, and validate its schema."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    expected = (
        path.with_suffix(path.suffix + ".sha256").read_text(encoding="utf-8").strip()
    )
    actual = checksum(data)
    if actual != expected:
        raise ValueError(f"hard-set checksum does not match: {path}")
    validate(data)
    return data


def validate(data: Mapping[str, object]) -> None:
    """Reject malformed, ambiguous, or insufficiently diagnostic artifacts."""
    if data.get("schema_version") != HARD_SET_SCHEMA_VERSION:
        raise ValueError("unsupported hard-set schema version")
    cases = data.get("cases")
    if not isinstance(cases, list) or not 40 <= len(cases) <= 60:
        raise ValueError("hard-set must contain 40 to 60 cases")

    seen_case_ids: set[str] = set()
    seen_candidate_ids: set[str] = set()
    categories: set[str] = set()
    groups: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    previous_case_id = ""
    for case in cases:
        if not isinstance(case, Mapping) or set(case) != _REQUIRED_CASE_FIELDS:
            raise ValueError("each hard-set case must have the required fields only")
        if not all(
            isinstance(case[field], str) and case[field]
            for field in _REQUIRED_CASE_FIELDS - {"relevance", "expected_constraints"}
        ):
            raise ValueError(
                "hard-set text, ID, and category fields must be non-empty strings"
            )
        case_id = case["id"]
        candidate_id = case["candidate_id"]
        if case_id in seen_case_ids or candidate_id in seen_candidate_ids:
            raise ValueError("hard-set case and candidate IDs must be unique")
        if case_id <= previous_case_id:
            raise ValueError("hard-set cases must be ordered by ascending case ID")
        seen_case_ids.add(case_id)
        seen_candidate_ids.add(candidate_id)
        previous_case_id = case_id
        if case["relevance"] not in {0, 1}:
            raise ValueError("hard-set relevance judgments must be 0 or 1")
        constraints = case["expected_constraints"]
        if not isinstance(constraints, Mapping) or not constraints:
            raise ValueError("hard-set cases require explicit expected constraints")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in constraints.items()
        ):
            raise ValueError("hard-set constraints must map strings to strings")
        categories.add(case["category"])
        groups[case["query_id"]].append(case)

    if not REQUIRED_CATEGORIES.issubset(categories):
        raise ValueError("hard-set is missing a required mismatch category")
    for query_id, group in groups.items():
        reference = group[0]
        if any(
            case["query_text"] != reference["query_text"]
            or case["expected_constraints"] != reference["expected_constraints"]
            or case["category"] != reference["category"]
            for case in group
        ):
            raise ValueError(f"hard-set query group {query_id!r} is inconsistent")
        if sum(case["relevance"] == 1 for case in group) != 1:
            raise ValueError(f"hard-set query group {query_id!r} needs one positive")
        if sum(case["relevance"] == 0 for case in group) < 2:
            raise ValueError(f"hard-set query group {query_id!r} needs hard negatives")
