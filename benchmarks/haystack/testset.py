"""Frozen, human-reviewed Ragas test set for the Haystack integration.

The test set couples a small retrieval corpus with hand-reviewed QA pairs. Each
pair records a question, a reference answer grounded in one or more reference
documents, and the IDs of those reference documents. It is the fixed,
reproducible input for the Ragas evaluation harness and is not a claim about
retrieval or generation quality on its own.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RAGAS_TEST_SET_SCHEMA_VERSION = "1.0"
DEFAULT_TESTSET_PATH = Path(__file__).parent / "testset" / "ragas-testset-v1.json"


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    """One retrievable document in the frozen Ragas test-set corpus."""

    id: str
    content: str
    meta: dict[str, str]


@dataclass(frozen=True, slots=True)
class QAPair:
    """One human-reviewed question with a grounded reference answer."""

    id: str
    question: str
    reference_answer: str
    reference_context_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RagasTestSet:
    """Validated, checksum-verified corpus plus QA pairs."""

    corpus: tuple[CorpusDocument, ...]
    qa_pairs: tuple[QAPair, ...]
    sha256: str


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a test-set artifact deterministically for checksum generation."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def checksum(value: object) -> str:
    """Return the SHA-256 checksum of canonical test-set content."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_testset(path: Path = DEFAULT_TESTSET_PATH) -> RagasTestSet:
    """Load, checksum-verify, and validate the frozen Ragas test set."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    expected = (
        path.with_suffix(path.suffix + ".sha256").read_text(encoding="utf-8").strip()
    )
    actual = checksum(data)
    if actual != expected:
        raise ValueError(f"ragas test-set checksum does not match: {path}")
    validate(data)
    corpus = tuple(
        CorpusDocument(
            id=document["id"],
            content=document["content"],
            meta=dict(document["meta"]),
        )
        for document in data["corpus"]
    )
    qa_pairs = tuple(
        QAPair(
            id=pair["id"],
            question=pair["question"],
            reference_answer=pair["reference_answer"],
            reference_context_ids=tuple(pair["reference_context_ids"]),
        )
        for pair in data["qa_pairs"]
    )
    return RagasTestSet(corpus=corpus, qa_pairs=qa_pairs, sha256=actual)


def validate(data: dict[str, Any]) -> None:
    """Reject malformed or internally inconsistent test-set artifacts."""
    if data.get("schema_version") != RAGAS_TEST_SET_SCHEMA_VERSION:
        raise ValueError("unsupported ragas test-set schema version")

    corpus = data.get("corpus")
    if not isinstance(corpus, list) or not corpus:
        raise ValueError("ragas test-set must contain a non-empty corpus")
    corpus_ids: set[str] = set()
    for document in corpus:
        if not isinstance(document, dict):
            raise ValueError("each corpus document must be an object")
        if set(document) != {"id", "content", "meta"}:
            raise ValueError("corpus documents require exactly id, content, and meta")
        if not isinstance(document["id"], str) or not document["id"]:
            raise ValueError("corpus document IDs must be non-empty strings")
        if not isinstance(document["content"], str) or not document["content"]:
            raise ValueError("corpus document content must be non-empty strings")
        if document["id"] in corpus_ids:
            raise ValueError(f"duplicate corpus document ID: {document['id']}")
        if not isinstance(document["meta"], dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in document["meta"].items()
        ):
            raise ValueError("corpus document meta must map strings to strings")
        corpus_ids.add(document["id"])

    qa_pairs = data.get("qa_pairs")
    if not isinstance(qa_pairs, list) or not qa_pairs:
        raise ValueError("ragas test-set must contain a non-empty qa_pairs list")
    pair_ids: set[str] = set()
    for pair in qa_pairs:
        if not isinstance(pair, dict):
            raise ValueError("each QA pair must be an object")
        required = {"id", "question", "reference_answer", "reference_context_ids"}
        if set(pair) != required:
            raise ValueError(
                "QA pairs require id, question, reference_answer, "
                "and reference_context_ids only"
            )
        if not isinstance(pair["id"], str) or not pair["id"]:
            raise ValueError("QA pair IDs must be non-empty strings")
        if pair["id"] in pair_ids:
            raise ValueError(f"duplicate QA pair ID: {pair['id']}")
        pair_ids.add(pair["id"])
        for field in ("question", "reference_answer"):
            if not isinstance(pair[field], str) or not pair[field]:
                raise ValueError(f"QA pair {field} must be a non-empty string")
        reference_ids = pair["reference_context_ids"]
        if (
            not isinstance(reference_ids, list)
            or not reference_ids
            or not all(isinstance(item, str) and item for item in reference_ids)
        ):
            raise ValueError(
                "QA pair reference_context_ids must be a non-empty list of strings"
            )
        for reference_id in reference_ids:
            if reference_id not in corpus_ids:
                raise ValueError(
                    f"QA pair {pair['id']!r} references unknown corpus ID "
                    f"{reference_id!r}"
                )
