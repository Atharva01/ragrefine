"""Core retriever-agnostic candidate domain models."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Candidate:
    """One immutable evidence unit supplied by an external retriever.

    The model preserves the retriever's identifier, text, metadata, and optional
    retrieval annotations without depending on any retrieval framework.
    """

    id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    retrieval_score: float | None = None
    retrieval_rank: int | None = None


@dataclass(frozen=True, slots=True)
class CandidateSet:
    """An immutable, ordered candidate collection from one named retriever."""

    name: str
    candidates: tuple[Candidate, ...]
