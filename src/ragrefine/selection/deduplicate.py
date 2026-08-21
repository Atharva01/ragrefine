"""Deterministic post-ranking exact and near-duplicate suppression."""

import hashlib
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from ragrefine.models import RefinedCandidate


def normalized_content(text: str) -> str:
    """Normalize text solely for duplicate comparison, never for output."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def content_hash(text: str) -> str:
    """Return the stable SHA-256 identity of normalized candidate content."""
    return hashlib.sha256(normalized_content(text).encode("utf-8")).hexdigest()


def _shingles(text: str, size: int) -> frozenset[tuple[str, ...]]:
    tokens = tuple(re.findall(r"\w+", normalized_content(text)))
    if not tokens:
        return frozenset()
    width = min(size, len(tokens))
    return frozenset(
        tokens[index : index + width] for index in range(len(tokens) - width + 1)
    )


def shingle_jaccard(left: str, right: str, *, size: int = 5) -> float:
    """Measure normalized token-shingle overlap without external dependencies."""
    if size < 1:
        raise ValueError("shingle size must be at least one")
    left_shingles, right_shingles = _shingles(left, size), _shingles(right, size)
    if not left_shingles and not right_shingles:
        return 1.0
    if not left_shingles or not right_shingles:
        return 0.0
    return len(left_shingles & right_shingles) / len(left_shingles | right_shingles)


@dataclass(frozen=True, slots=True)
class DeduplicationConfig:
    """Explicit, deterministic settings for post-ranking duplicate suppression."""

    near_duplicate_threshold: float | None = 0.9
    shingle_size: int = 5

    def __post_init__(self) -> None:
        if self.near_duplicate_threshold is not None and not (
            0.0 <= self.near_duplicate_threshold <= 1.0
        ):
            raise ValueError("near duplicate threshold must be between zero and one")
        if self.shingle_size < 1:
            raise ValueError("shingle size must be at least one")


@dataclass(frozen=True, slots=True)
class SuppressedCandidate:
    """One candidate omitted after ranking, with a stable retained reference."""

    candidate: RefinedCandidate
    reason: str
    retained_candidate_id: str
    normalized_sha256: str
    similarity: float | None = None


@dataclass(frozen=True, slots=True)
class DeduplicationResult:
    """Unchanged retained candidates and traceable duplicate suppressions."""

    candidates: tuple[RefinedCandidate, ...]
    suppressed: tuple[SuppressedCandidate, ...]


class CandidateDeduplicator:
    """Suppress later duplicates while preserving the supplied ranking order.

    Input order is the post-ranking order. The first candidate in each duplicate
    group is retained unchanged; no text, metadata, retrieval evidence, ranking
    evidence, score, or rank value is rewritten.
    """

    def __init__(self, config: DeduplicationConfig | None = None) -> None:
        self._config = config or DeduplicationConfig()

    def deduplicate(
        self, candidates: Sequence[RefinedCandidate]
    ) -> DeduplicationResult:
        """Return retained candidates and suppression records for one ranked pool."""
        retained: list[RefinedCandidate] = []
        suppressed: list[SuppressedCandidate] = []
        exact: dict[str, RefinedCandidate] = {}
        for candidate in candidates:
            digest = content_hash(candidate.candidate.text)
            exact_match = exact.get(digest)
            if exact_match is not None:
                suppressed.append(
                    SuppressedCandidate(
                        candidate=candidate,
                        reason="exact_normalized_content",
                        retained_candidate_id=exact_match.candidate.id,
                        normalized_sha256=digest,
                    )
                )
                continue
            near_match = self._near_match(candidate, retained)
            if near_match is not None:
                retained_candidate, similarity = near_match
                suppressed.append(
                    SuppressedCandidate(
                        candidate=candidate,
                        reason="near_token_shingle_jaccard",
                        retained_candidate_id=retained_candidate.candidate.id,
                        normalized_sha256=digest,
                        similarity=similarity,
                    )
                )
                continue
            exact[digest] = candidate
            retained.append(candidate)
        return DeduplicationResult(tuple(retained), tuple(suppressed))

    def _near_match(
        self,
        candidate: RefinedCandidate,
        retained: Sequence[RefinedCandidate],
    ) -> tuple[RefinedCandidate, float] | None:
        if self._config.near_duplicate_threshold is None:
            return None
        for retained_candidate in retained:
            similarity = shingle_jaccard(
                candidate.candidate.text,
                retained_candidate.candidate.text,
                size=self._config.shingle_size,
            )
            if similarity > self._config.near_duplicate_threshold:
                return retained_candidate, similarity
        return None
