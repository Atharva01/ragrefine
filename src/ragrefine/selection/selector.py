"""Deterministic rank-preserving context selection after ranking."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ragrefine.models import RefinedCandidate
from ragrefine.selection.deduplicate import SuppressedCandidate
from ragrefine.selection.tokens import TokenCounter


@dataclass(frozen=True, slots=True)
class SelectionRecord:
    """One selected or omitted candidate with a machine-readable explanation."""

    candidate_id: str
    status: str
    reason: str
    token_count: int | None = None
    retained_candidate_id: str | None = None


@dataclass(frozen=True, slots=True)
class ContextSelectionResult:
    """Selected unchanged candidates and a complete post-ranking selection trace."""

    candidates: tuple[RefinedCandidate, ...]
    records: tuple[SelectionRecord, ...]


class ContextSelector(Protocol):
    """Select ranked candidates under optional Top-K and token-budget limits."""

    def select(
        self,
        candidates: Sequence[RefinedCandidate],
        *,
        top_k: int | None = None,
        max_tokens: int | None = None,
        suppressed: Sequence[SuppressedCandidate] = (),
    ) -> ContextSelectionResult:
        """Return unchanged selected candidates and trace records."""


class RankPreservingContextSelector:
    """Select unmodified ranked candidates without changing ranking semantics.

    Candidates are considered in supplied post-ranking order. An oversized or
    budget-excluded candidate does not consume a Top-K slot, allowing a later
    candidate to be retained while preserving the relative order of selections.
    """

    def __init__(self, token_counter: TokenCounter | None = None) -> None:
        self._token_counter = token_counter

    def select(
        self,
        candidates: Sequence[RefinedCandidate],
        *,
        top_k: int | None = None,
        max_tokens: int | None = None,
        suppressed: Sequence[SuppressedCandidate] = (),
    ) -> ContextSelectionResult:
        """Select candidates while recording duplicate, budget, and Top-K outcomes."""
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be non-negative")
        if max_tokens is not None and max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if max_tokens is not None and self._token_counter is None:
            raise ValueError("max_tokens requires a TokenCounter")

        records = [
            SelectionRecord(
                candidate_id=item.candidate.candidate.id,
                status="duplicate_suppressed",
                reason=item.reason,
                retained_candidate_id=item.retained_candidate_id,
            )
            for item in suppressed
        ]
        selected: list[RefinedCandidate] = []
        used_tokens = 0
        for candidate in candidates:
            if top_k is not None and len(selected) >= top_k:
                records.append(
                    SelectionRecord(
                        candidate_id=candidate.candidate.id,
                        status="excluded",
                        reason="top_k_limit",
                    )
                )
                continue
            token_count: int | None = None
            if max_tokens is not None:
                token_count = self._count(candidate.candidate.text)
                if used_tokens + token_count > max_tokens:
                    records.append(
                        SelectionRecord(
                            candidate_id=candidate.candidate.id,
                            status="budget_excluded",
                            reason="max_tokens_exceeded",
                            token_count=token_count,
                        )
                    )
                    continue
                used_tokens += token_count
            selected.append(candidate)
            records.append(
                SelectionRecord(
                    candidate_id=candidate.candidate.id,
                    status="selected",
                    reason="within_constraints",
                    token_count=token_count,
                )
            )
        return ContextSelectionResult(tuple(selected), tuple(records))

    def _count(self, text: str) -> int:
        if self._token_counter is None:  # Narrowed by select() validation.
            raise RuntimeError("token counter is unavailable")
        count = self._token_counter.count(text)
        if count < 0:
            raise ValueError("TokenCounter must not return a negative count")
        return count
