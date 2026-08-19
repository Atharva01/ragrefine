"""Native deterministic weighted Reciprocal Rank Fusion."""

from collections.abc import Sequence
from typing import Protocol

from ragrefine.models import Candidate
from ragrefine.ranking.types import FusedCandidate, RankingChannel


class RankFusion(Protocol):
    """Combine named candidate rankings into one deterministic ranking."""

    def fuse(self, channels: Sequence[RankingChannel]) -> tuple[FusedCandidate, ...]:
        """Fuse active channels without inspecting their raw ranking scores."""


class ReciprocalRankFusion:
    """Fuse rank positions using ``sum(weight / (k + rank))``.

    Zero-weight channels are explicitly inactive.  Output ties use original
    retrieval rank then candidate ID, never input/dictionary iteration order.
    """

    def __init__(self, *, k: int = 60) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self._k = k

    @property
    def k(self) -> int:
        """Return the configured RRF rank constant for trace provenance."""
        return self._k

    def fuse(self, channels: Sequence[RankingChannel]) -> tuple[FusedCandidate, ...]:
        """Return an explainable fused ranking from active named channels."""
        names: set[str] = set()
        candidates: dict[str, Candidate] = {}
        ranks: dict[str, dict[str, int]] = {}
        contributions: dict[str, dict[str, float]] = {}
        for channel in channels:
            self._validate_channel(channel, names)
            if channel.weight == 0:
                continue
            names.add(channel.name)
            seen_ids: set[str] = set()
            for rank, candidate in enumerate(channel.candidates, start=1):
                if candidate.id in seen_ids:
                    raise ValueError(f"channel {channel.name!r} contains duplicate IDs")
                seen_ids.add(candidate.id)
                existing = candidates.setdefault(candidate.id, candidate)
                if existing != candidate:
                    raise ValueError(
                        f"candidate {candidate.id!r} differs between channels"
                    )
                ranks.setdefault(candidate.id, {})[channel.name] = rank
                contributions.setdefault(candidate.id, {})[channel.name] = (
                    channel.weight / (self._k + rank)
                )
        ordered = sorted(
            candidates.values(),
            key=lambda candidate: (
                -sum(contributions[candidate.id].values()),
                (1, 0)
                if candidate.retrieval_rank is None
                else (0, candidate.retrieval_rank),
                candidate.id,
            ),
        )
        return tuple(
            FusedCandidate(
                candidate=candidate,
                rank=rank,
                score=sum(contributions[candidate.id].values()),
                channel_ranks=ranks[candidate.id],
                contributions=contributions[candidate.id],
            )
            for rank, candidate in enumerate(ordered, start=1)
        )

    @staticmethod
    def _validate_channel(channel: RankingChannel, names: set[str]) -> None:
        if not channel.name.strip():
            raise ValueError("channel name must not be empty")
        if channel.name in names:
            raise ValueError(f"duplicate active channel name: {channel.name!r}")
        if channel.weight < 0:
            raise ValueError("channel weight must be non-negative")
