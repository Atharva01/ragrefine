"""Unit tests for deterministic weighted reciprocal rank fusion."""

import pytest

from ragrefine.models import Candidate
from ragrefine.ranking.rrf import ReciprocalRankFusion
from ragrefine.ranking.types import RankingChannel


def _candidate(identifier: str, rank: int) -> Candidate:
    return Candidate(id=identifier, text=identifier, retrieval_rank=rank)


def test_rrf_combines_rank_positions_and_exposes_contributions() -> None:
    """RRF uses rank positions only and records each channel contribution."""
    first, second = _candidate("a", 1), _candidate("b", 2)
    fused = ReciprocalRankFusion(k=60).fuse(
        (RankingChannel("one", (first, second)), RankingChannel("two", (second, first)))
    )
    assert tuple(item.candidate.id for item in fused) == ("a", "b")
    assert fused[0].channel_ranks == {"one": 1, "two": 2}
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_omits_inactive_channels_and_handles_partial_overlap() -> None:
    """Zero-weight channels contribute no synthetic rank evidence."""
    a, b, c = _candidate("a", 1), _candidate("b", 2), _candidate("c", 3)
    fused = ReciprocalRankFusion().fuse(
        (RankingChannel("active", (a, b)), RankingChannel("inactive", (c,), 0))
    )
    assert tuple(item.candidate.id for item in fused) == ("a", "b")
    assert fused[1].channel_ranks == {"active": 2}


def test_rrf_validates_configuration_and_ties() -> None:
    """Invalid configuration fails explicitly; ties use the ranking contract."""
    with pytest.raises(ValueError, match="k must be positive"):
        ReciprocalRankFusion(k=0)
    a, b = _candidate("a", 1), _candidate("b", 2)
    assert tuple(
        item.candidate.id
        for item in ReciprocalRankFusion().fuse((RankingChannel("x", (b, a)),))
    ) == ("b", "a")
    with pytest.raises(ValueError, match="non-negative"):
        ReciprocalRankFusion().fuse((RankingChannel("bad", (a,), -1),))
