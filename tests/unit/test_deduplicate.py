"""Deterministic post-ranking duplicate-suppression tests."""

from ragrefine import Candidate
from ragrefine.models import RankingSignal, RefinedCandidate
from ragrefine.selection import CandidateDeduplicator, DeduplicationConfig
from ragrefine.selection.deduplicate import content_hash, shingle_jaccard


def _ranked(candidate_id: str, text: str, rank: int) -> RefinedCandidate:
    candidate = Candidate(
        id=candidate_id,
        text=text,
        metadata={"source": candidate_id},
        retrieval_rank=rank,
        retrieval_score=1.0 / rank,
    )
    return RefinedCandidate(
        candidate=candidate,
        rank=rank,
        signals={"original": RankingSignal(rank=rank, score=10.0 / rank)},
    )


def test_exact_normalized_duplicates_retain_the_highest_ranked_candidate() -> None:
    """Unicode, case, and whitespace variants share one normalized SHA-256 hash."""
    first = _ranked("first", "CAFÉ   release", 1)
    duplicate = _ranked("duplicate", "cafe\u0301 release", 2)

    result = CandidateDeduplicator().deduplicate((first, duplicate))

    assert result.candidates == (first,)
    assert result.suppressed[0].candidate is duplicate
    assert result.suppressed[0].reason == "exact_normalized_content"
    assert result.suppressed[0].retained_candidate_id == "first"
    assert result.suppressed[0].normalized_sha256 == content_hash(first.candidate.text)
    assert result.suppressed[0].similarity is None


def test_near_duplicates_use_configurable_token_shingle_jaccard() -> None:
    """Near suppression retains the first ranked candidate and its explanation."""
    first = _ranked("first", "alpha beta gamma delta", 1)
    near = _ranked("near", "alpha beta gamma epsilon", 2)
    result = CandidateDeduplicator(
        DeduplicationConfig(near_duplicate_threshold=0.4, shingle_size=2)
    ).deduplicate((first, near))

    assert result.candidates == (first,)
    assert result.suppressed[0].reason == "near_token_shingle_jaccard"
    assert result.suppressed[0].retained_candidate_id == "first"
    assert result.suppressed[0].similarity == 0.5


def test_threshold_boundary_is_not_suppressed() -> None:
    """The documented threshold is strict: equality alone is not a duplicate."""
    first = _ranked("first", "alpha beta gamma delta", 1)
    boundary = _ranked("boundary", "alpha beta gamma epsilon", 2)
    assert shingle_jaccard(first.candidate.text, boundary.candidate.text, size=2) == 0.5

    result = CandidateDeduplicator(
        DeduplicationConfig(near_duplicate_threshold=0.5, shingle_size=2)
    ).deduplicate((first, boundary))

    assert result.candidates == (first, boundary)
    assert result.suppressed == ()


def test_near_duplicate_matching_can_be_disabled_for_exact_only_policy() -> None:
    """Exact-only mode retains non-identical candidates regardless of similarity."""
    first = _ranked("first", "alpha beta gamma delta", 1)
    near = _ranked("near", "alpha beta gamma epsilon", 2)

    result = CandidateDeduplicator(
        DeduplicationConfig(near_duplicate_threshold=None, shingle_size=2)
    ).deduplicate((first, near))

    assert result.candidates == (first, near)
    assert result.suppressed == ()


def test_unrelated_candidates_and_original_evidence_are_preserved() -> None:
    """Suppression never rewrites candidates, metadata, ranks, scores, or signals."""
    first = _ranked("first", "alpha beta gamma", 1)
    unrelated = _ranked("unrelated", "unrelated diagnostic evidence", 2)

    result = CandidateDeduplicator().deduplicate((first, unrelated))

    assert result.candidates == (first, unrelated)
    assert result.candidates[0].candidate.metadata == {"source": "first"}
    assert result.candidates[1].candidate.retrieval_score == 0.5
    assert result.candidates[1].signals["original"].rank == 2


def test_duplicate_suppression_is_deterministic() -> None:
    """Repeated runs over the same ranked input return identical records."""
    ranked = (
        _ranked("first", "alpha beta gamma delta", 1),
        _ranked("near", "alpha beta gamma epsilon", 2),
        _ranked("exact", "ALPHA beta gamma delta", 3),
    )
    deduplicator = CandidateDeduplicator(
        DeduplicationConfig(near_duplicate_threshold=0.4, shingle_size=2)
    )

    assert deduplicator.deduplicate(ranked) == deduplicator.deduplicate(ranked)
