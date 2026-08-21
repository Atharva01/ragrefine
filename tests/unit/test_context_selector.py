"""Deterministic post-ranking context-selection tests."""

from ragrefine import Candidate
from ragrefine.models import RankingSignal, RefinedCandidate
from ragrefine.selection import CandidateDeduplicator, RankPreservingContextSelector


class _WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())


def _ranked(candidate_id: str, text: str, rank: int) -> RefinedCandidate:
    return RefinedCandidate(
        candidate=Candidate(
            id=candidate_id,
            text=text,
            metadata={"source": candidate_id},
            retrieval_rank=rank,
            retrieval_score=1.0 / rank,
        ),
        rank=rank,
        signals={"original": RankingSignal(rank=rank, score=10.0 / rank)},
    )


def test_exact_fit_budget_selects_unchanged_candidate() -> None:
    """A candidate exactly fitting the budget is selected without text mutation."""
    candidate = _ranked("one", "two token", 1)

    result = RankPreservingContextSelector(_WordCounter()).select(
        (candidate,), max_tokens=2
    )

    assert result.candidates == (candidate,)
    assert result.records[0].status == "selected"
    assert result.records[0].selected_position == 1
    assert result.records[0].token_count == 2
    assert result.candidates[0].candidate.text == "two token"


def test_oversized_candidate_is_excluded_without_blocking_later_ranked_context() -> (
    None
):
    """Budget exclusion skips, never truncates, an oversized candidate."""
    oversized = _ranked("oversized", "one two three", 1)
    later = _ranked("later", "fits", 2)

    result = RankPreservingContextSelector(_WordCounter()).select(
        (oversized, later), max_tokens=1
    )

    assert result.candidates == (later,)
    assert result.records[0].status == "budget_excluded"
    assert result.records[0].candidate_id == "oversized"
    assert result.records[1].status == "selected"
    assert result.records[1].selected_position == 1


def test_empty_budget_excludes_every_nonempty_candidate() -> None:
    """A zero budget is valid and yields explicit budget-exclusion records."""
    candidates = (_ranked("one", "alpha", 1), _ranked("two", "beta", 2))

    result = RankPreservingContextSelector(_WordCounter()).select(
        candidates, max_tokens=0
    )

    assert result.candidates == ()
    assert [record.status for record in result.records] == [
        "budget_excluded",
        "budget_excluded",
    ]


def test_combined_constraints_preserve_order_and_trace_all_outcomes() -> None:
    """Top-K and budget compose after duplicates are suppressed."""
    first = _ranked("first", "alpha beta", 1)
    duplicate = _ranked("duplicate", "ALPHA beta", 2)
    second = _ranked("second", "gamma", 3)
    third = _ranked("third", "delta", 4)
    deduplicated = CandidateDeduplicator().deduplicate(
        (first, duplicate, second, third)
    )

    result = RankPreservingContextSelector(_WordCounter()).select(
        deduplicated.candidates,
        top_k=2,
        max_tokens=3,
        suppressed=deduplicated.suppressed,
    )

    assert result.candidates == (first, second)
    assert [record.status for record in result.records] == [
        "duplicate_suppressed",
        "selected",
        "selected",
        "excluded",
    ]
    assert result.records[0].retained_candidate_id == "first"
    assert result.records[-1].reason == "top_k_limit"
    assert [record.selected_position for record in result.records] == [None, 1, 2, None]


def test_selection_is_deterministic_and_validates_token_counter_requirement() -> None:
    """Identical ranked inputs replay identically and budgets require a counter."""
    candidates = (_ranked("one", "alpha", 1), _ranked("two", "beta", 2))
    selector = RankPreservingContextSelector(_WordCounter())

    assert selector.select(candidates, top_k=1) == selector.select(candidates, top_k=1)

    try:
        RankPreservingContextSelector().select(candidates, max_tokens=1)
    except ValueError as error:
        assert str(error) == "max_tokens requires a TokenCounter"
    else:
        raise AssertionError("expected a TokenCounter configuration error")
