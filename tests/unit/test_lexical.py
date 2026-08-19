"""Unit tests for deterministic lexical relevance ranking."""

from ragrefine.models import Candidate
from ragrefine.query.normalize import normalize_text, normalized_terms
from ragrefine.ranking.lexical import LexicalRanker


def test_normalize_text_is_unicode_case_and_whitespace_stable() -> None:
    """NFKC, casefolding, and whitespace normalization retain comparable text."""
    assert normalize_text("  CAF\u00c9\u3000\uff26\uff4f\uff4f\n") == "caf\u00e9 foo"


def test_normalized_terms_keep_technical_identifiers_and_remove_repeats() -> None:
    """Technical tokens remain intact while coverage terms are unique."""
    assert normalized_terms("Model_V17 model-v17 HTTP503 HTTP503") == (
        "model-v17",
        "http503",
    )


def test_lexical_ranker_records_coverage_and_stable_evidence() -> None:
    """Each output exposes query support without changing source candidates."""
    first = Candidate(
        id="first",
        text="MODEL-V17 explains HTTP503.",
        metadata={"source": "dense"},
        retrieval_score=0.82,
        retrieval_rank=3,
    )
    second = Candidate(
        id="second", text="Only model-v17 is discussed.", retrieval_rank=1
    )

    ranking = LexicalRanker().rank("model_v17 HTTP503 http503", (second, first))

    assert tuple(item.candidate for item in ranking) == (first, second)
    assert ranking[0].coverage == 1.0
    assert ranking[0].matched_terms == ("model-v17", "http503")
    assert ranking[0].missing_terms == ()
    assert ranking[1].coverage == 0.5
    assert ranking[1].matched_terms == ("model-v17",)
    assert ranking[1].missing_terms == ("http503",)
    assert ranking[0].candidate is first
    assert ranking[0].candidate.metadata == {"source": "dense"}
    assert ranking[0].candidate.retrieval_score == 0.82
    assert ranking[0].candidate.retrieval_rank == 3


def test_lexical_ranker_breaks_equal_coverage_by_retrieval_rank_then_id() -> None:
    """Equal lexical scores never rely on set or dictionary iteration order."""
    candidates = (
        Candidate(id="z", text="match", retrieval_rank=2),
        Candidate(id="b", text="match", retrieval_rank=1),
        Candidate(id="a", text="match", retrieval_rank=1),
        Candidate(id="unknown", text="match"),
    )

    ranking = LexicalRanker().rank("match", candidates)

    assert tuple(item.candidate.id for item in ranking) == ("a", "b", "z", "unknown")
    assert tuple(item.rank for item in ranking) == (1, 2, 3, 4)


def test_lexical_ranker_handles_empty_and_punctuation_only_queries() -> None:
    """Degenerate queries produce zero coverage and empty deterministic evidence."""
    candidates = (
        Candidate(id="b", text="evidence", retrieval_rank=2),
        Candidate(id="a", text="evidence", retrieval_rank=1),
    )

    for query in ("", " !!! "):
        ranking = LexicalRanker().rank(query, candidates)
        assert tuple(item.candidate.id for item in ranking) == ("a", "b")
        assert all(item.coverage == 0.0 for item in ranking)
        assert all(
            item.matched_terms == () and item.missing_terms == () for item in ranking
        )
