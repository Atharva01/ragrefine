"""Unit tests for configurable structured-pattern ranking."""

from dataclasses import FrozenInstanceError

import pytest

from ragrefine.errors import PatternError
from ragrefine.models import Candidate
from ragrefine.query.patterns import PatternRegistry, PatternRule, example_pattern_rules
from ragrefine.ranking.patterns import PatternAgreement, PatternRanker


def _version_registry() -> PatternRegistry:
    return PatternRegistry((PatternRule("version", r"(?i)\bmodel[-_ ]?v\d+\b"),))


def test_registry_preserves_values_normalization_and_source_spans() -> None:
    """Extraction is auditable against the exact original text."""
    text = "Compare Model_V17 with model-v18."
    matches = _version_registry().extract(text)

    assert tuple(match.value for match in matches) == ("Model_V17", "model-v18")
    assert tuple(match.normalized_value for match in matches) == (
        "model_v17",
        "model-v18",
    )
    assert tuple(text[match.start : match.end] for match in matches) == (
        "Model_V17",
        "model-v18",
    )


def test_pattern_ranker_orders_agreement_then_absence_then_conflict() -> None:
    """Exact support outranks absence, which in turn outranks a conflicting value."""
    candidates = (
        Candidate(id="conflict", text="model-v18", retrieval_rank=1),
        Candidate(id="absent", text="general guide", retrieval_rank=2),
        Candidate(id="exact", text="model-v17 guide", retrieval_rank=3),
    )

    ranking = PatternRanker(_version_registry()).rank("model-v17", candidates)

    assert tuple(item.candidate.id for item in ranking) == (
        "exact",
        "absent",
        "conflict",
    )
    assert ranking[0].evidence[0].agreement is PatternAgreement.EXACT
    assert ranking[1].evidence[0].agreement is PatternAgreement.ABSENT
    assert ranking[2].evidence[0].agreement is PatternAgreement.CONFLICT
    assert ranking[2].evidence[0].candidate_matches[0].value == "model-v18"


def test_pattern_ranker_reports_no_applicable_query_constraint() -> None:
    """A query without configured patterns preserves deterministic source ranking."""
    candidates = (
        Candidate(id="b", text="model-v17", retrieval_rank=2),
        Candidate(id="a", text="model-v18", retrieval_rank=1),
    )

    ranking = PatternRanker(_version_registry()).rank("general help", candidates)

    assert tuple(item.candidate.id for item in ranking) == ("a", "b")
    assert all(
        not item.has_query_constraints and item.evidence == () for item in ranking
    )


def test_multiple_and_overlapping_rules_have_stable_source_order() -> None:
    """Overlapping rules are retained rather than silently discarded."""
    registry = PatternRegistry(
        (
            PatternRule("code", r"HTTP\d{3}"),
            PatternRule("number", r"\d{3}"),
        )
    )

    matches = registry.extract("HTTP503")

    assert tuple(
        (match.label, match.value, match.start, match.end) for match in matches
    ) == (
        ("code", "HTTP503", 0, 7),
        ("number", "503", 4, 7),
    )


def test_custom_normalizer_compares_equivalent_values() -> None:
    """A custom normalizer can define an application-specific equivalence."""
    registry = PatternRegistry(
        (
            PatternRule(
                "version",
                r"(?i)\bv[-_ ]?\d+\b",
                normalizer=lambda value: (
                    value.casefold().replace("_", "-").replace(" ", "-")
                ),
            ),
        )
    )

    ranking = PatternRanker(registry).rank(
        "V_17", (Candidate(id="match", text="v-17", retrieval_rank=1),)
    )

    assert ranking[0].evidence[0].agreement is PatternAgreement.EXACT


def test_invalid_or_duplicate_patterns_fail_explicitly() -> None:
    """Malformed, empty, and duplicate rules do not lead to silent corruption."""
    with pytest.raises(PatternError, match="invalid pattern"):
        PatternRule("bad", "[")
    with pytest.raises(PatternError, match="regular-expression string"):
        PatternRule("bad", 1)  # type: ignore[arg-type]
    with pytest.raises(PatternError, match="normalizer must be callable"):
        PatternRule("bad", r"v\d+", normalizer="not-a-function")  # type: ignore[arg-type]
    with pytest.raises(PatternError, match="must not match empty"):
        PatternRule("empty", r"a*")
    rule = PatternRule("version", r"v\d+")
    with pytest.raises(PatternError, match="duplicate"):
        PatternRegistry((rule, rule))
    with pytest.raises(PatternError, match="PatternRule instances"):
        PatternRegistry(("not-a-rule",))  # type: ignore[arg-type]


def test_pattern_registry_is_immutable() -> None:
    """Configured rules cannot be replaced after validation."""
    registry = _version_registry()

    with pytest.raises(FrozenInstanceError):
        registry._rules = ()  # type: ignore[misc]


def test_examples_are_opt_in_and_cover_representative_constraints() -> None:
    """Examples cover common categories without becoming default application policy."""
    labels = {rule.label for rule in example_pattern_rules()}
    assert {"version", "code", "date", "percentage", "number"}.issubset(labels)
