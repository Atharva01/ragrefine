"""Unit tests for the no-op refinement baseline."""

import pytest

from ragrefine.config import ChannelConfig, RefinerConfig
from ragrefine.errors import (
    ChannelExecutionError,
    MultipleActiveChannelsError,
    RerankerError,
)
from ragrefine.models import Candidate, CandidateSet
from ragrefine.query.patterns import PatternRegistry, PatternRule
from ragrefine.ranking.lexical import LexicalRanker
from ragrefine.ranking.patterns import PatternRanker
from ragrefine.ranking.rrf import ReciprocalRankFusion
from ragrefine.refiner import Refiner
from ragrefine.rerank.base import ScoredCandidate
from ragrefine.selection import CandidateDeduplicator


def _stage(result: object, name: str) -> object:
    trace = getattr(result, "trace")
    return next(stage for stage in trace.stages if stage.name == name)


class ReversingReranker:
    """Fast fake that makes neural-stage behaviour observable without a model."""

    name = "test-reranker"

    def rank(
        self, query: str, candidates: tuple[Candidate, ...]
    ) -> tuple[ScoredCandidate, ...]:
        del query
        return tuple(
            ScoredCandidate(
                candidate=candidate,
                score=float(index),
                rank=index,
                model="test-model",
                model_revision="abc123",
                backend="test-backend",
            )
            for index, candidate in enumerate(reversed(candidates), start=1)
        )


def test_refiner_preserves_input_order_and_candidate_provenance() -> None:
    """The no-op pipeline returns caller-owned candidates in input order."""
    first = Candidate(id="first", text="First evidence", retrieval_rank=4)
    second = Candidate(id="second", text="Second evidence", retrieval_rank=9)
    result = Refiner().refine(
        "query",
        CandidateSet(name="externally-merged", candidates=(first, second)),
        top_k=2,
    )

    assert tuple(item.candidate for item in result.candidates) == (first, second)
    assert tuple(item.rank for item in result.candidates) == (1, 2)
    assert tuple(item.candidate.retrieval_rank for item in result.candidates) == (4, 9)


def test_refiner_handles_empty_and_oversized_selection() -> None:
    """Empty input and a large requested limit both complete successfully."""
    refiner = Refiner()
    empty_result = refiner.refine(
        "query", CandidateSet(name="empty", candidates=()), top_k=5
    )
    candidate = Candidate(id="only", text="Evidence")
    oversized_result = refiner.refine(
        "query",
        CandidateSet(name="dense", candidates=(candidate,)),
        top_k=10,
    )

    assert empty_result.candidates == ()
    assert tuple(item.candidate for item in oversized_result.candidates) == (candidate,)


def test_refiner_honors_zero_and_rejects_negative_top_k() -> None:
    """Selection limits are explicit and never invoke a hidden fallback."""
    candidate_set = CandidateSet(
        name="dense", candidates=(Candidate(id="one", text="Evidence"),)
    )

    assert Refiner().refine("query", candidate_set, top_k=0).candidates == ()
    with pytest.raises(ValueError, match="top_k must be non-negative"):
        Refiner().refine("query", candidate_set, top_k=-1)


def test_refiner_requires_one_externally_merged_candidate_set() -> None:
    """Candidate discovery and multi-source merging remain outside ragrefine."""
    candidate_set = CandidateSet(name="dense", candidates=())

    with pytest.raises(TypeError, match="merge multiple sources upstream"):
        Refiner().refine("query", (candidate_set, candidate_set))  # type: ignore[arg-type]


def test_refiner_records_minimal_no_op_trace() -> None:
    """The trace captures stage timing and configuration without integrations."""
    result = Refiner().refine(
        "query", CandidateSet(name="empty", candidates=()), top_k=3
    )

    assert len(result.trace.config_fingerprint) == 64
    assert _stage(result, "original_ranking").configuration["status"] == "completed"
    assert _stage(result, "context_selection").configuration["top_k"] == 3
    assert result.trace.duration_ms >= 0


def test_refiner_applies_neural_order_before_top_k_and_records_trace() -> None:
    """The neural stage preserves evidence and exposes ranking provenance."""
    first = Candidate(
        id="first",
        text="First evidence",
        metadata={"source": "dense"},
        retrieval_score=0.7,
        retrieval_rank=1,
    )
    second = Candidate(id="second", text="Second evidence", retrieval_rank=2)
    result = Refiner(reranker=ReversingReranker()).refine(
        "query", CandidateSet(name="dense", candidates=(first, second)), top_k=1
    )
    repeated = Refiner(reranker=ReversingReranker()).refine(
        "query", CandidateSet(name="dense", candidates=(first, second)), top_k=1
    )

    selected = result.candidates[0]
    stage = _stage(result, "neural_ranking")
    assert selected.candidate is second
    assert tuple(item.candidate for item in repeated.candidates) == (second,)
    assert selected.candidate.retrieval_rank == 2
    assert selected.signals["neural"].rank == 1
    assert selected.signals["neural"].score == 1.0
    assert selected.signals["neural"].details == {
        "model": "test-model",
        "revision": "abc123",
        "backend": "test-backend",
    }
    assert stage.name == "neural_ranking"
    assert stage.duration_ms >= 0
    assert (
        _stage(result, "context_selection").configuration["reranker"] == "test-reranker"
    )


def test_refiner_surfaces_neural_reranker_failure() -> None:
    """The pipeline never silently changes back to B0 after a backend failure."""

    class FailingReranker:
        name = "failing-reranker"

        def rank(
            self, query: str, candidates: tuple[Candidate, ...]
        ) -> tuple[ScoredCandidate, ...]:
            del query, candidates
            raise RerankerError("backend unavailable")

    with pytest.raises(RerankerError, match="backend unavailable"):
        Refiner(reranker=FailingReranker()).refine(
            "query",
            CandidateSet(name="dense", candidates=(Candidate(id="one", text="e"),)),
        )


def test_refiner_applies_opt_in_fusion_after_active_channels() -> None:
    """Fusion retains source candidates and exposes RRF contribution evidence."""
    first = Candidate(id="first", text="first", retrieval_rank=1)
    second = Candidate(id="second", text="second", retrieval_rank=2)
    result = Refiner(
        reranker=ReversingReranker(), fusion=ReciprocalRankFusion()
    ).refine("query", CandidateSet(name="dense", candidates=(first, second)))

    assert _stage(result, "rank_fusion").configuration["enabled"] is True
    assert _stage(result, "context_selection").configuration["channels"] == [
        "original",
        "neural",
    ]
    assert result.candidates[0].signals["fusion"].details["channel_ranks"] == {
        "original": 1,
        "neural": 2,
    }


def _candidate_set() -> CandidateSet:
    return CandidateSet(
        name="dense",
        candidates=(
            Candidate(id="one", text="model-v17 returned HTTP 503", retrieval_rank=1),
            Candidate(id="two", text="model-v16 returned HTTP 503", retrieval_rank=2),
            Candidate(id="three", text="general deployment advice", retrieval_rank=3),
        ),
    )


def _pattern_ranker() -> PatternRanker:
    return PatternRanker(PatternRegistry((PatternRule("version", r"v\d+"),)))


@pytest.mark.parametrize(
    ("config", "expected_signal"),
    (
        (RefinerConfig(), "original"),
        (
            RefinerConfig(
                original=ChannelConfig(enabled=False),
                neural=ChannelConfig(enabled=True),
            ),
            "neural",
        ),
        (
            RefinerConfig(
                original=ChannelConfig(enabled=False),
                lexical=ChannelConfig(enabled=True),
            ),
            "lexical",
        ),
        (
            RefinerConfig(
                original=ChannelConfig(enabled=False),
                pattern=ChannelConfig(enabled=True),
            ),
            "pattern",
        ),
    ),
)
def test_refiner_supports_independent_single_channel_modes(
    config: RefinerConfig, expected_signal: str
) -> None:
    """B0, B1, B2-L, and B2-P each remain independently executable."""
    result = Refiner(
        reranker=ReversingReranker(),
        lexical_ranker=LexicalRanker(),
        pattern_ranker=_pattern_ranker(),
        config=config,
    ).refine("model-v17 HTTP 503", _candidate_set())

    assert set(result.candidates[0].signals) == {expected_signal}
    assert _stage(result, "context_selection").configuration[
        "participating_channels"
    ] == [expected_signal]


@pytest.mark.parametrize(
    "config",
    (
        RefinerConfig(
            original=ChannelConfig(enabled=True),
            neural=ChannelConfig(enabled=True),
            fusion_enabled=True,
        ),
        RefinerConfig(
            original=ChannelConfig(enabled=True),
            lexical=ChannelConfig(enabled=True),
            fusion_enabled=True,
        ),
        RefinerConfig(
            original=ChannelConfig(enabled=True),
            pattern=ChannelConfig(enabled=True),
            fusion_enabled=True,
        ),
        RefinerConfig(
            original=ChannelConfig(enabled=False),
            lexical=ChannelConfig(enabled=True),
            pattern=ChannelConfig(enabled=True),
            fusion_enabled=True,
        ),
        RefinerConfig(
            original=ChannelConfig(enabled=True),
            lexical=ChannelConfig(enabled=True),
            pattern=ChannelConfig(enabled=True),
            fusion_enabled=True,
        ),
        RefinerConfig(
            original=ChannelConfig(enabled=True),
            neural=ChannelConfig(enabled=True),
            lexical=ChannelConfig(enabled=True),
            pattern=ChannelConfig(enabled=True),
            fusion_enabled=True,
        ),
    ),
)
def test_refiner_fuses_each_valid_multi_channel_configuration(
    config: RefinerConfig,
) -> None:
    """B3-capable combinations preserve upstream evidence and RRF provenance."""
    candidates = _candidate_set()
    result = Refiner(
        reranker=ReversingReranker(),
        lexical_ranker=LexicalRanker(),
        pattern_ranker=_pattern_ranker(),
        fusion=ReciprocalRankFusion(k=23),
        config=config,
    ).refine("model-v17 HTTP 503", candidates, top_k=2)

    stage = _stage(result, "context_selection")
    channels = stage.configuration["participating_channels"]
    assert _stage(result, "rank_fusion").configuration == {"enabled": True}
    assert stage.configuration["fusion"] == {"enabled": True, "k": 23}
    assert set(result.candidates[0].signals) == {*channels, "fusion"}
    assert set(item.candidate.id for item in result.candidates).issubset(
        {candidate.id for candidate in candidates.candidates}
    )
    assert tuple(item.rank for item in result.candidates) == (1, 2)


def test_refiner_rejects_multiple_active_channels_without_fusion() -> None:
    """Independent rankings cannot become an implicit sequential pipeline."""
    with pytest.raises(MultipleActiveChannelsError, match="fusion is disabled"):
        Refiner(
            lexical_ranker=LexicalRanker(),
            config=RefinerConfig(lexical=ChannelConfig(enabled=True)),
        ).refine("model-v17", _candidate_set())


def test_refiner_rejects_enabled_fusion_without_implementation() -> None:
    """Fusion configuration is explicit and cannot silently do nothing."""
    with pytest.raises(ValueError, match="requires a RankFusion"):
        Refiner(config=RefinerConfig(fusion_enabled=True))


def test_refiner_required_and_optional_channel_failures_are_explicit() -> None:
    """Optional failures are traceable; required failures abort refinement."""
    required_config = RefinerConfig(
        original=ChannelConfig(enabled=False), neural=ChannelConfig(enabled=True)
    )
    with pytest.raises(ChannelExecutionError, match="required 'neural'"):
        Refiner(config=required_config).refine("query", _candidate_set())

    optional_config = RefinerConfig(
        original=ChannelConfig(enabled=True),
        neural=ChannelConfig(enabled=True, required=False),
        fusion_enabled=True,
    )
    result = Refiner(fusion=ReciprocalRankFusion(), config=optional_config).refine(
        "query", _candidate_set()
    )

    stage = _stage(result, "context_selection")
    assert stage.configuration["participating_channels"] == ["original"]
    assert stage.configuration["channel_weights"] == {"original": 1.0, "neural": 1.0}
    assert stage.configuration["omitted_channels"] == [
        {
            "channel": "neural",
            "status": "failed",
            "failure_type": "RerankerError",
            "reason": "neural channel is enabled but no reranker is configured",
        }
    ]


def test_refiner_rejects_when_all_enabled_channels_are_optional_failures() -> None:
    """A traceable omission still cannot create a hidden ranking fallback."""
    with pytest.raises(ChannelExecutionError, match="no ranking channels"):
        Refiner(
            config=RefinerConfig(
                original=ChannelConfig(enabled=False),
                neural=ChannelConfig(enabled=True, required=False),
            )
        ).refine("query", _candidate_set())


def test_optional_configuration_failure_propagates() -> None:
    """Only typed backend failures are eligible for optional-channel suppression."""
    with pytest.raises(ValueError, match="no lexical ranker"):
        Refiner(
            config=RefinerConfig(
                original=ChannelConfig(enabled=False),
                lexical=ChannelConfig(enabled=True, required=False),
            )
        ).refine("query", _candidate_set())


def test_refiner_applies_top_k_after_fusion_and_is_deterministic() -> None:
    """All channels rank the full pool before final selection is truncated."""
    config = RefinerConfig(
        original=ChannelConfig(enabled=True),
        lexical=ChannelConfig(enabled=True),
        pattern=ChannelConfig(enabled=True),
        fusion_enabled=True,
    )
    refiner = Refiner(
        lexical_ranker=LexicalRanker(),
        pattern_ranker=_pattern_ranker(),
        fusion=ReciprocalRankFusion(),
        config=config,
    )
    first = refiner.refine("model-v17 HTTP 503", _candidate_set(), top_k=1)
    second = refiner.refine("model-v17 HTTP 503", _candidate_set(), top_k=1)

    assert len(first.candidates) == 1
    assert first.candidates == second.candidates
    assert first.trace.stages[0].configuration == second.trace.stages[0].configuration
    signals = first.candidates[0].signals
    assert set(signals) == {"original", "lexical", "pattern", "fusion"}
    assert signals["fusion"].details["channel_ranks"] == {
        "original": signals["original"].rank,
        "lexical": signals["lexical"].rank,
        "pattern": signals["pattern"].rank,
    }


def test_refiner_applies_deduplication_and_budget_only_after_full_ranking() -> None:
    """Final selection preserves ranked evidence and records every omission."""
    candidates = CandidateSet(
        name="dense",
        candidates=(
            Candidate(id="first", text="one two", retrieval_rank=1),
            Candidate(id="duplicate", text="ONE two", retrieval_rank=2),
            Candidate(id="later", text="fits", retrieval_rank=3),
        ),
    )

    class WordCounter:
        def count(self, text: str) -> int:
            return len(text.split())

    result = Refiner(
        deduplicator=CandidateDeduplicator(), token_counter=WordCounter()
    ).refine("query", candidates, top_k=2, max_tokens=1)

    assert tuple(item.candidate.id for item in result.candidates) == ("later",)
    assert result.candidates[0].rank == 3
    assert [record.status for record in result.selection] == [
        "duplicate_suppressed",
        "budget_excluded",
        "selected",
    ]
    assert result.selection[-1].selected_position == 1
    assert _stage(result, "context_selection").configuration["records"] == [
        {
            "candidate_id": "duplicate",
            "status": "duplicate_suppressed",
            "reason": "exact_normalized_content",
            "selected_position": None,
            "token_count": None,
            "retained_candidate_id": "first",
        },
        {
            "candidate_id": "first",
            "status": "budget_excluded",
            "reason": "max_tokens_exceeded",
            "selected_position": None,
            "token_count": 2,
            "retained_candidate_id": None,
        },
        {
            "candidate_id": "later",
            "status": "selected",
            "reason": "within_constraints",
            "selected_position": 1,
            "token_count": 1,
            "retained_candidate_id": None,
        },
    ]


def test_refiner_matches_direct_lexical_composition_and_fingerprints_config() -> None:
    """Refiner preserves direct lexical ordering/evidence for equivalent inputs."""
    candidates = _candidate_set()
    direct = LexicalRanker().rank("model-v17 HTTP 503", candidates.candidates)
    config = RefinerConfig(
        original=ChannelConfig(enabled=False), lexical=ChannelConfig(enabled=True)
    )
    first = Refiner(lexical_ranker=LexicalRanker(), config=config).refine(
        "model-v17 HTTP 503", candidates
    )
    second = Refiner(lexical_ranker=LexicalRanker(), config=config).refine(
        "model-v17 HTTP 503", candidates
    )

    assert [item.candidate for item in first.candidates] == [
        item.candidate for item in direct
    ]
    assert [item.signals["lexical"].rank for item in first.candidates] == [
        item.rank for item in direct
    ]
    assert first.trace.config_fingerprint == second.trace.config_fingerprint
    assert (
        first.trace.config_fingerprint
        != Refiner().refine("query", candidates).trace.config_fingerprint
    )
