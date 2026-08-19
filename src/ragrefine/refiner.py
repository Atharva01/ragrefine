"""Explicit orchestration of independent candidate-ranking channels."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter

from ragrefine.errors import (
    ChannelExecutionError,
    MultipleActiveChannelsError,
    RerankerError,
)
from ragrefine.models import Candidate, CandidateSet, RankingSignal, RefinedCandidate
from ragrefine.orchestration import ChannelConfig, RefinerConfig, _ChannelResult
from ragrefine.ranking.lexical import LexicalRanker
from ragrefine.ranking.patterns import PatternRanker
from ragrefine.ranking.rrf import RankFusion
from ragrefine.ranking.types import RankingChannel
from ragrefine.rerank.base import NeuralReranker
from ragrefine.results import RefinementResult
from ragrefine.tracing.models import RefinementTrace, StageTrace


class Refiner:
    """Refine a fixed candidate pool through explicitly configured channels.

    Every enabled channel receives the same complete input pool. A single
    successful channel is selected directly; multiple successful channels need
    a configured rank-fusion implementation. This preserves the independence
    required for B0, B1, B2, and B3 experiments.
    """

    def __init__(
        self,
        *,
        reranker: NeuralReranker | None = None,
        lexical_ranker: LexicalRanker | None = None,
        pattern_ranker: PatternRanker | None = None,
        fusion: RankFusion | None = None,
        config: RefinerConfig | None = None,
    ) -> None:
        self._reranker = reranker
        self._lexical_ranker = lexical_ranker
        self._pattern_ranker = pattern_ranker
        self._fusion = fusion
        self._legacy_configuration = config is None
        self._config = config or self._legacy_config(reranker, fusion)
        if self._config.fusion_enabled and self._fusion is None:
            raise ValueError("fusion_enabled requires a RankFusion implementation")

    @staticmethod
    def _legacy_config(
        reranker: NeuralReranker | None, fusion: RankFusion | None
    ) -> RefinerConfig:
        """Map the original constructor to explicit B0/B1 configurations."""
        if fusion is not None:
            return RefinerConfig(
                original=ChannelConfig(enabled=True),
                neural=ChannelConfig(enabled=reranker is not None),
                fusion_enabled=True,
            )
        if reranker is not None:
            return RefinerConfig(
                original=ChannelConfig(enabled=False),
                neural=ChannelConfig(enabled=True),
            )
        return RefinerConfig(original=ChannelConfig(enabled=True))

    def refine(
        self,
        query: str,
        candidate_sets: Sequence[CandidateSet],
        *,
        top_k: int = 5,
    ) -> RefinementResult:
        """Run enabled channels and select only after final ranking is resolved."""
        if top_k < 0:
            raise ValueError("top_k must be non-negative")

        started_at = perf_counter()
        candidates = tuple(
            candidate
            for candidate_set in candidate_sets
            for candidate in candidate_set.candidates
        )
        channel_results, failures = self._run_channels(query, candidates)
        ranked_candidates = self._resolve_final_ranking(channel_results)
        refined_candidates = tuple(
            RefinedCandidate(
                candidate=ranked.candidate,
                original_rank=ranked.candidate.retrieval_rank,
                final_rank=rank,
                final_score=ranked.score,
                signals=ranked.signals,
            )
            for rank, ranked in enumerate(ranked_candidates[:top_k], start=1)
        )
        duration_ms = (perf_counter() - started_at) * 1_000
        trace = RefinementTrace(
            stages=(
                StageTrace(
                    name=self._stage_name(channel_results),
                    duration_ms=duration_ms,
                    configuration=self._trace_configuration(
                        top_k, channel_results, failures
                    ),
                ),
            ),
            duration_ms=duration_ms,
            config_fingerprint=self._config_fingerprint(),
        )
        return RefinementResult(candidates=refined_candidates, trace=trace)

    def _run_channels(
        self, query: str, candidates: tuple[Candidate, ...]
    ) -> tuple[tuple[_ChannelResult, ...], tuple[Mapping[str, object], ...]]:
        """Run enabled channels independently against the unchanged pool."""
        successful: list[_ChannelResult] = []
        failures: list[Mapping[str, object]] = []
        configured_channels: tuple[
            tuple[str, ChannelConfig, Callable[[], _ChannelResult]], ...
        ] = (
            ("original", self._config.original, lambda: self._run_original(candidates)),
            (
                "neural",
                self._config.neural,
                lambda: self._run_neural(query, candidates),
            ),
            (
                "lexical",
                self._config.lexical,
                lambda: self._run_lexical(query, candidates),
            ),
            (
                "pattern",
                self._config.pattern,
                lambda: self._run_pattern(query, candidates),
            ),
        )
        for name, channel_config, execute in configured_channels:
            if not channel_config.enabled:
                continue
            try:
                result = execute()
                self._validate_channel_result(name, result, candidates)
            except Exception as error:
                if channel_config.required:
                    if (
                        self._legacy_configuration
                        and name == "neural"
                        and isinstance(error, RerankerError)
                    ):
                        raise
                    raise ChannelExecutionError(
                        f"required {name!r} ranking channel failed: {error}"
                    ) from error
                failures.append(
                    {
                        "channel": name,
                        "status": "failed",
                        "failure_type": type(error).__name__,
                        "reason": str(error),
                    }
                )
            else:
                successful.append(result)
        return tuple(successful), tuple(failures)

    @staticmethod
    def _run_original(candidates: tuple[Candidate, ...]) -> _ChannelResult:
        return _ChannelResult(
            name="original",
            ranking=candidates,
            evidence={
                candidate.id: {
                    "rank": rank,
                    "retrieval_rank": candidate.retrieval_rank,
                    "retrieval_score": candidate.retrieval_score,
                }
                for rank, candidate in enumerate(candidates, start=1)
            },
        )

    def _run_neural(
        self, query: str, candidates: tuple[Candidate, ...]
    ) -> _ChannelResult:
        if self._reranker is None:
            raise RerankerError(
                "neural channel is enabled but no reranker is configured"
            )
        try:
            scored = tuple(self._reranker.rank(query, candidates))
        except RerankerError:
            raise
        except Exception as error:
            raise RerankerError("neural reranker failed") from error
        return _ChannelResult(
            name="neural",
            ranking=tuple(item.candidate for item in scored),
            evidence={
                item.candidate.id: {
                    "rank": item.rank,
                    "raw_score": item.score,
                    "model": item.model,
                    "revision": item.model_revision,
                    "backend": item.backend,
                }
                for item in scored
            },
            metadata={"reranker": self._reranker.name},
        )

    def _run_lexical(
        self, query: str, candidates: tuple[Candidate, ...]
    ) -> _ChannelResult:
        if self._lexical_ranker is None:
            raise ValueError(
                "lexical channel is enabled but no lexical ranker is configured"
            )
        ranked = self._lexical_ranker.rank(query, candidates)
        return _ChannelResult(
            name="lexical",
            ranking=tuple(item.candidate for item in ranked),
            evidence={
                item.candidate.id: {
                    "rank": item.rank,
                    "coverage": item.coverage,
                    "matched_terms": item.matched_terms,
                    "missing_terms": item.missing_terms,
                }
                for item in ranked
            },
        )

    def _run_pattern(
        self, query: str, candidates: tuple[Candidate, ...]
    ) -> _ChannelResult:
        if self._pattern_ranker is None:
            raise ValueError(
                "pattern channel is enabled but no pattern ranker is configured"
            )
        ranked = self._pattern_ranker.rank(query, candidates)
        return _ChannelResult(
            name="pattern",
            ranking=tuple(item.candidate for item in ranked),
            evidence={
                item.candidate.id: {
                    "rank": item.rank,
                    "has_query_constraints": item.has_query_constraints,
                    "exact_agreements": item.exact_agreements,
                    "conflicts": item.conflicts,
                    "absent_constraints": item.absent_constraints,
                    "evidence": item.evidence,
                }
                for item in ranked
            },
        )

    @staticmethod
    def _validate_channel_result(
        name: str, result: _ChannelResult, candidates: tuple[Candidate, ...]
    ) -> None:
        expected_ids = {candidate.id for candidate in candidates}
        result_ids = tuple(candidate.id for candidate in result.ranking)
        if result.name != name:
            raise ValueError("channel result name does not match its configuration")
        if len(result_ids) != len(candidates) or len(set(result_ids)) != len(
            result_ids
        ):
            raise ValueError("channel must return each input candidate exactly once")
        if set(result_ids) != expected_ids or set(result.evidence) != expected_ids:
            raise ValueError(
                "channel result must cover the complete input candidate pool"
            )

    def _resolve_final_ranking(
        self, results: tuple[_ChannelResult, ...]
    ) -> tuple["_RankedCandidate", ...]:
        if not results:
            raise ChannelExecutionError("no ranking channels completed successfully")
        if len(results) == 1:
            result = results[0]
            return tuple(
                _RankedCandidate(
                    candidate=candidate,
                    score=self._direct_score(result.evidence[candidate.id]),
                    signals={result.name: self._signal(result.evidence[candidate.id])},
                )
                for candidate in result.ranking
            )
        if not self._config.fusion_enabled or self._fusion is None:
            raise MultipleActiveChannelsError(
                "multiple ranking channels are active but fusion is disabled"
            )
        fused = self._fusion.fuse(
            tuple(
                RankingChannel(
                    result.name,
                    result.ranking,
                    self._channel_config(result.name).weight,
                )
                for result in results
            )
        )
        return tuple(
            _RankedCandidate(
                candidate=item.candidate,
                score=item.score,
                signals={
                    **{
                        result.name: self._signal(result.evidence[item.candidate.id])
                        for result in results
                    },
                    "fusion": RankingSignal(
                        rank=item.rank,
                        score=item.score,
                        details={
                            "channel_ranks": item.channel_ranks,
                            "contributions": item.contributions,
                        },
                    ),
                },
            )
            for item in fused
        )

    def _channel_config(self, name: str) -> ChannelConfig:
        if name == "original":
            return self._config.original
        if name == "neural":
            return self._config.neural
        if name == "lexical":
            return self._config.lexical
        if name == "pattern":
            return self._config.pattern
        raise ValueError(f"unknown ranking channel: {name}")

    @staticmethod
    def _direct_score(evidence: Mapping[str, object]) -> float:
        score = evidence.get("raw_score", evidence.get("coverage", 0.0))
        return float(score) if isinstance(score, int | float) else 0.0

    @staticmethod
    def _signal(evidence: Mapping[str, object]) -> RankingSignal:
        score = evidence.get("raw_score", evidence.get("coverage"))
        rank = evidence["rank"]
        if not isinstance(rank, int):
            raise ValueError("channel evidence rank must be an integer")
        return RankingSignal(
            rank=rank,
            score=float(score) if isinstance(score, int | float) else None,
            details={
                key: value
                for key, value in evidence.items()
                if key not in {"rank", "raw_score", "coverage"}
            },
        )

    def _trace_configuration(
        self,
        top_k: int,
        results: tuple[_ChannelResult, ...],
        failures: tuple[Mapping[str, object], ...],
    ) -> Mapping[str, object]:
        if self._legacy_configuration and not failures:
            configuration: dict[str, object] = {"top_k": top_k}
            if self._reranker is not None and results:
                neural = next((item for item in results if item.name == "neural"), None)
                if neural is not None and neural.ranking:
                    evidence = neural.evidence[neural.ranking[0].id]
                    configuration.update(
                        {
                            "reranker": self._reranker.name,
                            "model": evidence["model"],
                            "revision": evidence["revision"],
                            "backend": evidence["backend"],
                        }
                    )
            if self._fusion is not None:
                configuration["fusion"] = "rrf"
                configuration["channels"] = [item.name for item in results]
            return configuration
        fusion: Mapping[str, object] = {"enabled": self._config.fusion_enabled}
        if self._config.fusion_enabled and self._fusion is not None:
            fusion = {"enabled": True, "k": getattr(self._fusion, "k", None)}
        return {
            "top_k": top_k,
            "participating_channels": [item.name for item in results],
            "channel_weights": {
                name: channel_config.weight
                for name, channel_config in (
                    ("original", self._config.original),
                    ("neural", self._config.neural),
                    ("lexical", self._config.lexical),
                    ("pattern", self._config.pattern),
                )
                if channel_config.enabled
            },
            "omitted_channels": list(failures),
            "fusion": fusion,
        }

    def _stage_name(self, results: tuple[_ChannelResult, ...]) -> str:
        if len(results) > 1 and self._config.fusion_enabled:
            return "rank_fusion"
        if self._legacy_configuration and results:
            if results[0].name == "original":
                return "no_op_selection"
            if results[0].name == "neural":
                return "neural_reranking"
        return f"{results[0].name}_ranking"

    def _config_fingerprint(self) -> str:
        if not self._legacy_configuration:
            return "channel-orchestration-v1"
        if self._fusion is not None:
            return "rank-fusion-v1"
        return "neural-rerank-v1" if self._reranker is not None else "no-op-v1"


@dataclass(frozen=True, slots=True)
class _RankedCandidate:
    candidate: Candidate
    score: float
    signals: Mapping[str, RankingSignal]
