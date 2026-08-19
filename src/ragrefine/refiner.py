"""Refinement orchestration entry point."""

from collections.abc import Sequence
from time import perf_counter

from ragrefine.errors import RerankerError
from ragrefine.models import Candidate, CandidateSet, RankingSignal, RefinedCandidate
from ragrefine.rerank.base import NeuralReranker
from ragrefine.results import RefinementResult
from ragrefine.tracing.models import RefinementTrace, StageTrace


class Refiner:
    """Apply the current refinement pipeline to externally retrieved candidates.

    The initial implementation is intentionally a no-op baseline: it preserves
    candidate-set order and limits the returned candidate pool to ``top_k``.
    """

    def __init__(self, *, reranker: NeuralReranker | None = None) -> None:
        self._reranker = reranker

    def refine(
        self,
        query: str,
        candidate_sets: Sequence[CandidateSet],
        *,
        top_k: int = 5,
    ) -> RefinementResult:
        """Return the first ``top_k`` input candidates without ranking changes.

        ``query`` is accepted as part of the stable public contract but is not
        interpreted until a later refinement stage. A negative ``top_k`` is an
        invalid selection request; zero safely returns an empty result.
        """
        if top_k < 0:
            msg = "top_k must be non-negative"
            raise ValueError(msg)

        started_at = perf_counter()
        input_candidates = tuple(
            candidate
            for candidate_set in candidate_sets
            for candidate in candidate_set.candidates
        )
        scored_candidates = self._rank(query, input_candidates)
        selected_candidates = scored_candidates[:top_k]
        refined_candidates = tuple(
            RefinedCandidate(
                candidate=scored.candidate,
                original_rank=scored.candidate.retrieval_rank,
                final_rank=index,
                final_score=scored.score,
                signals=scored.signals,
            )
            for index, scored in enumerate(selected_candidates, start=1)
        )
        duration_ms = (perf_counter() - started_at) * 1_000
        configuration: dict[str, object] = {"top_k": top_k}
        if self._reranker:
            configuration["reranker"] = self._reranker.name
            configuration.update(self._neural_configuration(scored_candidates))
        trace = RefinementTrace(
            stages=(
                StageTrace(
                    name="neural_reranking" if self._reranker else "no_op_selection",
                    duration_ms=duration_ms,
                    configuration=configuration,
                ),
            ),
            duration_ms=duration_ms,
            config_fingerprint="neural-rerank-v1" if self._reranker else "no-op-v1",
        )
        return RefinementResult(candidates=refined_candidates, trace=trace)

    def _rank(
        self, query: str, candidates: tuple[Candidate, ...]
    ) -> tuple["_RankedCandidate", ...]:
        if self._reranker is None:
            return tuple(
                _RankedCandidate(candidate, 0.0, {}) for candidate in candidates
            )
        try:
            scored = tuple(self._reranker.rank(query, candidates))
        except RerankerError:
            raise
        except Exception as error:
            raise RerankerError("neural reranker failed") from error
        if len(scored) != len(candidates) or {
            id(item.candidate) for item in scored
        } != {id(item) for item in candidates}:
            raise RerankerError(
                "neural reranker must return each input candidate exactly once"
            )
        return tuple(
            _RankedCandidate(
                item.candidate,
                item.score,
                {
                    "neural": RankingSignal(
                        item.rank,
                        item.score,
                        {
                            "model": item.model,
                            "revision": item.model_revision,
                            "backend": item.backend,
                        },
                    )
                },
            )
            for item in scored
        )

    @staticmethod
    def _neural_configuration(
        scored_candidates: Sequence["_RankedCandidate"],
    ) -> dict[str, object]:
        """Extract stable backend provenance from the first neural result."""
        if not scored_candidates:
            return {}
        details = scored_candidates[0].signals["neural"].details
        return {
            "model": details["model"],
            "revision": details["revision"],
            "backend": details["backend"],
        }


class _RankedCandidate:
    def __init__(
        self, candidate: "Candidate", score: float, signals: dict[str, RankingSignal]
    ) -> None:
        self.candidate, self.score, self.signals = candidate, score, signals
