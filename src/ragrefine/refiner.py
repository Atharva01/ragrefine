"""Refinement orchestration entry point."""

from collections.abc import Sequence
from time import perf_counter

from ragrefine.models import CandidateSet, RefinedCandidate
from ragrefine.results import RefinementResult
from ragrefine.tracing.models import RefinementTrace, StageTrace


class Refiner:
    """Apply the current refinement pipeline to externally retrieved candidates.

    The initial implementation is intentionally a no-op baseline: it preserves
    candidate-set order and limits the returned candidate pool to ``top_k``.
    """

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
        del query
        if top_k < 0:
            msg = "top_k must be non-negative"
            raise ValueError(msg)

        started_at = perf_counter()
        selected_candidates = tuple(
            candidate
            for candidate_set in candidate_sets
            for candidate in candidate_set.candidates
        )[:top_k]
        refined_candidates = tuple(
            RefinedCandidate(
                candidate=candidate,
                original_rank=candidate.retrieval_rank,
                final_rank=index,
                final_score=0.0,
            )
            for index, candidate in enumerate(selected_candidates, start=1)
        )
        duration_ms = (perf_counter() - started_at) * 1_000
        trace = RefinementTrace(
            stages=(
                StageTrace(
                    name="no_op_selection",
                    duration_ms=duration_ms,
                    configuration={"top_k": top_k},
                ),
            ),
            duration_ms=duration_ms,
            config_fingerprint="no-op-v1",
        )
        return RefinementResult(candidates=refined_candidates, trace=trace)
