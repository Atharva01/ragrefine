"""Refinement result contract."""

from dataclasses import dataclass

from ragrefine.models import RefinedCandidate
from ragrefine.tracing.models import RefinementTrace


@dataclass(frozen=True, slots=True)
class RefinementResult:
    """The refined candidates together with an execution trace."""

    candidates: tuple[RefinedCandidate, ...]
    trace: RefinementTrace
