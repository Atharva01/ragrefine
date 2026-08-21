"""Refinement result contract."""

from dataclasses import dataclass

from ragrefine.models import RefinedCandidate
from ragrefine.selection.selector import SelectionRecord
from ragrefine.tracing.models import RefinementTrace


@dataclass(frozen=True, slots=True)
class RefinementResult:
    """Ranked candidates together with an execution trace.

    The result is prior to optional context selection. Each candidate's ``rank``
    is a ranking position, not a selected-context position.
    """

    candidates: tuple[RefinedCandidate, ...]
    trace: RefinementTrace
    selection: tuple[SelectionRecord, ...] = ()
