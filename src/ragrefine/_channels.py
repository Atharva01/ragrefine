"""Private channel execution result used only by :mod:`ragrefine.refiner`."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from ragrefine.models import Candidate


@dataclass(frozen=True, slots=True)
class _ChannelResult:
    """One successful channel execution over the unchanged input candidate pool."""

    name: str
    ranking: tuple[Candidate, ...]
    evidence: Mapping[str, Mapping[str, object]]
    metadata: Mapping[str, object] = field(default_factory=dict)
