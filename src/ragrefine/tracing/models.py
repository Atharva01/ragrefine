"""Typed, dependency-free refinement tracing models."""

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class StageTrace:
    """Timing and configuration recorded for one refinement stage."""

    name: str
    duration_ms: float
    configuration: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RefinementTrace:
    """Structured trace data that applications may export to observability tools."""

    stages: tuple[StageTrace, ...]
    duration_ms: float
    config_fingerprint: str
