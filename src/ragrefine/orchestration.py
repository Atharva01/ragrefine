"""Internal ranking-channel configuration and execution results."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from ragrefine.models import Candidate


@dataclass(frozen=True, slots=True)
class ChannelConfig:
    """Whether to run one channel, require success, and its fusion weight."""

    enabled: bool = False
    required: bool = True
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("channel weight must be non-negative")


@dataclass(frozen=True, slots=True)
class RefinerConfig:
    """Explicit independent channel configuration for B0 through B3 modes."""

    original: ChannelConfig = field(default_factory=lambda: ChannelConfig(enabled=True))
    neural: ChannelConfig = field(default_factory=ChannelConfig)
    lexical: ChannelConfig = field(default_factory=ChannelConfig)
    pattern: ChannelConfig = field(default_factory=ChannelConfig)
    fusion_enabled: bool = False


@dataclass(frozen=True, slots=True)
class _ChannelResult:
    """One successful channel execution over the unchanged input candidate pool."""

    name: str
    ranking: tuple[Candidate, ...]
    evidence: Mapping[str, Mapping[str, object]]
    metadata: Mapping[str, object] = field(default_factory=dict)
