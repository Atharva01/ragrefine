"""Public configuration for deterministic refinement channels."""

from dataclasses import dataclass, field


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
