"""Optional Haystack integration for ragrefine."""

from integrations.haystack.adapter import HaystackDocumentAdapter
from integrations.haystack.component import RagRefineComponent
from integrations.haystack.fixture import FrozenHaystackFixture

__all__ = ["FrozenHaystackFixture", "HaystackDocumentAdapter", "RagRefineComponent"]
