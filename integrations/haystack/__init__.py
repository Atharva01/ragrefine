"""Optional Haystack integration for ragrefine."""

from integrations.haystack.adapter import HaystackDocumentAdapter
from integrations.haystack.component import RagRefineComponent
from integrations.haystack.fixture import FrozenHaystackFixture
from integrations.haystack.prompt import PROMPT_TEMPLATE, prompt_builder

__all__ = [
    "FrozenHaystackFixture",
    "HaystackDocumentAdapter",
    "PROMPT_TEMPLATE",
    "RagRefineComponent",
    "prompt_builder",
]
