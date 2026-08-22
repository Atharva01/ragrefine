"""Optional Haystack integration for ragrefine.

Installed with ``ragrefine[haystack]``. Provides the document adapter,
``RagRefineComponent`` with serializable refiner profiles, the prompt builder,
a frozen retrieval fixture, and a ready-made RAG pipeline builder.
"""

from ragrefine.integrations.haystack.adapter import HaystackDocumentAdapter
from ragrefine.integrations.haystack.component import (
    RagRefineComponent,
    refiner_for_profile,
    register_refiner_factory,
)
from ragrefine.integrations.haystack.fixture import FrozenHaystackFixture
from ragrefine.integrations.haystack.pipeline import (
    ChatPromptAdapter,
    build_rag_pipeline,
    demo_documents,
)
from ragrefine.integrations.haystack.prompt import PROMPT_TEMPLATE, prompt_builder

__all__ = [
    "ChatPromptAdapter",
    "FrozenHaystackFixture",
    "HaystackDocumentAdapter",
    "PROMPT_TEMPLATE",
    "RagRefineComponent",
    "build_rag_pipeline",
    "demo_documents",
    "prompt_builder",
    "refiner_for_profile",
    "register_refiner_factory",
]
