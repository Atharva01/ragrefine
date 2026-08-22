"""Runnable Haystack RAG pipeline wiring for the ragrefine integration.

``build_rag_pipeline`` wires the frozen BM25 retriever, ``RagRefineComponent``
with a registered refiner profile, the fixed prompt builder, and an optional
chat generator:

```text
InMemoryBM25Retriever -> RagRefineComponent -> PromptBuilder -> ChatPromptAdapter
                                                                  -> OpenAIChatGenerator
```

The generator is built for DeepSeek's OpenAI-compatible endpoint from
``DEEPSEEK_API_KEY``/``DEEPSEEK_MODEL`` when configured, or omitted entirely
(prompt-only pipeline). Refinement is post-retrieval only: it never changes
retrieval, indexing, or generation semantics.
"""

import os
from collections.abc import Mapping
from typing import Any, cast

from haystack import Document, Pipeline, component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.core.component.component import Component
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret

from ragrefine.integrations.haystack.component import (
    RagRefineComponent,
    refiner_for_profile,
)
from ragrefine.integrations.haystack.fixture import FROZEN_CORPUS
from ragrefine.integrations.haystack.prompt import prompt_builder

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


@component
class ChatPromptAdapter:
    """Turn a PromptBuilder ``prompt`` string into a chat ``messages`` list."""

    @component.output_types(messages=list[ChatMessage])
    def run(self, prompt: str) -> dict[str, Any]:
        """Wrap one prompt string as a user chat message."""
        return {"messages": [ChatMessage.from_user(prompt)]}


def demo_documents() -> list[Document]:
    """Return the frozen retrieval corpus as documents for an example store."""
    return [
        Document(
            id=str(row["id"]),
            content=str(row["content"]),
            meta=dict(cast(Mapping[str, Any], row["meta"])),
        )
        for row in FROZEN_CORPUS
    ]


def build_rag_pipeline(
    document_store: InMemoryDocumentStore,
    *,
    top_k: int = 3,
    profile: str = "b2-l-lexical",
    api_key: str | None = None,
    model: str | None = None,
    generator: Component | None = None,
) -> Pipeline:
    """Return a retriever -> refine -> prompt -> chat-generator Haystack pipeline.

    ``profile`` is any registered refiner profile (built-in ``b0`` and
    ``b2-l-lexical``). When ``generator`` is supplied it is used as the final
    chat generator; otherwise one is built for DeepSeek's OpenAI-compatible
    endpoint when ``api_key`` is available, and the pipeline stops after the
    prompt builder when it is not.
    """
    pipeline = Pipeline()
    pipeline.add_component(
        "retriever", InMemoryBM25Retriever(document_store=document_store)
    )
    pipeline.add_component(
        "refine",
        RagRefineComponent(top_k=top_k, refiner=refiner_for_profile(profile)),
    )
    pipeline.add_component("prompt", prompt_builder())
    pipeline.connect("retriever.documents", "refine.documents")
    pipeline.connect("refine.documents", "prompt.documents")
    if generator is None:
        if not api_key:
            return pipeline
        generator = OpenAIChatGenerator(
            api_key=Secret.from_token(api_key),
            model=model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            api_base_url=DEEPSEEK_BASE_URL,
            generation_kwargs={"temperature": 0},
            max_retries=0,
        )
    pipeline.add_component("chat", ChatPromptAdapter())
    pipeline.add_component("generator", generator)
    pipeline.connect("prompt.prompt", "chat.prompt")
    pipeline.connect("chat.messages", "generator.messages")
    return pipeline
