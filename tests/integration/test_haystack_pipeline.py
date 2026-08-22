"""End-to-end Haystack RAG pipeline tests using an in-memory fake generator."""

import pytest

haystack = pytest.importorskip("haystack")
Document = haystack.Document
Pipeline = haystack.Pipeline
component = haystack.component

from haystack.dataclasses import ChatMessage  # noqa: E402
from haystack.document_stores.in_memory import InMemoryDocumentStore  # noqa: E402

from integrations.haystack.pipeline import (  # noqa: E402
    build_rag_pipeline,
    demo_documents,
)

QUERY = "Which Python version improves f-string parsing?"


@component
class _FakeChatGenerator:
    """Records the received prompt and returns a fixed fixture reply."""

    def __init__(self) -> None:
        self.received: list[str] = []

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]) -> dict[str, list[ChatMessage]]:
        self.received.append(messages[0].text)
        return {"replies": [ChatMessage.from_assistant("fixture answer")]}


def _store() -> InMemoryDocumentStore:
    store = InMemoryDocumentStore(bm25_algorithm="BM25L")
    store.write_documents(demo_documents())
    return store


def _run_kwargs(query: str = QUERY) -> dict[str, dict[str, str]]:
    return {
        "retriever": {"query": query},
        "refine": {"query": query},
        "prompt": {"query": query},
    }


def _run(pipeline: Pipeline, query: str = QUERY) -> dict:
    # Connected outputs (refine.documents, prompt.prompt) are excluded from
    # results by default, so keep them for assertions.
    return pipeline.run(
        _run_kwargs(query),
        include_outputs_from={"retriever", "refine", "prompt"},
    )


def test_end_to_end_pipeline_retrieves_refines_prompts_and_generates() -> None:
    generator = _FakeChatGenerator()
    pipeline = build_rag_pipeline(_store(), top_k=2, generator=generator)

    result = _run(pipeline)

    assert result["generator"]["replies"][0].text == "fixture answer"
    assert generator.received
    assert QUERY in generator.received[0]
    refined = result["refine"]["documents"]
    assert [str(document.id) for document in refined] == ["python-312", "python-311"]
    assert result["refine"]["trace"]["stages"]
    prompt = result["prompt"]["prompt"]
    assert "[python-312]" in prompt
    assert "[haystack]" not in prompt


def test_pipeline_uses_registered_profile_and_round_trips_through_yaml() -> None:
    pipeline = build_rag_pipeline(_store(), top_k=2, profile="b2-l-lexical")

    restored = Pipeline.loads(pipeline.dumps())
    restored_refine = restored.get_component("refine")
    assert (
        restored_refine.to_dict()["init_parameters"]["refiner_profile"]
        == "b2-l-lexical"
    )

    result = _run(restored)
    refined = result["refine"]["documents"]
    assert [str(document.id) for document in refined] == ["python-312", "python-311"]


def test_pipeline_without_generator_stops_at_prompt() -> None:
    pipeline = build_rag_pipeline(_store(), top_k=2)
    assert "generator" not in pipeline.graph.nodes
    assert "chat" not in pipeline.graph.nodes

    result = _run(pipeline)
    assert "prompt" in result
    assert QUERY in result["prompt"]["prompt"]
    assert "generator" not in result
