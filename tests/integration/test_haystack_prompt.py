"""Integration tests connecting selected ragrefine documents to PromptBuilder."""

import pytest

haystack = pytest.importorskip("haystack")
Document = haystack.Document
Pipeline = haystack.Pipeline

from integrations.haystack.component import RagRefineComponent  # noqa: E402
from integrations.haystack.prompt import PROMPT_TEMPLATE, prompt_builder  # noqa: E402


def test_refined_pipeline_sends_only_selected_documents_to_fixed_prompt() -> None:
    query = "Which evidence?"
    documents = [
        Document(id="selected", content="Use this", meta={"source": "one"}),
        Document(id="excluded", content="Do not use this", meta={"source": "two"}),
    ]
    pipeline = Pipeline()
    pipeline.add_component("refine", RagRefineComponent(top_k=1))
    pipeline.add_component("prompt", prompt_builder())
    pipeline.connect("refine.documents", "prompt.documents")

    refined = pipeline.run(
        {"refine": {"query": query, "documents": documents}, "prompt": {"query": query}}
    )
    bypass = prompt_builder().run(query=query, documents=documents)

    refined_prompt = refined["prompt"]["prompt"]
    assert PROMPT_TEMPLATE
    assert "[selected] Use this" in refined_prompt
    assert "excluded" not in refined_prompt
    assert "[selected] Use this" in bypass["prompt"]
    assert "[excluded] Do not use this" in bypass["prompt"]
    assert refined["refine"]["trace"]["stages"]
