"""Pipeline-level tests for the optional native Haystack component."""

import pytest

haystack = pytest.importorskip("haystack")
Document = haystack.Document
Pipeline = haystack.Pipeline

from integrations.haystack.component import RagRefineComponent  # noqa: E402


def test_component_runs_in_pipeline_and_preserves_selected_order() -> None:
    component = RagRefineComponent(top_k=1)
    pipeline = Pipeline()
    pipeline.add_component("refine", component)

    result = pipeline.run(
        {
            "refine": {
                "query": "question",
                "documents": [
                    Document(id="first", content="First"),
                    Document(id="second", content="Second"),
                ],
            }
        }
    )

    output = result["refine"]
    assert [document.id for document in output["documents"]] == ["first"]
    assert output["trace"]["stages"]


def test_component_serializes_default_profile_and_surfaces_bad_input() -> None:
    component = RagRefineComponent(candidate_set_name="retrieved", top_k=2)

    restored = RagRefineComponent.from_dict(component.to_dict())

    assert restored.to_dict() == component.to_dict()
    with pytest.raises(TypeError, match="query must be a string"):
        component.run(query=object(), documents=[])
