"""Pipeline-level tests for the optional native Haystack component."""

import pytest

haystack = pytest.importorskip("haystack")
Document = haystack.Document
Pipeline = haystack.Pipeline

from integrations.haystack.component import (  # noqa: E402
    RagRefineComponent,
    register_refiner_factory,
)
from ragrefine import Refiner, RefinerConfig  # noqa: E402
from ragrefine.config import ChannelConfig  # noqa: E402
from ragrefine.ranking.lexical import LexicalRanker  # noqa: E402


def _lexical_refiner() -> Refiner:
    return Refiner(
        lexical_ranker=LexicalRanker(),
        config=RefinerConfig(
            original=ChannelConfig(enabled=False),
            lexical=ChannelConfig(enabled=True),
        ),
    )


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


def test_b2_l_profile_reranks_and_serializes() -> None:
    component = RagRefineComponent(refiner=_lexical_refiner(), top_k=2)
    documents = [
        Document(id="unrelated", content="Python version syntax parser."),
        Document(id="somewhat", content="Haystack pipelines exist."),
        Document(id="relevant", content="Haystack pipeline retrieval ranking."),
    ]

    output = component.run("haystack pipeline", documents)["documents"]
    assert [document.id for document in output] == ["relevant", "somewhat"]

    restored = RagRefineComponent.from_dict(component.to_dict())
    assert restored.to_dict()["init_parameters"]["refiner_profile"] == "b2-l-lexical"
    assert restored.run("haystack pipeline", documents)["documents"] == output

    pipeline = Pipeline()
    pipeline.add_component("refine", component)
    loaded = Pipeline.loads(pipeline.dumps())
    assert (
        loaded.get_component("refine").to_dict()["init_parameters"]["refiner_profile"]
        == "b2-l-lexical"
    )


def test_custom_refiner_without_factory_is_rejected() -> None:
    component = RagRefineComponent(
        refiner=Refiner(
            config=RefinerConfig(pattern=ChannelConfig(enabled=True)),
        )
    )
    with pytest.raises(ValueError, match="registered refiner profile"):
        component.to_dict()


def test_registered_custom_factory_round_trips() -> None:
    name = "test-profile-pattern"

    def factory() -> Refiner:
        from ragrefine.query.patterns import PatternRegistry, example_pattern_rules
        from ragrefine.ranking.patterns import PatternRanker

        return Refiner(
            pattern_ranker=PatternRanker(PatternRegistry(example_pattern_rules())),
            config=RefinerConfig(
                original=ChannelConfig(enabled=False),
                pattern=ChannelConfig(enabled=True),
            ),
        )

    register_refiner_factory(name, factory)
    component = RagRefineComponent(refiner=factory(), top_k=1)

    restored = RagRefineComponent.from_dict(component.to_dict())
    assert restored.to_dict()["init_parameters"]["refiner_profile"] == name
    with pytest.raises(ValueError, match="already registered"):
        register_refiner_factory(name, factory)
