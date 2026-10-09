"""Tests for the optional LangChain document compressor."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("langchain_core")
from langchain_core.documents import Document  # noqa: E402

from ragrefine import Refiner, RefinerConfig  # noqa: E402
from ragrefine.config import ChannelConfig  # noqa: E402
from ragrefine.integrations.langchain import RagRefineCompressor  # noqa: E402
from ragrefine.ranking.lexical import LexicalRanker  # noqa: E402
from ragrefine.tracing.models import RefinementTrace  # noqa: E402


def _lexical_refiner() -> Refiner:
    return Refiner(
        lexical_ranker=LexicalRanker(),
        config=RefinerConfig(
            original=ChannelConfig(enabled=False),
            lexical=ChannelConfig(enabled=True),
        ),
    )


def _docs() -> list[Document]:
    return [
        Document(
            id="wrong",
            page_content="Upgrade guide for release 1.0 of the gateway.",
            metadata={"n": 1},
        ),
        Document(
            id="noise",
            page_content="Billing invoices are issued monthly.",
            metadata={"n": 2},
        ),
        Document(
            id="right",
            page_content="Upgrade guide for release 2.4.1 of the gateway.",
            metadata={"n": 3},
        ),
    ]


def _ids(docs: list[Document]) -> list[str | None]:
    return [d.id for d in docs]


def test_lexical_refiner_promotes_exact_version_and_b0_keeps_order() -> None:
    query = "upgrade gateway release 2.4.1"
    lexical = RagRefineCompressor(refiner=_lexical_refiner(), top_k=3)
    assert _ids(lexical.compress_documents(_docs(), query))[0] == "right"
    assert _ids(
        RagRefineCompressor(refiner=Refiner(), top_k=3).compress_documents(
            _docs(), query
        )
    ) == [
        "wrong",
        "noise",
        "right",
    ]


def test_returns_original_objects_without_mutation() -> None:
    docs = _docs()
    before = [(d.page_content, dict(d.metadata)) for d in docs]
    out = RagRefineCompressor(refiner=_lexical_refiner(), top_k=3).compress_documents(
        docs, "gateway 2.4.1"
    )
    assert all(any(o is d for d in docs) for o in out)
    assert [(d.page_content, dict(d.metadata)) for d in docs] == before


def test_missing_ids_are_synthesized_deterministically() -> None:
    def make() -> list[Document]:
        return [Document(page_content=f"text {i}") for i in range(3)]

    compressor = RagRefineCompressor(refiner=Refiner(), top_k=3)
    docs = make()
    first = compressor.compress_documents(docs, "q")
    assert first == docs
    assert compressor.compress_documents(make(), "q") == first


@pytest.mark.parametrize(
    "docs",
    [
        [Document(id="a", page_content="x"), Document(id="a", page_content="y")],
        [Document(page_content="x"), Document(id="lc:1", page_content="y")],
    ],
    ids=["explicit-duplicate", "synthesized-collision"],
)
def test_duplicate_ids_rejected(docs: list[Document]) -> None:
    with pytest.raises(ValueError, match="duplicate LangChain Document ID"):
        RagRefineCompressor(refiner=Refiner()).compress_documents(docs, "q")


@pytest.mark.parametrize("top_k", [0, 1, 2, 10])
def test_top_k_is_respected(top_k: int) -> None:
    out = RagRefineCompressor(
        refiner=_lexical_refiner(), top_k=top_k
    ).compress_documents(_docs(), "gateway")
    assert len(out) == min(top_k, 3)


def test_empty_input_returns_empty_and_trace() -> None:
    compressor = RagRefineCompressor(refiner=Refiner())
    assert compressor.compress_documents([], "q") == []
    docs, trace = compressor.refine("q", [])
    assert docs == [] and isinstance(trace, RefinementTrace)


def test_compress_documents_equals_refine_result() -> None:
    compressor = RagRefineCompressor(refiner=_lexical_refiner(), top_k=2)
    docs = _docs()
    refined, trace = compressor.refine("gateway 2.4.1", docs)
    assert isinstance(trace, RefinementTrace)
    assert compressor.compress_documents(docs, "gateway 2.4.1") == refined


def test_validation_and_type_errors() -> None:
    with pytest.raises(ValueError):
        RagRefineCompressor(refiner=Refiner(), top_k=-1)
    with pytest.raises(ValueError):
        RagRefineCompressor(refiner=Refiner(), candidate_set_name="")
    compressor = RagRefineCompressor(refiner=Refiner())
    with pytest.raises(TypeError):
        compressor.compress_documents([_docs()[0], "not a document"], "q")  # type: ignore[list-item]
    with pytest.raises(TypeError):
        compressor.compress_documents(_docs(), 123)  # type: ignore[arg-type]


@settings(max_examples=40, deadline=None)
@given(
    texts=st.lists(st.text(alphabet="abc xyz", max_size=20), max_size=8),
    top_k=st.integers(min_value=0, max_value=10),
    lexical=st.booleans(),
)
def test_output_is_bounded_unique_subset_of_input(
    texts: list[str], top_k: int, lexical: bool
) -> None:
    docs = [Document(page_content=t) for t in texts]
    refiner = _lexical_refiner() if lexical else Refiner()
    out = RagRefineCompressor(refiner=refiner, top_k=top_k).compress_documents(
        docs, "abc xyz"
    )
    assert len(out) == min(top_k, len(docs))
    assert len({id(d) for d in out}) == len(out)
    assert all(any(o is d for d in docs) for o in out)
