"""The compressor inside real LangChain retrieval chains.

``ContextualCompressionRetriever`` (``langchain-classic``) calls
``compress_documents``/``acompress_documents`` with a ``callbacks=`` keyword; the
LCEL case is the ``RunnableLambda`` recipe documented in the README.
"""

import asyncio

import pytest

pytest.importorskip("langchain_core")
from langchain_core.callbacks import CallbackManagerForRetrieverRun  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.retrievers import BaseRetriever  # noqa: E402
from langchain_core.runnables import (  # noqa: E402
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from ragrefine import Refiner, RefinerConfig  # noqa: E402
from ragrefine.config import ChannelConfig  # noqa: E402
from ragrefine.integrations.langchain import RagRefineCompressor  # noqa: E402
from ragrefine.ranking.lexical import LexicalRanker  # noqa: E402

QUERY = "upgrade guide release 2.4.1 gateway"


class _FixedRetriever(BaseRetriever):
    """First-stage retriever stand-in returning a frozen candidate pool."""

    docs: list[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return list(self.docs)


def _retriever() -> _FixedRetriever:
    # Hard negative first: right topic, wrong version ranks above the answer.
    return _FixedRetriever(
        docs=[
            Document(
                id="wrong",
                page_content="Upgrade guide for release 1.0 of the gateway.",
                metadata={"n": 1},
            ),
            Document(id="noise", page_content="Billing invoices are issued monthly."),
            Document(
                id="right",
                page_content="Upgrade guide for release 2.4.1 of the gateway.",
                metadata={"n": 3},
            ),
        ]
    )


def _compressor(top_k: int = 2) -> RagRefineCompressor:
    refiner = Refiner(
        lexical_ranker=LexicalRanker(),
        config=RefinerConfig(
            original=ChannelConfig(enabled=False),
            lexical=ChannelConfig(enabled=True),
        ),
    )
    return RagRefineCompressor(refiner=refiner, top_k=top_k)


def _assert_refined(result: list[Document], retriever: _FixedRetriever) -> None:
    assert [d.id for d in result][0] == "right"
    assert len(result) == 2
    # Same objects the retriever produced: never copied, never edited.
    assert all(any(d is source for source in retriever.docs) for d in result)
    assert result[0].metadata == {"n": 3}


def test_contextual_compression_retriever_sync_and_async() -> None:
    classic = pytest.importorskip("langchain_classic.retrievers")
    retriever = _retriever()
    chain = classic.ContextualCompressionRetriever(
        base_compressor=_compressor(), base_retriever=retriever
    )

    _assert_refined(chain.invoke(QUERY), retriever)
    _assert_refined(asyncio.run(chain.ainvoke(QUERY)), retriever)


def test_lcel_runnable_lambda_chain_matches_readme_recipe() -> None:
    retriever = _retriever()
    compressor = _compressor()
    chain = RunnableParallel(
        query=RunnablePassthrough(), documents=retriever
    ) | RunnableLambda(
        lambda x: compressor.compress_documents(x["documents"], x["query"])
    )

    _assert_refined(list(chain.invoke(QUERY)), retriever)
