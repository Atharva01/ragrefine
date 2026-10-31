"""LangChain document compressor backed by a core ``Refiner``."""

from collections.abc import Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict, Field

from ragrefine import Refiner
from ragrefine.models import Candidate, CandidateSet
from ragrefine.tracing.models import RefinementTrace


class RagRefineCompressor(BaseDocumentCompressor):
    """Refine retrieved LangChain documents with a ragrefine ``Refiner``.

    Invariants: the output is a subset of the input; its order is exactly the
    refiner's final order; returned ``Document`` objects are the original input
    objects (never copies), and their ``page_content``/``metadata`` are never
    modified. All ranking semantics are delegated to the ``Refiner``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    refiner: Refiner
    top_k: int = Field(default=5, ge=0)
    max_tokens: int | None = None
    candidate_set_name: str = Field(default="langchain", min_length=1)

    def refine(
        self, query: str, documents: Sequence[Document]
    ) -> tuple[list[Document], RefinementTrace]:
        """Return the refined original documents and the refinement trace.

        Documents without an ``id`` get a synthesized ``lc:<1-based position>``
        candidate ID. Duplicate IDs raise ``ValueError``.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        by_id: dict[str, Document] = {}
        candidates: list[Candidate] = []
        for position, document in enumerate(documents, start=1):
            if not isinstance(document, Document):
                raise TypeError("documents must contain LangChain Document instances")
            candidate_id = document.id or f"lc:{position}"
            if candidate_id in by_id:
                raise ValueError(f"duplicate LangChain Document ID: {candidate_id}")
            by_id[candidate_id] = document
            candidates.append(
                Candidate(
                    id=candidate_id,
                    text=document.page_content,
                    metadata=dict(document.metadata),
                    retrieval_score=None,
                    retrieval_rank=position,
                )
            )
        result = self.refiner.refine(
            query,
            CandidateSet(name=self.candidate_set_name, candidates=tuple(candidates)),
            top_k=self.top_k,
            max_tokens=self.max_tokens,
        )
        return [by_id[item.candidate.id] for item in result.candidates], result.trace

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> list[Document]:
        """LangChain entry point; returns ``refine(...)[0]``.

        ``callbacks`` is accepted for interface compatibility and ignored.
        """
        return self.refine(query, documents)[0]
