"""Optional Haystack integration, installed with ``ragrefine[haystack]``."""

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, cast

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.components.builders import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from ragrefine import Refiner
from ragrefine.models import Candidate, CandidateSet, RefinedCandidate

PROMPT_TEMPLATE = """Question: {{ query }}
Context:
{% for document in documents %}
[{{ document.id }}] {{ document.content }} | {{ document.meta }}
{% endfor %}"""
FROZEN_CORPUS = (
    {
        "id": "python-311",
        "content": "Python 3.11 adds exception groups.",
        "meta": {"topic": "python"},
    },
    {
        "id": "python-312",
        "content": "Python 3.12 improves the f-string parser.",
        "meta": {"topic": "python"},
    },
    {
        "id": "haystack",
        "content": "Haystack pipelines connect document retrieval components.",
        "meta": {"topic": "rag"},
    },
)
FROZEN_QUERIES = {
    "python": "Which Python version improves f-string parsing?",
    "rag": "How do Haystack pipelines retrieve documents?",
}


class HaystackDocumentAdapter:
    """Adapt one ordered Haystack result set while preserving object identity."""

    def __init__(
        self, documents: Sequence[Document], *, name: str = "haystack"
    ) -> None:
        if not name:
            raise ValueError("candidate-set name must be non-empty")
        self._documents = tuple(documents)
        self._by_id: dict[str, Document] = {}
        for document in self._documents:
            if not isinstance(document, Document):
                raise TypeError("documents must contain Haystack Document instances")
            identifier = str(document.id)
            if not identifier or identifier in self._by_id:
                raise ValueError(
                    f"invalid or duplicate Haystack Document ID: {identifier}"
                )
            self._by_id[identifier] = document
        self._name = name

    def candidate_set(self) -> CandidateSet:
        return CandidateSet(
            self._name,
            tuple(
                Candidate(
                    str(document.id),
                    document.content or "",
                    dict(document.meta),
                    document.score,
                    rank,
                )
                for rank, document in enumerate(self._documents, 1)
            ),
        )

    def documents_for(
        self, candidates: Iterable[RefinedCandidate | Candidate]
    ) -> tuple[Document, ...]:
        documents: list[Document] = []
        for item in candidates:
            candidate = item.candidate if isinstance(item, RefinedCandidate) else item
            if not isinstance(candidate, Candidate):
                raise TypeError("candidates must contain Candidate or RefinedCandidate")
            document = self._by_id.get(candidate.id)
            if document is None or document.content != candidate.text:
                raise ValueError(
                    f"candidate {candidate.id!r} was not created by this adapter"
                )
            documents.append(document)
        return tuple(documents)


@component
class RagRefineComponent:
    """Native Haystack component delegating final order/selection to Refiner."""

    def __init__(
        self,
        *,
        refiner: Refiner | None = None,
        candidate_set_name: str = "haystack",
        top_k: int = 5,
        max_tokens: int | None = None,
    ) -> None:
        if top_k < 0 or not candidate_set_name:
            raise ValueError(
                "top_k must be non-negative and candidate_set_name non-empty"
            )
        self._refiner, self._candidate_set_name = (
            refiner or Refiner(),
            candidate_set_name,
        )
        self._top_k, self._max_tokens = top_k, max_tokens

    @component.output_types(documents=list[Document], trace=dict[str, Any])
    def run(self, query: str, documents: list[Document]) -> dict[str, object]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        adapter = HaystackDocumentAdapter(documents, name=self._candidate_set_name)
        result = self._refiner.refine(
            query,
            adapter.candidate_set(),
            top_k=self._top_k,
            max_tokens=self._max_tokens,
        )
        return {
            "documents": list(adapter.documents_for(result.candidates)),
            "trace": asdict(result.trace),
        }

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            candidate_set_name=self._candidate_set_name,
            top_k=self._top_k,
            max_tokens=self._max_tokens,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RagRefineComponent":
        return default_from_dict(cls, data)


def prompt_builder() -> PromptBuilder:
    """Create the fixed prompt builder shared by bypass and refined paths."""
    return PromptBuilder(
        template=PROMPT_TEMPLATE, required_variables=["query", "documents"]
    )


@dataclass(frozen=True, slots=True)
class RetrievalIdentity:
    corpus_sha256: str
    queries_sha256: str
    retriever: str
    document_store: str
    top_n: int


class FrozenHaystackFixture:
    """Build a real deterministic Haystack BM25 fixture independent of refinement."""

    def __init__(self, *, top_n: int = 2) -> None:
        if top_n < 1:
            raise ValueError("top_n must be positive")
        self._top_n = top_n
        self._store = InMemoryDocumentStore(bm25_algorithm="BM25L")
        self._store.write_documents(
            [
                Document(
                    id=str(row["id"]),
                    content=str(row["content"]),
                    meta=dict(cast(Mapping[str, Any], row["meta"])),
                )
                for row in FROZEN_CORPUS
            ]
        )
        self._retriever = InMemoryBM25Retriever(document_store=self._store)

    def identity(self) -> RetrievalIdentity:
        return RetrievalIdentity(
            _checksum(FROZEN_CORPUS),
            _checksum(FROZEN_QUERIES),
            "haystack.InMemoryBM25Retriever/BM25L",
            "haystack.InMemoryDocumentStore",
            self._top_n,
        )

    def retrieve(self, query: str) -> tuple[Document, ...]:
        documents = self._retriever.run(query=query, top_k=self._top_n)["documents"]
        if any(document.score is None for document in documents):
            raise RuntimeError("Haystack retriever returned a document without a score")
        return tuple(documents)

    def frozen_query(self, name: str) -> str:
        return FROZEN_QUERIES[name]


def _checksum(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
