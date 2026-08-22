"""Frozen, reproducible Haystack retrieval fixture independent of ragrefine."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

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


@dataclass(frozen=True, slots=True)
class RetrievalIdentity:
    """Persistable identity for the frozen retrieval corpus and configuration."""

    corpus_sha256: str
    queries_sha256: str
    retriever: str
    document_store: str
    top_n: int


class FrozenHaystackFixture:
    """Build one real deterministic BM25 document-store retrieval fixture."""

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
        """Return frozen corpus/query/retriever/index identity for persistence."""
        return RetrievalIdentity(
            corpus_sha256=_checksum(FROZEN_CORPUS),
            queries_sha256=_checksum(FROZEN_QUERIES),
            retriever="haystack.InMemoryBM25Retriever/BM25L",
            document_store="haystack.InMemoryDocumentStore",
            top_n=self._top_n,
        )

    def retrieve(self, query: str) -> tuple[Document, ...]:
        """Return the single frozen pool to reuse for every comparison branch."""
        if not isinstance(query, str) or not query:
            raise ValueError("query must be a non-empty string")
        documents = self._retriever.run(query=query, top_k=self._top_n)["documents"]
        if any(document.score is None for document in documents):
            raise RuntimeError("Haystack retriever returned a document without a score")
        return tuple(documents)

    def frozen_query(self, name: str) -> str:
        """Return one named immutable fixture query."""
        try:
            return FROZEN_QUERIES[name]
        except KeyError as error:
            raise ValueError(f"unknown frozen query: {name}") from error


def _checksum(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
