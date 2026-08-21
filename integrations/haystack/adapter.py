"""Optional Haystack ``Document`` conversion without core-package coupling."""

from collections.abc import Iterable, Sequence

from haystack import Document
from ragrefine.models import Candidate, CandidateSet, RefinedCandidate


class HaystackDocumentAdapter:
    """Adapt one ordered Haystack result set while retaining document identity.

    The adapter owns no ranking logic. It converts documents to immutable core
    candidates, then maps selected/refined candidates back to the exact original
    ``Document`` objects supplied at construction.
    """

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
            document_id = str(document.id)
            if not document_id:
                raise ValueError("Haystack Document IDs must be non-empty")
            if document_id in self._by_id:
                raise ValueError(f"duplicate Haystack Document ID: {document_id}")
            self._by_id[document_id] = document
        self._name = name

    def candidate_set(self) -> CandidateSet:
        """Return the ordered, provenance-preserving core candidate set."""
        return CandidateSet(
            name=self._name,
            candidates=tuple(
                Candidate(
                    id=str(document.id),
                    text=document.content,
                    metadata=dict(document.meta),
                    retrieval_score=document.score,
                    retrieval_rank=index,
                )
                for index, document in enumerate(self._documents, start=1)
            ),
        )

    def documents_for(
        self, candidates: Iterable[RefinedCandidate | Candidate]
    ) -> tuple[Document, ...]:
        """Return the original documents for refined/selected core candidates."""
        resolved: list[Document] = []
        for item in candidates:
            candidate = item.candidate if isinstance(item, RefinedCandidate) else item
            if not isinstance(candidate, Candidate):
                raise TypeError("candidates must contain Candidate or RefinedCandidate")
            try:
                document = self._by_id[candidate.id]
            except KeyError as error:
                raise ValueError(
                    f"candidate {candidate.id!r} was not created by this adapter"
                ) from error
            if document.content != candidate.text:
                raise ValueError(
                    f"candidate {candidate.id!r} text does not match source"
                )
            resolved.append(document)
        return tuple(resolved)
