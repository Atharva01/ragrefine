"""Native optional Haystack component for post-retrieval refinement."""

from dataclasses import asdict
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from integrations.haystack.adapter import HaystackDocumentAdapter
from ragrefine import Refiner


@component
class RagRefineComponent:
    """Run a configured core refiner over one Haystack document result set.

    The component intentionally delegates all ranking and selection semantics to
    ``Refiner``. It returns the original document objects in the selected final
    order plus a JSON-compatible trace.
    """

    def __init__(
        self,
        *,
        refiner: Refiner | None = None,
        candidate_set_name: str = "haystack",
        top_k: int = 5,
        max_tokens: int | None = None,
    ) -> None:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not candidate_set_name:
            raise ValueError("candidate_set_name must be non-empty")
        self._refiner = refiner or Refiner()
        self._candidate_set_name = candidate_set_name
        self._top_k = top_k
        self._max_tokens = max_tokens

    @component.output_types(documents=list[Document], trace=dict[str, Any])
    def run(self, query: str, documents: list[Document]) -> dict[str, object]:
        """Refine retrieved documents, preserving final core ordering exactly."""
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
        """Serialize the component when its refiner uses the default profile."""
        if not self._refiner_is_default():
            raise ValueError(
                "serialization of a custom Refiner requires a registered factory"
            )
        return default_to_dict(
            self,
            candidate_set_name=self._candidate_set_name,
            top_k=self._top_k,
            max_tokens=self._max_tokens,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RagRefineComponent":
        """Recreate a serializable default-profile component."""
        return default_from_dict(cls, data)

    def _refiner_is_default(self) -> bool:
        config = self._refiner._config  # noqa: SLF001 - serialization boundary.
        return (
            config.original.enabled
            and not config.neural.enabled
            and not config.lexical.enabled
            and not config.pattern.enabled
            and not config.fusion_enabled
        )
