"""Native optional Haystack component for post-retrieval refinement."""

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from haystack import Document, component, default_to_dict

from ragrefine import Refiner, RefinerConfig
from ragrefine.config import ChannelConfig
from ragrefine.integrations.haystack.adapter import HaystackDocumentAdapter
from ragrefine.ranking.lexical import LexicalRanker

_REFINER_FACTORIES: dict[str, Callable[[], Refiner]] = {}


def register_refiner_factory(name: str, factory: Callable[[], Refiner]) -> None:
    """Register a named refiner profile for serializable components.

    Built-in profiles are ``b0`` (original retrieval order) and
    ``b2-l-lexical`` (the retained lightweight lexical profile). A custom
    profile must be registered before a component using it is serialized, so
    Haystack pipeline YAML can rebuild the exact refiner on load.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("refiner profile name must be a non-empty string")
    if not callable(factory):
        raise TypeError("refiner factory must be callable")
    if name in _REFINER_FACTORIES:
        raise ValueError(f"refiner profile already registered: {name!r}")
    _REFINER_FACTORIES[name] = factory


def refiner_for_profile(name: str) -> Refiner:
    """Build the refiner for one registered profile name."""
    try:
        return _REFINER_FACTORIES[name]()
    except KeyError as error:
        raise ValueError(
            f"unknown refiner profile {name!r}; "
            f"registered: {sorted(_REFINER_FACTORIES)}"
        ) from error


def _b0_refiner() -> Refiner:
    """Original-order (B0) profile: refinement leaves the pool order unchanged."""
    return Refiner()


def _b2_l_lexical_refiner() -> Refiner:
    """Retained B2-L profile: stdlib lexical-coverage ranking only."""
    return Refiner(
        lexical_ranker=LexicalRanker(),
        config=RefinerConfig(
            original=ChannelConfig(enabled=False),
            lexical=ChannelConfig(enabled=True),
        ),
    )


register_refiner_factory("b0", _b0_refiner)
register_refiner_factory("b2-l-lexical", _b2_l_lexical_refiner)


@component
class RagRefineComponent:
    """Run a configured core refiner over one Haystack document result set.

    The component intentionally delegates all ranking and selection semantics to
    ``Refiner``. It returns the original document objects in the selected final
    order plus a JSON-compatible trace. Built-in serializable profiles are
    ``b0`` and ``b2-l-lexical``; custom refiners need a registered factory.
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
        """Serialize the component with its named refiner profile."""
        return default_to_dict(
            self,
            candidate_set_name=self._candidate_set_name,
            top_k=self._top_k,
            max_tokens=self._max_tokens,
            refiner_profile=self._profile_name(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RagRefineComponent":
        """Recreate a component from its serialized profile and settings."""
        init_parameters = dict(data.get("init_parameters", {}))
        profile = init_parameters.pop("refiner_profile", "b0")
        return cls(refiner=refiner_for_profile(profile), **init_parameters)

    def _profile_name(self) -> str:
        for name, factory in _REFINER_FACTORIES.items():
            try:
                candidate = factory()
            except Exception:
                continue
            if self._equivalent(candidate):
                return name
        raise ValueError(
            "serialization requires a registered refiner profile; register one "
            "with register_refiner_factory(...) "
            f"(registered: {sorted(_REFINER_FACTORIES)})"
        )

    def _equivalent(self, other: "Refiner") -> bool:
        # SLF001 - serialization boundary; compares the exact ranking wiring.
        return (
            self._refiner._config == other._config
            and type(self._refiner._reranker) is type(other._reranker)
            and type(self._refiner._lexical_ranker) is type(other._lexical_ranker)
            and type(self._refiner._pattern_ranker) is type(other._pattern_ranker)
            and type(self._refiner._fusion) is type(other._fusion)
            and type(self._refiner._deduplicator) is type(other._deduplicator)
            and type(self._refiner._selector) is type(other._selector)
        )
