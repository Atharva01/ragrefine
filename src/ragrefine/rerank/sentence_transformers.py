"""Optional SentenceTransformers CrossEncoder adapter."""

from collections.abc import Sequence
from typing import Protocol

from ragrefine.errors import RerankerError
from ragrefine.models import Candidate
from ragrefine.rerank.base import ScoredCandidate


class _CrossEncoder(Protocol):
    def predict(
        self, sentences: Sequence[tuple[str, str]], *, batch_size: int
    ) -> Sequence[float]: ...


class SentenceTransformersReranker:
    """Rerank a supplied pool with an optional SentenceTransformers CrossEncoder."""

    name = "sentence-transformers-cross-encoder"

    def __init__(
        self,
        model: str,
        *,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        cross_encoder: _CrossEncoder | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model, self.revision, self.device = model, revision, device
        self.batch_size, self._cross_encoder = batch_size, cross_encoder

    def _backend(self) -> _CrossEncoder:
        if self._cross_encoder is not None:
            return self._cross_encoder
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RerankerError(
                "install the optional extra: ragrefine[rerank]"
            ) from error
        try:
            self._cross_encoder = CrossEncoder(
                self.model, revision=self.revision, device=self.device
            )
        except Exception as error:
            raise RerankerError(
                f"failed to load CrossEncoder '{self.model}'"
            ) from error
        return self._cross_encoder

    def rank(
        self, query: str, candidates: Sequence[Candidate]
    ) -> tuple[ScoredCandidate, ...]:
        """Score every supplied candidate in one batched inference call."""
        if not candidates:
            return ()
        try:
            scores = self._backend().predict(
                [(query, candidate.text) for candidate in candidates],
                batch_size=self.batch_size,
            )
        except RerankerError:
            raise
        except Exception as error:
            raise RerankerError("CrossEncoder inference failed") from error
        if len(scores) != len(candidates):
            raise RerankerError(
                "CrossEncoder returned a score count different from inputs"
            )
        ordered = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (
                -float(item[1]),
                item[0].retrieval_rank is None,
                item[0].retrieval_rank if item[0].retrieval_rank is not None else 0,
                item[0].id,
            ),
        )
        return tuple(
            ScoredCandidate(
                candidate=candidate,
                score=float(score),
                rank=index,
                model=self.model,
                model_revision=self.revision,
                backend="sentence-transformers",
            )
            for index, (candidate, score) in enumerate(ordered, start=1)
        )
