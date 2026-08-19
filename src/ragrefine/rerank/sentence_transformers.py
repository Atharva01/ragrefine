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
            from huggingface_hub import snapshot_download
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RerankerError(
                "install the optional extra: ragrefine[rerank]"
            ) from error
        try:
            model_path = snapshot_download(
                repo_id=self.model,
                revision=self.revision,
                allow_patterns=[
                    "config.json",
                    "model.safetensors",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.txt",
                ],
            )
            self._cross_encoder = CrossEncoder(
                model_path, device=self.device, local_files_only=True
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
        return self.rank_many(((query, candidates),))[0]

    def rank_many(
        self,
        queries: Sequence[tuple[str, Sequence[Candidate]]],
    ) -> tuple[tuple[ScoredCandidate, ...], ...]:
        """Score several query candidate pools in one backend inference batch.

        Each returned ranking corresponds to the input query at the same position.
        This is useful for GPU inference: a CrossEncoder sees enough pairs to fill
        its inference batches, while each query's candidates remain independently
        and deterministically ranked.
        """
        pairs: list[tuple[str, str]] = []
        offsets: list[tuple[int, int]] = []
        for query, candidates in queries:
            start = len(pairs)
            pairs.extend((query, candidate.text) for candidate in candidates)
            offsets.append((start, len(pairs)))
        if not pairs:
            return tuple(() for _ in queries)
        try:
            scores = self._backend().predict(
                pairs,
                batch_size=self.batch_size,
            )
        except RerankerError:
            raise
        except Exception as error:
            raise RerankerError("CrossEncoder inference failed") from error
        if len(scores) != len(pairs):
            raise RerankerError(
                "CrossEncoder returned a score count different from inputs"
            )
        return tuple(
            self._rank_candidates(candidates, scores[start:end])
            for (_, candidates), (start, end) in zip(queries, offsets, strict=True)
        )

    def _rank_candidates(
        self, candidates: Sequence[Candidate], scores: Sequence[float]
    ) -> tuple[ScoredCandidate, ...]:
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
