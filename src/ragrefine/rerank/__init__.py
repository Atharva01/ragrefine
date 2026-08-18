"""Neural reranking contracts and optional adapters."""

from ragrefine.rerank.base import NeuralReranker, ScoredCandidate
from ragrefine.rerank.sentence_transformers import SentenceTransformersReranker

__all__ = ["NeuralReranker", "ScoredCandidate", "SentenceTransformersReranker"]
