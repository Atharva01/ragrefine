"""Deterministic redundancy measurements for frozen candidate snapshots."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import mean

from ragrefine.filtering.deduplicate import content_hash, normalized_content

NEAR_DUPLICATE_THRESHOLDS = (0.80, 0.90, 0.95)
SHINGLE_SIZE = 5
WORD_TOKENIZATION = "unicode_word_regex_v1"


@dataclass(frozen=True, slots=True)
class _PoolStats:
    candidates: int
    word_tokens: int
    duplicate_clusters: int
    duplicate_candidates: int
    exact_duplicate_candidates: int
    near_duplicate_candidates: int
    redundant_word_tokens: int


@dataclass(frozen=True, slots=True)
class _PoolCandidate:
    candidate_id: str
    text: str
    rank: int
    digest: str
    shingles: frozenset[tuple[str, ...]]
    word_tokens: int


def word_token_count(text: str) -> int:
    """Return a deterministic word-token proxy for context-size characterization.

    This deliberately is not an LLM tokenizer. The core package keeps token
    counting model-specific; the benchmark uses this proxy solely to compare
    frozen pools under one stable, dependency-free definition.
    """
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def analyze_snapshot(
    snapshot: Mapping[str, object],
    *,
    thresholds: Sequence[float] = NEAR_DUPLICATE_THRESHOLDS,
) -> dict[str, object]:
    """Measure candidate-pool redundancy without inspecting relevance outcomes."""
    queries = snapshot.get("queries")
    if not isinstance(queries, list):
        raise ValueError("snapshot must contain a queries list")
    if not thresholds:
        raise ValueError("at least one near-duplicate threshold is required")

    pools = [_candidates_for_query(query) for query in queries]
    exact = [_measure_pool(pool, near_duplicate_threshold=1.0) for pool in pools]
    by_threshold = {
        threshold: [
            _measure_pool(pool, near_duplicate_threshold=threshold) for pool in pools
        ]
        for threshold in thresholds
    }
    dataset = snapshot.get("dataset")
    dataset_name = (
        str(dataset.get("name", "unknown"))
        if isinstance(dataset, Mapping)
        else "unknown"
    )
    return {
        "dataset": dataset_name,
        "query_count": len(pools),
        "measurement": {
            "word_tokenization": WORD_TOKENIZATION,
            "shingle_size": SHINGLE_SIZE,
            "near_duplicate_thresholds": list(thresholds),
            "relevance_labels_used": False,
        },
        "candidate_pool": _summarize(exact),
        "exact_normalized_duplicates": _summarize_duplicates(exact),
        "near_duplicate_thresholds": {
            f"{threshold:.2f}": _summarize_duplicates(stats)
            for threshold, stats in by_threshold.items()
        },
    }


def _candidates_for_query(query: object) -> tuple[_PoolCandidate, ...]:
    if not isinstance(query, Mapping):
        raise ValueError("snapshot query must be an object")
    candidates = query.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("snapshot query must contain a candidates list")

    ranked: list[_PoolCandidate] = []
    for raw_candidate in candidates:
        if not isinstance(raw_candidate, Mapping):
            raise ValueError("snapshot candidate must be an object")
        rank = int(raw_candidate["rank"])
        text = str(raw_candidate["text"])
        ranked.append(
            _PoolCandidate(
                candidate_id=str(raw_candidate["id"]),
                text=text,
                rank=rank,
                digest=content_hash(text),
                shingles=_shingles(text),
                word_tokens=word_token_count(text),
            )
        )
    return tuple(sorted(ranked, key=lambda item: (item.rank, item.candidate_id)))


def _measure_pool(
    candidates: Sequence[_PoolCandidate], *, near_duplicate_threshold: float
) -> _PoolStats:
    retained: list[_PoolCandidate] = []
    exact: dict[str, _PoolCandidate] = {}
    suppressed: list[tuple[_PoolCandidate, str, str]] = []
    for candidate in candidates:
        exact_match = exact.get(candidate.digest)
        if exact_match is not None:
            suppressed.append(
                (candidate, "exact_normalized_content", exact_match.candidate_id)
            )
            continue
        near_match = next(
            (
                retained_candidate
                for retained_candidate in retained
                if _jaccard(candidate.shingles, retained_candidate.shingles)
                > near_duplicate_threshold
            ),
            None,
        )
        if near_match is not None:
            suppressed.append(
                (
                    candidate,
                    "near_token_shingle_jaccard",
                    near_match.candidate_id,
                )
            )
            continue
        exact[candidate.digest] = candidate
        retained.append(candidate)
    return _PoolStats(
        candidates=len(candidates),
        word_tokens=sum(item.word_tokens for item in candidates),
        duplicate_clusters=len({retained_id for _, _, retained_id in suppressed}),
        duplicate_candidates=len(suppressed),
        exact_duplicate_candidates=sum(
            reason == "exact_normalized_content" for _, reason, _ in suppressed
        ),
        near_duplicate_candidates=sum(
            reason == "near_token_shingle_jaccard" for _, reason, _ in suppressed
        ),
        redundant_word_tokens=sum(item.word_tokens for item, _, _ in suppressed),
    )


def _summarize(stats: Sequence[_PoolStats]) -> dict[str, object]:
    total_candidates = sum(item.candidates for item in stats)
    total_tokens = sum(item.word_tokens for item in stats)
    return {
        "candidates": total_candidates,
        "candidates_per_query": _per_query(item.candidates for item in stats),
        "word_tokens": total_tokens,
        "word_tokens_per_query": _per_query(item.word_tokens for item in stats),
    }


def _summarize_duplicates(stats: Sequence[_PoolStats]) -> dict[str, object]:
    total_candidates = sum(item.candidates for item in stats)
    total_tokens = sum(item.word_tokens for item in stats)
    duplicate_candidates = sum(item.duplicate_candidates for item in stats)
    redundant_tokens = sum(item.redundant_word_tokens for item in stats)
    return {
        "duplicate_clusters": sum(item.duplicate_clusters for item in stats),
        "duplicate_clusters_per_query": _per_query(
            item.duplicate_clusters for item in stats
        ),
        "duplicate_candidates": duplicate_candidates,
        "duplicate_candidate_share": _share(duplicate_candidates, total_candidates),
        "exact_duplicate_candidates": sum(
            item.exact_duplicate_candidates for item in stats
        ),
        "near_duplicate_candidates": sum(
            item.near_duplicate_candidates for item in stats
        ),
        "redundant_word_tokens": redundant_tokens,
        "redundant_word_token_share": _share(redundant_tokens, total_tokens),
    }


def _per_query(values: Sequence[int]) -> dict[str, float | int]:
    materialized = list(values)
    return {
        "mean": mean(materialized) if materialized else 0.0,
        "min": min(materialized) if materialized else 0,
        "max": max(materialized) if materialized else 0,
    }


def _share(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _shingles(text: str) -> frozenset[tuple[str, ...]]:
    tokens = tuple(re.findall(r"\w+", normalized_content(text)))
    if not tokens:
        return frozenset()
    width = min(SHINGLE_SIZE, len(tokens))
    return frozenset(
        tokens[index : index + width] for index in range(len(tokens) - width + 1)
    )


def _jaccard(
    left: frozenset[tuple[str, ...]], right: frozenset[tuple[str, ...]]
) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
