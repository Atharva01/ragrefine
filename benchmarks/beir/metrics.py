"""Dependency-free metrics for evaluating a frozen candidate snapshot."""

import math
from collections.abc import Mapping, Sequence


def _average(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _dcg(relevance: Sequence[int]) -> float:
    return sum(value / math.log2(index + 2) for index, value in enumerate(relevance))


def evaluate_snapshot(
    snapshot: Mapping[str, object], *, k: int = 5
) -> dict[str, float]:
    """Evaluate the original snapshot ranking without applying refinement."""
    queries = snapshot["queries"]
    qrels = snapshot["qrels"]
    if not isinstance(queries, list) or not isinstance(qrels, dict):
        raise ValueError("snapshot must contain queries and qrels")

    ndcg: list[float] = []
    reciprocal_rank: list[float] = []
    precision: list[float] = []
    recall: list[float] = []
    pool_recall: list[float] = []
    for query in queries:
        if not isinstance(query, dict):
            raise ValueError("query must be an object")
        query_id = str(query["id"])
        candidates = query["candidates"]
        relevance_map = {
            str(key): int(value) for key, value in qrels.get(query_id, {}).items()
        }
        if not isinstance(candidates, list):
            raise ValueError("candidates must be a list")
        relevant_ids = {
            document_id for document_id, value in relevance_map.items() if value > 0
        }
        ranked_ids = [str(candidate["id"]) for candidate in candidates]
        top_ids = ranked_ids[:k]
        top_relevance = [relevance_map.get(document_id, 0) for document_id in top_ids]
        ideal_relevance = sorted(relevance_map.values(), reverse=True)[:k]
        ideal_dcg = _dcg(ideal_relevance)
        ndcg.append(_dcg(top_relevance) / ideal_dcg if ideal_dcg else 0.0)
        first_relevant = next(
            (
                index
                for index, document_id in enumerate(top_ids, start=1)
                if document_id in relevant_ids
            ),
            None,
        )
        reciprocal_rank.append(1 / first_relevant if first_relevant else 0.0)
        hits = len(set(top_ids) & relevant_ids)
        precision.append(hits / k)
        recall.append(hits / len(relevant_ids) if relevant_ids else 0.0)
        pool_recall.append(
            len(set(ranked_ids) & relevant_ids) / len(relevant_ids)
            if relevant_ids
            else 0.0
        )

    top_n = int(snapshot["retriever"]["top_n"])
    return {
        f"ndcg@{k}": _average(ndcg),
        f"mrr@{k}": _average(reciprocal_rank),
        f"precision@{k}": _average(precision),
        f"recall@{k}": _average(recall),
        f"candidate_pool_recall@{top_n}": _average(pool_recall),
    }
