"""Frozen-pool Haystack bypass versus retained lexical-profile evaluation."""

import json
from dataclasses import asdict
from pathlib import Path

from integrations.haystack.adapter import HaystackDocumentAdapter
from integrations.haystack.fixture import FrozenHaystackFixture
from ragrefine import Refiner, RefinerConfig
from ragrefine.config import ChannelConfig
from ragrefine.ranking.lexical import LexicalRanker

RELEVANT_IDS = {"python": "python-312", "rag": "haystack"}


def run(output_dir: Path) -> dict[str, object]:
    """Persist a B0 bypass/B2-L comparison using identical Haystack pools."""
    if output_dir.exists():
        raise FileExistsError(f"Haystack A/B output already exists: {output_dir}")
    fixture = FrozenHaystackFixture(top_n=2)
    lexical = Refiner(
        lexical_ranker=LexicalRanker(),
        config=RefinerConfig(
            original=ChannelConfig(enabled=False),
            lexical=ChannelConfig(enabled=True),
        ),
    )
    rows: list[dict[str, object]] = []
    for name, relevant_id in RELEVANT_IDS.items():
        query = fixture.frozen_query(name)
        pool = fixture.retrieve(query)
        candidate_set = HaystackDocumentAdapter(pool).candidate_set()
        refined = lexical.refine(query, candidate_set, top_k=len(pool))
        baseline_ids = [str(document.id) for document in pool]
        refined_ids = [candidate.candidate.id for candidate in refined.candidates]
        rows.append(
            {
                "query_name": name,
                "query": query,
                "relevant_id": relevant_id,
                "input_pool": [
                    {"id": str(document.id), "score": document.score}
                    for document in pool
                ],
                "baseline_ids": baseline_ids,
                "refined_ids": refined_ids,
                "trace": {"stages": [stage.name for stage in refined.trace.stages]},
                "context": {
                    "baseline_candidate_count": len(baseline_ids),
                    "refined_candidate_count": len(refined_ids),
                    "candidate_reduction": len(baseline_ids) - len(refined_ids),
                    "baseline_word_tokens": _word_tokens(pool),
                    "refined_word_tokens": _word_tokens(
                        HaystackDocumentAdapter(pool).documents_for(refined.candidates)
                    ),
                },
            }
        )
    result: dict[str, object] = {
        "schema_version": "1.0",
        "fixture_identity": asdict(fixture.identity()),
        "profile": "B2-L lexical",
        "pool_reused": True,
        "per_query": rows,
        "metrics": {
            "baseline": _metrics(rows, "baseline_ids"),
            "refined": _metrics(rows, "refined_ids"),
        },
        "context_reduction": _context_reduction(rows),
    }
    output_dir.mkdir(parents=True)
    (output_dir / "ab-results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _metrics(rows: list[dict[str, object]], key: str) -> dict[str, float]:
    ranks = [list(row[key]).index(str(row["relevant_id"])) + 1 for row in rows]
    return {
        "mrr": sum(1 / rank for rank in ranks) / len(ranks),
        "precision@1": sum(rank == 1 for rank in ranks) / len(ranks),
    }


def _word_tokens(documents: object) -> int:
    return sum(len(str(document.content).split()) for document in documents)


def _context_reduction(rows: list[dict[str, object]]) -> dict[str, int]:
    contexts = [row["context"] for row in rows]
    return {
        "baseline_candidates": sum(
            context["baseline_candidate_count"] for context in contexts
        ),
        "refined_candidates": sum(
            context["refined_candidate_count"] for context in contexts
        ),
        "candidate_reduction": sum(
            context["candidate_reduction"] for context in contexts
        ),
        "baseline_word_tokens": sum(
            context["baseline_word_tokens"] for context in contexts
        ),
        "refined_word_tokens": sum(
            context["refined_word_tokens"] for context in contexts
        ),
    }
