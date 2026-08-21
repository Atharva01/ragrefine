# ragrefine v0.1 technical hypotheses

## Controlled comparison rule

Candidate discovery and refinement are separate. Every B0/B1/B2/B3 comparison
uses the exact same frozen Top-50 candidates; no refinement stage can recover
documents absent from that pool. Candidate-pool Recall@50 is therefore a
ceiling, not an improvement metric.

## Hypotheses and current evidence

| Hypothesis | Current status |
| --- | --- |
| Neural reranking improves the retained SciFact/NFCorpus/FiQA configurations. | Measured for the exact retained B1-reference profile only. |
| Deterministic lexical ranking is useful across datasets. | Mixed: retained as a lightweight profile; FiQA regression prevents a general claim. |
| Pattern ranking improves profile selection. | Not established; B2-P is not retained. |
| Combining channels with equal-weight RRF improves over the strongest channel. | Not established; evaluated B3 profiles are not retained. |
| Deduplication and greedy token-budget selection improve context efficiency. | Implemented; effectiveness claims require separately scoped evidence. |

Entity extraction, query analysis, and relevance gating remain future
hypotheses. No result in v0.1 establishes them.

## Interpretation rules

Use nDCG@5, MRR@5, Precision@5, Recall@5, candidate-pool Recall@50, and
separate runtime evidence. Do not call a profile improved without its recorded
machine-readable artifact and the fixed configuration it represents.
