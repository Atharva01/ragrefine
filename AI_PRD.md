# ragrefine v0.1 product requirements

## Product boundary

`ragrefine` is a retriever-agnostic Python library for post-retrieval
refinement. It accepts one externally owned, immutable candidate pool and
returns selected candidates plus an auditable refinement trace. It does not
perform ingestion, indexing, retrieval, embedding generation, or generation.

## Implemented v0.1 scope

- immutable candidate contracts and a small `Refiner` API;
- independent original, neural, lexical, and pattern ranking channels;
- optional reciprocal-rank fusion of two or more successful channels;
- exact/near-duplicate suppression and greedy rank-preserving token-budget
  selection after final ranking;
- structured trace, channel failure handling, and deterministic tie-breaking;
- optional SentenceTransformers CrossEncoder integration.

Entity extraction, a generic query-analyzer layer, and relevance gates are
future hypotheses. They are not v0.1 runtime requirements and must not be
represented as implemented functionality.

## Evaluation and decisions

All post-retrieval profiles reuse frozen first-stage Top-50 pools. Measured
results retain B1-reference (MiniLM-L6/CUDA) and B2-L (stdlib lexical/CPU) as
qualified profiles. B2-P and evaluated B3 fusion profiles are not retained:
their recorded evidence does not justify a general improvement claim. The v1
structured hard-negative set is a schema regression fixture, not a profile
selection benchmark, because every relevant candidate begins at rank 1.

Machine-readable benchmark artifacts are the source of truth. See
`docs/evaluation-contract-v0.1.md` and `benchmarks/artifacts/README.md`.

## Source-of-truth order

1. This PRD defines product scope and decisions.
2. `TECHNICAL_HYPOTHESIS.md` defines testable, unproven claims.
3. `TECHNICAL_DESIGN.md` defines architecture and contracts.
4. Jira tickets define bounded implementation work and may refine the design.
5. `README.md` is a concise user-facing reference and must not override the
   documents above or machine-readable results.
