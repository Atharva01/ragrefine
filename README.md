# ragrefine

**Post-retrieval refinement for RAG systems.**

`ragrefine` is an experimental Python library for improving the quality of retrieved context **after retrieval and before generation**.

Instead of replacing your retriever, vector database, or RAG framework, `ragrefine` takes an existing candidate set and applies additional relevance signals, reranking, fusion, and context-selection logic.

> **Status:** Early development. The package API and benchmark results are not yet stable. No retrieval-quality improvement is claimed until the planned evaluations are complete.

---

## Why ragrefine?

Dense retrieval is good at finding semantically related text, but *semantic similarity is not the same as context utility*.

A retrieved chunk can be topically similar while still being wrong for the actual information need—for example:

- the wrong entity,
- the wrong software/model version,
- the wrong date,
- a related but incorrect error code,
- duplicated evidence,
- or a chunk that is relevant but too incomplete to support an answer.

`ragrefine` explores whether post-retrieval signals can improve that final context set without rebuilding the retrieval stack.

---

## Where it fits

```text
                     existing retrieval system
                              │
Query ──► dense / sparse / hybrid retriever
                              │
                         Top-N candidates
                              │
                              ▼
                      ┌───────────────┐
                      │   ragrefine   │
                      │               │
                      │ query signals │
                      │ reranking     │
                      │ rank fusion   │
                      │ deduplication │
                      │ selection     │
                      └───────┬───────┘
                              │
                         Top-K context
                              │
                              ▼
                             LLM
```

The core boundary is intentionally small:

```text
Query + Retrieved Candidates
            ↓
        ragrefine
            ↓
Refined Candidates + Refinement Trace
```

---

## Planned refinement pipeline

| Stage | Purpose |
|---|---|
| Query normalization | Produce deterministic normalized query features |
| Lexical signals | Reward explicit term agreement |
| Pattern matching | Capture structured values such as versions, IDs, dates, codes, and measurements |
| Entity signals | Detect entity agreement and conflicts |
| Neural reranking | Re-evaluate a small candidate pool with a CrossEncoder |
| Rank fusion | Combine heterogeneous rankings without assuming raw score comparability |
| Deduplication | Remove exact and near-duplicate evidence |
| Context selection | Select the highest-value context under Top-K/token constraints |
| Tracing | Explain which stages affected each candidate |

The design avoids directly summing unrelated raw scores such as embedding similarity, BM25 scores, entity signals, and CrossEncoder logits. Ranking channels are intended to be combined through rank-based fusion.

---

## Research question

The project tests the hypothesis that:

> **Semantic similarity alone is not always sufficient to identify the most useful context for a query.**

The goal is not to prove this assumption by implementation. Each refinement stage must earn its place through controlled evaluation.

---

## Evaluation

Experiments are designed around a **frozen candidate pool** so that refinement methods are compared against exactly the same first-stage retrieval results.

Planned ablations:

```text
B0  Dense retrieval baseline
B1  + Neural reranking
B2  + Lexical / pattern / entity signals
B3  + Multi-signal rank fusion
B4  + Deduplication / context selection
```

Primary retrieval metrics:

- nDCG@5
- MRR
- Precision@5
- Recall@5
- candidate-pool Recall@N

Operational measurements include refinement latency and candidate throughput.

The initial evaluation harness is planned around BEIR datasets plus a small hard-negative set targeting cases such as wrong entities, versions, dates, and technical identifiers.

---

## Scope

`ragrefine` is **not** intended to become another end-to-end RAG framework.

It does not own:

- document ingestion or parsing,
- chunking,
- embedding generation,
- vector databases,
- first-stage retrieval,
- LLM generation,
- agent orchestration,
- conversation memory,
- cloud infrastructure.

Its responsibility is deliberately constrained to:

```text
candidate refinement + context selection + traceability
```

---

## Design principles

- **Retriever agnostic** — consume candidates from an existing retrieval system.
- **Framework agnostic** — keep the core independent of LangChain, Haystack, vector databases, and cloud SDKs.
- **Evidence preserving** — never invent or rewrite source evidence during ranking.
- **Deterministic where possible** — stable ordering and explicit tie-breaking.
- **Traceable** — expose ranking decisions and stage timings.
- **Low-dependency core** — ML/NLP integrations remain optional.
- **Evaluation first** — implementation completion is not evidence of retrieval improvement.

---

## Development

Python **3.12+** and [`uv`](https://docs.astral.sh/uv/) are used for local development.

Sync the project environment:

```bash
uv sync
```

Run the quality gates:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/
uv run pytest
```

Build the package:

```bash
uv build
```

Heavy model and benchmark dependencies are intentionally kept outside the core development path.

---

## Repository layout

```text
src/ragrefine/       core library
tests/unit/          deterministic unit tests
tests/property/      invariant/property tests
tests/integration/   optional integration tests
benchmarks/beir/     retrieval evaluation harness
benchmarks/results/  benchmark artifacts
docs/                product, hypothesis, and technical design
```

---

## Roadmap

- [x] Package scaffold and engineering quality gates
- [ ] Core candidate domain contracts
- [ ] No-op refinement contract and tracing
- [ ] Frozen B0 retrieval baseline
- [ ] CrossEncoder reranking
- [ ] Lexical, regex/pattern, and entity signals
- [ ] Reciprocal Rank Fusion
- [ ] Deduplication and context budgeting
- [ ] Hard-negative evaluation
- [ ] Haystack integration
- [ ] ONNX inference experiment

---

## Project philosophy

`ragrefine` treats retrieval refinement as an **information-retrieval problem first** and an LLM integration problem second.

The project will prefer measurable ranking improvements, reproducible experiments, and interpretable failure analysis over adding more orchestration layers.
