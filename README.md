# ragrefine

**Post-retrieval refinement for RAG systems.**

`ragrefine` is an experimental Python library for improving the quality of retrieved context **after retrieval and before generation**.

Instead of replacing your retriever, vector database, or RAG framework, `ragrefine` takes an existing candidate set and applies additional relevance signals, reranking, fusion, and context-selection logic.

> **Status:** v0.1 implements independent refinement channels, optional RRF,
> deduplication, greedy budget selection, and structured tracing. Only the
> recorded B1-reference and B2-L configurations have measured evidence; no
> unmeasured technique is described as an improvement.

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

## Current API

`Refiner()` is the B0-compatible original-order configuration. Explicit
`RefinerConfig` enables independent neural, lexical, pattern, and fused
channels over the same input pool; selection is applied only after final
ranking.

```python
from ragrefine import Candidate, CandidateSet, Refiner

candidate_set = CandidateSet(
    name="dense",
    candidates=(Candidate(id="chunk-42", text="Retrieved evidence", retrieval_rank=1),),
)

result = Refiner().refine("What is the evidence?", candidate_set, top_k=5)

assert result.candidates[0].candidate.id == "chunk-42"
assert result.candidates[0].rank == 1
```

`Refiner`, `RefinementResult`, and trace models are available from the package
root. Internal modules remain implementation details.

---

## Implemented refinement pipeline

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

## Evaluation Results

### B2 independent signal ablations

Both B2 signals reused the frozen SciFact B0 Top-50 snapshot; retrieval was not
rerun. The curated hard-negative set is a diagnostic, not a general retrieval
benchmark. Machine-readable artifacts under `benchmarks/results/scifact-b2-*`
remain the source of truth.

| Experiment | nDCG@5 | MRR@5 | Precision@5 | Recall@5 | Candidate-pool Recall@50 | Decision |
|---|---:|---:|---:|---:|---:|---|
| B2-L — lexical / CPU | 0.5432 | 0.5238 | 0.1353 | 0.6289 | 0.7919 | Retain |
| B2-P — patterns / CPU | 0.4559 | 0.4370 | 0.1207 | 0.5447 | 0.7919 | Modify |

B2-L improved the measured SciFact ranking metrics relative to B0. B2-P was
perfect on the curated constraint diagnostic (Top-1 and unrestricted MRR both 1.0) but did
not improve SciFact, so it is not retained for later fusion without revision.

### B3 frozen-fusion evaluation

All B3 profiles used the same frozen SciFact B0 Top-50 candidate population,
equal RRF weights, and `k=60`. The machine-readable consolidated comparison in
`benchmarks/results/b3-comparison-v5/comparison.json` is the source of truth.

| Experiment | nDCG@5 | MRR@5 | Precision@5 | Recall@5 | Decision |
|---|---:|---:|---:|---:|---|
| B3 original + lexical | 0.5383 | 0.5177 | 0.1360 | 0.6223 | Reject — lower nDCG, MRR@5, and recall than B2 lexical |
| B3 original + pattern | 0.4672 | 0.4470 | 0.1247 | 0.5553 | Reject — no hard-set separation and below B2 lexical |
| B3-light | 0.5272 | 0.5069 | 0.1367 | 0.6153 | Reject — lower nDCG, MRR@5, and recall than B2 lexical |
| B3-reference | 0.5715 | 0.5541 | 0.1440 | 0.6477 | Reject — below B1-reference with additional stages |

B3-reference remains better than B0 on SciFact, but it is not an improvement
over the retained B1-reference profile. B3-light is likewise not an improvement
over B2 lexical. Per-query win/loss/unchanged counts and representative cases
are persisted in the consolidated comparison rather than inferred from the
aggregate table.

The hard-negative set is saturated: original, individual channels, and all B3
profiles have Top-1 accuracy and unrestricted MRR of 1.0 for wrong-date,
wrong-identifier, wrong-numeric-value, and wrong-version cases, because each
v1 group places its relevant candidate at retrieval rank 1. It is therefore a
deterministic schema diagnostic, not a discriminative profile-selection set;
it must not be used to claim a fusion gain.

| B3 profile | Deployment composition | Total runtime | p50 / p95 query estimate | Pairs/s |
|---|---|---:|---:|---:|
| B3-light | CPU lexical + pattern + RRF | 7.10 s | 16.3 / 25.9 ms | 2,112.1 |
| B3-reference | CUDA neural + CPU signals + RRF | 488.79 s | 1,565.4 / 1,761.9 ms | 30.7 |

These are additive component estimates, kept separate from effectiveness. The
recommended quality profile is B1-reference where CUDA latency is acceptable;
B2 lexical is the recommended CPU-only low-cost profile. No B3 profile is
retained as the default refinement strategy.

Experiments reuse a **frozen first-stage candidate snapshot**: retrieval is not
regenerated for B0/B1 comparisons. Later refinement metrics must therefore be
interpreted relative to the fixed candidate-pool recall ceiling, rather than as
evidence that first-stage retrieval changed. Machine-readable artifacts are the
source of truth; this README is a concise reference. Improvement is never
assumed—a run is described as improved only when its recorded metrics support it.

```text
B0 frozen Top-50
├── B1-reference: MiniLM-L6 / CUDA
└── B1-light: TinyBERT-L2 / CPU
```

### SciFact B0 baseline

The validated B0 run used **SciFact**, **300 queries**, and **Top-50 candidates
per query**. Its frozen snapshot SHA-256 is
`fc08cf7b496c8cd7c9020a81560d08edc10f389b27682f2b6e722ccfef0793dc`.

| Experiment | nDCG@5 | MRR@5 | Precision@5 | Recall@5 | Candidate-pool Recall@50 | Status |
|---|---:|---:|---:|---:|---:|---|
| B0 | 0.4592 | 0.4381 | 0.1240 | 0.5567 | 0.7919 | Validated baseline |
| B1-reference — `cross-encoder/ms-marco-MiniLM-L6-v2` / CUDA | 0.6262 | 0.6190 | 0.1507 | 0.6839 | 0.7919 | Measured on the B0 snapshot |
| B1-light — `cross-encoder/ms-marco-TinyBERT-L2-v2` / CPU | 0.6112 | 0.6043 | 0.1473 | 0.6644 | 0.7919 | Measured on the B0 snapshot |
| B2 / B3 / B4 | — | — | — | — | — | Not evaluated |

### Neural runtime / deployment observations

Runtime results are kept separate from retrieval quality. The B1-reference and
B1-light rows are different **model and device** profiles, so they are not a
pure backend or hardware comparison. The TinyBERT CPU and CUDA rows use the
same model and configuration. p50/p95 values are per-query estimates derived
from batch timings, not independently timed queries.

| Profile | Device / backend | Batch size | Total reranking runtime | p50 query estimate | p95 query estimate | Query-document pairs/s |
|---|---|---:|---:|---|---|---:|
| B1-reference — MiniLM-L6 | CUDA / SentenceTransformers | 512 | 481.7 s | 1,549.0 ms | 1,735.5 ms | 31.1 |
| B1-light — TinyBERT-L2 | CPU / SentenceTransformers | 512 | 79.1 s | 236.8 ms | 315.8 ms | 189.6 |
| B1-light — TinyBERT-L2 | CUDA / SentenceTransformers | 512 | 18.0 s | 31.9 ms | 41.1 ms | 833.5 |

For the recorded TinyBERT-L2 runs, CUDA completed the same 15,000
query-document pairs about **4.4× faster** than CPU (18.0 s versus 79.1 s).
This is an execution comparison only; both runs produced the same aggregate
retrieval metrics.

Both runs evaluated **300 queries × 50 frozen candidates = 15,000
query-document pairs**. `queries_per_batch=24` controls how many query pools
the runner groups together (up to 1,200 pairs); `batch_size=512` controls the
CrossEncoder inference minibatch size. CUDA therefore processed the same work
with higher parallel throughput—it did not evaluate more queries.

Artifacts referenced above:

- `benchmarks/results/scifact-b0/` — snapshot, checksum, configuration, environment, and B0 metrics.
- `benchmarks/results/scifact-b1-batched/` — MiniLM-L6/CUDA ranking, metrics, environment, and batch timings.
- `benchmarks/results/scifact-b1-tinybert/` — TinyBERT-L2/CPU ranking, metrics, environment, and batch timings.
- `benchmarks/results/scifact-b1-tinybert-cuda/` — TinyBERT-L2/CUDA ranking, metrics, environment, and batch timings.

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
- MRR@5
- Precision@5
- Recall@5
- candidate-pool Recall@N

Operational measurements include refinement latency and candidate throughput.

The frozen BEIR evaluation harness is implemented for SciFact, NFCorpus, and
FiQA. See the [benchmark baseline guide](docs/benchmark-baseline.md) for setup,
artifact layout, and reproducibility commands.

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
- [x] Core candidate domain contracts
- [x] No-op refinement contract and tracing
- [x] Frozen B0 retrieval baseline
- [x] CrossEncoder reranking
- [x] Lexical and regex/pattern signals
- [ ] Entity signals (future hypothesis)
- [x] Reciprocal Rank Fusion
- [x] Deduplication and context budgeting
- [x] Hard-negative diagnostic evaluation
- [ ] Haystack integration
- [ ] ONNX inference experiment

---

## Project philosophy

`ragrefine` treats retrieval refinement as an **information-retrieval problem first** and an LLM integration problem second.

The project will prefer measurable ranking improvements, reproducible experiments, and interpretable failure analysis over adding more orchestration layers.
