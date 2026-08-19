# Benchmark Signal and Coverage Audit

**Scope:** RRF-58 — current B0–B3 evidence before secondary-dataset and B4 work.

Machine-readable artifacts remain the source of truth. This audit records what
the current suite can support, not a claim that any retained profile generalizes.

## Current evidence

| Component | What it measures | Current signal | Important limitation |
|---|---|---|---|
| SciFact frozen Top-50 | General post-retrieval ranking effectiveness on public scientific claims | 300 queries; nDCG@5, MRR, Precision@5, Recall@5, and candidate-pool Recall@50 | One domain and one fixed first-stage retriever; the `0.7919` candidate-pool recall ceiling limits every post-ranking profile. |
| Structured hard-negative set | Exact-constraint handling for dates, identifiers, numeric values, and versions | 12 curated queries across four categories; Top-1, MRR, and relevant rank | Every evaluated original, neural, lexical, pattern, and B3 ranking is Top-1/MRR `1.0`; it cannot distinguish the profiles. It is not a general retrieval benchmark. |
| B1/B2/B3 runtime artifacts | Measured execution cost for the recorded model/device/configuration | Candidate count, total time, p50/p95 estimates, and throughput where recorded | These are not service latency or cross-hardware claims. B3 values are additive component estimates, not an independently measured end-to-end deployment. |
| Context efficiency | Evidence retention, duplicate suppression, token/candidate reduction, and selection runtime | Not yet measured | RRF-56 and RRF-57 implement the deterministic components, but no B4 ablation artifact exists yet. |

## Known blind spots and unsupported conclusions

- SciFact results do **not** establish that B1-reference or B2 lexical
  generalize to finance, question-answering, web, or product-specific corpora.
- Candidate-pool Recall@50 is a first-stage retrieval ceiling, not a refinement
  result. A relevant document outside the frozen pool cannot be recovered by a
  reranker, signal, deduplicator, or selector.
- The structured hard-negative set is saturated and must not be cited as proof
  of relative ranking robustness. It remains useful only as a deterministic
  schema/constraint regression diagnostic until it becomes discriminative.
- B3 rejection is supported on the fixed SciFact population and saturated hard
  set only. It does not establish a universal statement about rank fusion.
- Current runtime numbers describe the recorded models, revisions, batch
  configuration, device, and host. They do not measure production tail latency,
  concurrent throughput, memory, cost, or a pure CPU-versus-CUDA comparison when
  model profiles differ.
- No current artifact measures whether post-ranking context selection retains
  relevant evidence while reducing tokens or duplicate context. That is B4 work,
  not an inference from ranking metrics.

## Evaluation concerns must remain separate

| Concern | Primary evidence | Do not infer |
|---|---|---|
| General retrieval ranking | Frozen BEIR qrels and ranking metrics | Context efficiency, production latency, or first-stage retrieval improvement |
| Targeted structured diagnostics | Frozen hard-negative category outcomes | Generalization or aggregate retrieval quality |
| Context efficiency | B4 selected IDs, exclusion reasons, token/candidate reduction, and evidence retention | Better ranking quality without B4 measurements |
| Runtime/deployment | Recorded environment and timing artifacts | Retrieval-quality improvement or cross-profile hardware superiority |

## Secondary-dataset qualification criteria

NFCorpus and FiQA are candidates, not yet permanent benchmark components. Each
must satisfy every criterion below before retained B1-reference and B2 lexical
profiles are evaluated on it.

| Criterion | Measurable qualification evidence |
|---|---|
| Sensitivity | Frozen B0 ranking is not saturated, and a deterministic ranking perturbation changes at least one reported metric or per-query outcome. |
| Relevance | Dataset task/corpus differs materially from SciFact and has public qrels appropriate for ranking evaluation. |
| Reproducibility | Pinned dataset/retriever configuration, frozen Top-N snapshot, SHA-256 sidecar, environment metadata, and repeated evaluation with identical metrics. |
| Runtime cost | Candidate generation and retained-profile ranking times, candidate count, and device/backend are recorded; model work remains outside fast CI. |
| Interpretability | Query IDs/text, candidate IDs/text, qrels, aggregate metrics, and per-query wins/losses are persisted so regressions can be inspected. |

RRF-59 must produce the NFCorpus and FiQA frozen B0 snapshots and document
their candidate-pool Recall@N. RRF-62 and RRF-63 can then determine whether
retained-profile findings are stable across qualified datasets.

## Interim benchmark decision

- Retain SciFact as the current primary effectiveness benchmark.
- Retain the hard set only as a deterministic constraint-regression diagnostic;
  do not treat it as profile-selection evidence while saturated.
- Retain runtime artifacts as deployment observations separate from quality.
- Add no permanent benchmark tier until RRF-59, RRF-62, RRF-63, and RRF-64
  supply the missing cross-dataset sensitivity and cost evidence.
