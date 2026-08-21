# v0.1 evaluation and regression contract

This contract freezes how `ragrefine` v0.1 measures retained profiles. It
implements the balanced strategy selected in RRF-64. Machine-readable
artifacts are the source of truth; this document defines their required
meaning, not a claim that every profile improves every query or domain.

## Frozen comparison invariant

Every post-retrieval comparison for one dataset uses the same immutable
first-stage Top-50 candidate snapshot. Refinement may only reorder, suppress,
or select candidates from that pool. It must not regenerate retrieval. The
candidate-pool recall is therefore a first-stage ceiling, not a refinement
metric that can improve between profiles.

## Retained profiles and configuration

| Profile | Status | Frozen configuration |
| --- | --- | --- |
| B0 | Baseline | Original Top-50 retrieval order |
| B1-reference | Retained neural profile | `cross-encoder/ms-marco-MiniLM-L6-v2`, revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`, SentenceTransformers, CUDA, batch size 512, 24 query pools per batch |
| B2-L | Retained lightweight profile | Deterministic standard-library lexical ranker on CPU |
| B2-P | Not retained | Not part of routine quality evaluation |
| B3 profiles | Not retained | Not part of routine quality evaluation |

Any change to a retained model, revision, backend, device, ranking semantics,
or profile configuration creates a new profile. It cannot overwrite or be
reported as the retained profile.

## Permanent tiers

| Tier | Inputs and profiles | When required | Pass/fail interpretation |
| --- | --- | --- | --- |
| Primary | SciFact B0, B1-reference, B2-L; 300-query frozen Top-50 | Any proposed retained-profile quality change; persisted-ranking analysis is suitable for a fast artifact smoke check | Report all aggregate and per-query deltas; do not call a profile improved without the recorded comparison |
| Secondary | NFCorpus and FiQA B0, B1-reference, B2-L; each on its own frozen Top-50 | Before generalizing a primary-tier result or retaining a changed ranking profile | A regression on either dataset must be reported; FiQA is the lexical-regression guard |
| Diagnostic | `structured-hard-negatives-v1` | Deterministic pattern/schema regression only | Do not use it to select profiles: its initial relevant candidate is rank 1 and current results are saturated |
| Context-efficiency | SciFact B4 S0 ranking-only control | Any deduplication or context-selection policy change | Compare evidence retention, candidate/token reduction, exclusion reasons, and runtime; quality claims require a separately scoped effectiveness evaluation |
| Runtime | B1-reference and B2-L timings, each with model/backend/device/batch configuration | Any deployment or performance statement | Runtime is deployment evidence only. It is not retrieval-quality evidence or a cross-hardware comparison when model and device differ |

Default fast CI must not download models, regenerate retrieval, or run neural
inference. It may validate code and persisted-artifact analysis where those
inputs are declared available. Full neural and effectiveness runs remain an
explicit benchmark workflow outside fast CI.

## Metrics and fixed cutoffs

Effectiveness reports must include these metrics for every qualified dataset:

| Metric | Definition |
| --- | --- |
| nDCG@5 | Graded relevance quality in the first five candidates |
| MRR@5 | Mean reciprocal rank of the first relevant candidate in the first five candidates; zero if none appears at ranks 1–5 |
| Precision@5 | Relevant candidates divided by five |
| Recall@5 | Relevant candidates in the first five divided by all relevant qrels |
| candidate-pool Recall@50 | Relevant qrels present anywhere in the frozen Top-50 pool; first-stage ceiling only |

The current machine-readable field is named `mrr`, but its implemented
semantics are **MRR@5**. New documentation and decisions must state the cutoff
explicitly and must not represent the current value as unrestricted MRR.

Fixed settings are Top-N = 50 and Top-K = 5 for all retained effectiveness
comparisons. A changed cutoff, dataset revision, candidate snapshot, model
revision, or ranking configuration is a new experiment and must produce a new
artifact directory rather than overwrite a retained result.

## Artifact and reproducibility requirements

Each qualified B0 dataset needs a snapshot containing query ID/text, candidate
ID/text, original rank, retrieval score, qrels, and configuration, plus a
SHA-256 sidecar. Each profile ranking needs a sidecar and environment metadata
that records the B0 snapshot checksum, ranking checksum, implementation/model
identity, revision where relevant, backend, device, and batch settings where
relevant.

Each evaluation/analysis directory must contain:

- frozen configuration and input manifest with checksums;
- aggregate metrics and complete per-query metrics or diagnostics;
- environment/provenance metadata;
- ranking, selection, or diagnostic evidence needed to inspect a decision;
- a reproduction record proving logically identical non-timing outputs.

`run` commands must use a new explicit output directory and fail rather than
silently overwrite history. `reproduce` commands consume only persisted inputs:
they verify sidecars/provenance and must not rerun retrieval or CrossEncoder
inference. The exact commands are maintained in
`docs/benchmark-baseline.md`.

## Reporting and regression policy

Before evaluation, declare the dataset tier, profiles, frozen inputs, metric
cutoffs, and configuration. Do not select a dataset, profile, cutoff, RRF
weight, or candidate pool after inspecting outcomes.

For each profile-versus-B0 comparison, report all five metrics above, absolute
aggregate deltas, per-query win/loss/unchanged counts, candidate-pool ceiling,
artifact paths/checksums, and environment. Keep runtime in a separate table.
Representative improvements and regressions may explain a result but cannot
replace aggregate reporting.

An aggregate decrease in any primary metric on a required dataset is a
regression and must be recorded; it cannot be hidden by wins on another
dataset. A profile is described as improved only for the exact measured
dataset/configuration and only when the reported metrics support that wording.
No claim may generalize a SciFact-only outcome. Changes that fail a required
tier are either rejected or explicitly marked for modification in a new,
predeclared experiment.

## Reproduction entry points

The baseline guide provides the commands for B0/B1/B2 generation and analysis.
The retained cross-dataset evidence is replayed with:

```powershell
uv run python -m benchmarks.beir.cross_dataset reproduce `
  --output-dir benchmarks/results/cross-dataset-stability-v2
```

The retained-artifact distribution policy does not alter these frozen
comparison, reporting, or provenance rules.
