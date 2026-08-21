# Candidate benchmark-suite strategies (RRF-64)

This decision compares the existing frozen benchmark artifacts before RRF-65
freezes the v0.1 evaluation contract. Machine-readable result artifacts remain
the source of truth. This document selects a suite strategy; it does not claim
that a profile improves retrieval beyond the measured datasets.

## Evidence used

All retained comparisons use the same frozen Top-50 pool within each dataset.
The qualified datasets and retained profiles are:

| Dataset | Queries | Domain | Retained profiles | Unique signal |
| --- | ---: | --- | --- | --- |
| SciFact | 300 | Scientific claims | B0, B1-reference, B2-L | Controlled primary ranking reference; high pool Recall@50 (0.7919) |
| NFCorpus | 323 | Biomedical information retrieval | B0, B1-reference, B2-L | Ranking behaviour under a low pool-recall ceiling (0.1851) |
| FiQA | 648 | Financial questions | B0, B1-reference, B2-L | Domain-diverse lexical-regression guard: B2-L regresses on all primary aggregate metrics |

The cross-dataset analysis shows B1-reference above B0 on all four primary
metrics for all three datasets. B2-L improves on SciFact and NFCorpus but
regresses on FiQA. Thus a SciFact-only suite would fail to detect the observed
lexical failure mode.

## Strategy comparison

| Strategy | Included effectiveness evidence | Regression sensitivity and coverage | Recorded execution cost | Reproducibility | Decision |
| --- | --- | --- | --- | --- | --- |
| Minimal | SciFact B0, B1-reference, B2-L | Fast and controlled, but one scientific domain; misses the observed FiQA B2-L regression | B2-L: 2.78 s; B1-reference: 481.74 s for the recorded 300-query run | Strong: frozen snapshot, sidecars, environment, and replay | Reject as the permanent suite; retain as a fast pre-merge smoke tier only |
| Balanced | SciFact, NFCorpus, and FiQA B0/B1-reference/B2-L | Detects both cross-dataset B1 stability and the FiQA lexical regression; includes scientific, biomedical, and financial tasks | B2-L: 10.40 s across 1,271 queries; recorded B1-reference total: 4,987.93 s (83.13 min) | Strong for all three datasets: frozen Top-50 snapshots, checksums, profile provenance, per-query outputs, and replay | **Select** |
| Broad | Balanced suite plus B2-P, rejected B3 profiles, saturated hard set as profile-selection evidence, and all B4 policy variants | More artifacts but no demonstrated additional retained-profile decision signal; increases review and runtime burden | Includes B3 component estimates and expensive near-duplicate policies without a retained quality decision | Reproducible artifacts exist, but reproducibility alone does not make a component informative | Reject |

The B1-reference total is a recorded, not predicted, cost. Its FiQA run contains
a 2,983.1-second batch outlier and must not be treated as a stable latency
estimate. B1-reference/CUDA and B2-L/CPU also differ in model and device, so
their figures are not a hardware comparison. Candidate generation is excluded
from regression execution because the suite consumes the existing frozen
snapshots; regenerating retrieval would violate the comparison invariant.

## Non-redundant and rejected components

| Component | Role in the selected strategy | Evidence-based disposition |
| --- | --- | --- |
| SciFact | Primary effectiveness reference | Retain: controlled and high candidate-pool availability |
| NFCorpus | Secondary effectiveness coverage | Retain: exposes candidate-generation ceiling while both retained profiles improve ranking metrics |
| FiQA | Secondary effectiveness coverage | Retain: reveals B2-L’s aggregate regression and prevents SciFact-only generalization |
| Structured hard-negative set v1 | Deterministic schema/constraint regression diagnostic only | Retain as diagnostic; reject for profile selection because its relevant candidate is initially rank 1 in every group, making original and evaluated rankings Top-1/unrestricted-MRR 1.0 |
| B2-P pattern profile | No retained effectiveness tier | Reject from routine suite: it did not improve SciFact and the hard set is saturated |
| B3 profiles | No retained effectiveness tier | Reject from routine suite: none surpassed its retained single-channel comparator on SciFact |
| B4 S1/S2/S4 deduplication policies | No routine context-efficiency tier | Reject: no incremental benefit in the measured SciFact ablation; S2/S4 are also expensive |
| B4 S3 1,000-word budget | Candidate for a future modified experiment | Do not retain: its measured evidence-retention drop exceeds the predeclared limit |

## Selected balanced strategy

Use a tiered balanced suite for the v0.1 contract:

1. **Fast regression tier:** SciFact B0, B1-reference, and B2-L analysis from
   persisted rankings. This is a fast validation of artifact compatibility and
   known ordering, not model inference in default CI.
2. **Effectiveness tier:** SciFact, NFCorpus, and FiQA with B0,
   B1-reference, and B2-L on their own frozen Top-50 snapshots. Require this
   before a change is described as a retained-profile quality improvement.
3. **Diagnostic tier:** structured-hard-negatives-v1 only for deterministic
   schema/constraint regression. It must not select or compare profiles until
   a discriminative replacement exists.
4. **Context-efficiency tier:** retain the B4 ranking-only control (S0) as a
   measurement baseline. Run further deduplication or token-budget policies
   only as explicitly scoped experiments, not as routine regressions.
5. **Runtime tier:** record B1-reference and B2-L timing separately by
   model/device/configuration. Do not combine runtime with retrieval-quality
   pass/fail rules or run neural inference in default CI.

RRF-65 should freeze the precise commands, required profiles by tier, metric
names, Top-N/Top-K settings, artifact policy, and reporting thresholds. It
should preserve this selection unless new measured evidence changes it.

## Artifact references

- `benchmarks/results/scifact-b0/`, `scifact-b1-batched/`, and
  `scifact-b2-lexical/`
- `benchmarks/results/nfcorpus-*-v1/` and
  `benchmarks/results/nfcorpus-secondary-evaluation-v1/`
- `benchmarks/results/fiqa-*-v1/`, `fiqa-b0-v2/`, and
  `benchmarks/results/fiqa-secondary-evaluation-v1/`
- `benchmarks/results/cross-dataset-stability-v2/`
- `benchmarks/results/scifact-b4-selection-v1/`

See `docs/cross-dataset-stability.md`, `docs/secondary-evaluation.md`,
`docs/b4-context-selection.md`, and `docs/benchmark-audit.md` for the measured
underlying evidence and reproduction commands.
