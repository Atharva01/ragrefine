# B4 context-selection ablation

RRF-61 evaluated context selection over the persisted SciFact B1-reference and
B2-L rankings. Both inputs contain 300 queries and the same frozen Top-50
candidate population, identified by B0 checksum
`fc08cf7b496c8cd7c9020a81560d08edc10f389b27682f2b6e722ccfef0793dc`.
No retrieval or neural reranking was executed.

## Frozen configuration

- S0: ranking-only Top-5 control.
- S1: exact normalized-content deduplication, then Top-5.
- S2: exact plus strict `> 0.90` 5-token-shingle Jaccard deduplication,
  then Top-5.
- S3: 1,000-word budget and Top-5, without deduplication.
- S4: S2 deduplication plus the S3 budget and Top-5.

The dependency-free token counter uses the Unicode word-regex definition from
the RRF-60 audit. Evidence retention is measured only against relevant IDs
present in the frozen Top-50 pool, so candidate-generation misses are excluded.

## Selection results

| Profile | Policy | Evidence retention (micro / mean query) | Selected candidates | Candidate reduction | Selected word tokens | Token reduction | Exact / near suppressed | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B1-reference | S0 | 84.33% / 86.07% | 1,500 | 90.00% | 365,678 | 89.15% | 0 / 0 | Retain |
| B1-reference | S1 | 84.33% / 86.07% | 1,500 | 90.00% | 365,678 | 89.15% | 0 / 0 | Reject |
| B1-reference | S2 | 84.33% / 86.07% | 1,500 | 90.00% | 365,678 | 89.15% | 0 / 0 | Reject |
| B1-reference | S3 | 80.60% / 84.06% | 1,306 | 91.29% | 282,409 | 91.62% | 0 / 0 | Modify |
| B1-reference | S4 | 80.60% / 84.06% | 1,306 | 91.29% | 282,409 | 91.62% | 0 / 0 | Reject |
| B2-L | S0 | 75.75% / 79.25% | 1,500 | 90.00% | 388,574 | 88.47% | 0 / 0 | Retain |
| B2-L | S1 | 75.75% / 79.25% | 1,500 | 90.00% | 388,574 | 88.47% | 0 / 0 | Reject |
| B2-L | S2 | 75.75% / 79.25% | 1,500 | 90.00% | 388,574 | 88.47% | 0 / 0 | Reject |
| B2-L | S3 | 67.91% / 72.03% | 1,287 | 91.42% | 285,438 | 91.53% | 0 / 0 | Modify |
| B2-L | S4 | 67.91% / 72.03% | 1,287 | 91.42% | 285,438 | 91.53% | 0 / 0 | Reject |

Candidate and token reductions above are relative to the complete 15,000-item
Top-50 input. S1 and S2 have no incremental effect over S0. S4 has no
incremental effect over S3. The 1,000-word S3 budget reduces selected context,
but its maximum mean-query evidence-retention drop is 7.23 percentage points,
above the predeclared 2-point limit. The measured policy decisions are therefore:

- retain S0 as the ranking-only control;
- reject S1 and S2 for this SciFact population because they remove no duplicates;
- modify S3 by revisiting its budget rather than retaining the tested 1,000-word
  setting;
- reject S4 because deduplication adds no effect over S3.

These decisions apply to this frozen SciFact experiment. They do not establish
that deduplication is universally ineffective; the separate RRF-60 redundancy
audit found corpus-dependent redundancy.

## Runtime

| Profile | Policy | Total | p50/query | p95/query | Input candidates/s |
|---|---:|---:|---:|---:|---:|
| B1-reference | S0 | 16.3 ms | 0.05 ms | 0.08 ms | 919,467 |
| B1-reference | S1 | 320.4 ms | 1.03 ms | 1.37 ms | 46,814 |
| B1-reference | S2 | 100.75 s | 323.18 ms | 442.00 ms | 148.9 |
| B1-reference | S3 | 777.9 ms | 3.47 ms | 5.25 ms | 19,282 |
| B1-reference | S4 | 101.82 s | 326.36 ms | 458.41 ms | 147.3 |
| B2-L | S0 | 18.6 ms | 0.05 ms | 0.11 ms | 805,789 |
| B2-L | S1 | 328.9 ms | 1.06 ms | 1.54 ms | 45,605 |
| B2-L | S2 | 103.24 s | 334.56 ms | 446.07 ms | 145.3 |
| B2-L | S3 | 894.6 ms | 3.69 ms | 5.77 ms | 16,767 |
| B2-L | S4 | 103.81 s | 331.96 ms | 460.86 ms | 144.5 |

Near-duplicate matching is the dominant cost because it exhaustively compares
candidate pairs. Runtime is reported separately from evidence and reduction;
it is not evidence of retrieval-quality improvement.

## Artifacts and reproducibility

Machine-readable outputs are under
`benchmarks/results/scifact-b4-selection-v1/`. `input-manifest.json` records
both upstream ranking checksums, environments, and the common B0 checksum;
`config.json` records thresholds and token-counter settings; `per-query.json`
records selected IDs, relevant evidence, exclusion reasons, and timings; and
`summary.json` records aggregates and decisions. `reproduction.json` confirms
that replaying the exact persisted rankings produced logically identical
results, excluding nondeterministic wall-clock values.
