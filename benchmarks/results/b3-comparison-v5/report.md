# B3 Consolidated Experiment Analysis

Machine-readable artifacts are the source of truth. This report is generated from frozen rankings; it does not rerun retrieval or neural inference.

## SciFact comparison

| Experiment | nDCG@5 | MRR | Precision@5 | Recall@5 |
|---|---:|---:|---:|---:|
| B0 | 0.4592 | 0.4381 | 0.1240 | 0.5567 |
| B1-reference | 0.6262 | 0.6190 | 0.1507 | 0.6839 |
| B1-light | 0.6112 | 0.6043 | 0.1473 | 0.6644 |
| B2-lexical | 0.5432 | 0.5238 | 0.1353 | 0.6289 |
| B2-pattern | 0.4559 | 0.4370 | 0.1207 | 0.5447 |
| B3-original-lexical | 0.5383 | 0.5177 | 0.1360 | 0.6223 |
| B3-original-pattern | 0.4672 | 0.4470 | 0.1247 | 0.5553 |
| B3-light | 0.5272 | 0.5069 | 0.1367 | 0.6153 |
| B3-reference | 0.5715 | 0.5541 | 0.1440 | 0.6477 |

## B3 decision

| Profile | Decision | Evidence |
|---|---|---|
| B3-original-lexical | reject | SciFact nDCG@5, MRR, and Recall@5 are lower than B2-lexical, which uses fewer stages. |
| B3-original-pattern | reject | Its small B0 gain adds no hard-set separation and is dominated by retained B2-lexical on SciFact. |
| B3-light | reject | SciFact nDCG@5, MRR, and Recall@5 are lower than B2-lexical; it also adds pattern plus fusion cost. |
| B3-reference | reject | It regresses every primary SciFact metric relative to B1-reference and adds CPU signal cost; the hard set is saturated. |

Recommended default: **retain B1-reference when CUDA latency is acceptable** — B1-reference has the strongest measured SciFact metrics. Use B2-lexical for a CPU-only low-cost deployment.

## Hard-negative interpretation

All frozen hard-negative categories are saturated at Top-1/MRR 1.0 for the original and evaluated B3 rankings; they provide no measured evidence that fusion adds robustness for this set.

## Runtime

The B3 profile times are additive component estimates. They are deployment observations, not retrieval-quality evidence.
