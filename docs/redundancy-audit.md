# Frozen Candidate-Pool Redundancy Audit

**Scope:** RRF-60 — characterize existing B0 candidate pools before B4
deduplication or context-selection evaluation.

Machine-readable results in
`benchmarks/results/redundancy-audit-v1.json` are the source of truth. This
document summarizes candidate-pool pressure only; it does not measure ranking
quality, evidence retention, or downstream answer quality.

## Method

- Inputs are the frozen SciFact, NFCorpus, and FiQA Top-50 B0 snapshots.
- Each candidate is counted using `unicode_word_regex_v1`, a deterministic
  word-token proxy. It is **not** an LLM tokenizer; model-specific context
  budgets still require a caller-provided `TokenCounter`.
- Exact duplicates use the existing normalized-content SHA-256 rule.
- Near duplicates use the existing rank-ordered five-token-shingle Jaccard
  rule with predeclared strict thresholds `0.80`, `0.90`, and `0.95`.
- Qrels and relevance outcomes are not read by this audit.

## Candidate-pool context pressure

| Dataset | Queries | Candidates/query | Mean word tokens/query | Exact duplicate-candidate share |
|---|---:|---:|---:|---:|
| SciFact | 300 | 50 | 11,230.8 | 0.0000% |
| NFCorpus | 323 | 50 | 12,133.0 | 1.1641% |
| FiQA | 648 | 50 | 8,620.8 | 0.0154% |

All available pools exceed a typical small context budget when their complete
Top-50 text is assembled, so budget-aware selection remains a meaningful B4
concern. This result does not establish that a particular selector preserves
relevance; that needs a separate B4 evaluation.

## Duplicate characteristics

| Dataset | Threshold | Duplicate clusters/query | Redundant word-token share |
|---|---:|---:|---:|
| SciFact | 0.80 / 0.90 / 0.95 | 0.0000 | 0.0000% |
| NFCorpus | 0.80 / 0.90 / 0.95 | 0.5913 | 1.2336% |
| FiQA | 0.80 | 0.0231 | 0.0713% |
| FiQA | 0.90 | 0.0201 | 0.0685% |
| FiQA | 0.95 | 0.0170 | 0.0650% |

NFCorpus is unchanged across the near-duplicate thresholds, indicating that
the observed redundancy is already exact-normalized duplication rather than a
threshold-sensitive near-duplicate signal. SciFact has no measured candidate
duplication under this rule; FiQA has only a negligible amount.

## Decision

A separate targeted redundancy diagnostic is necessary for deduplication
evaluation. The public frozen pools provide insufficient duplicate volume and
near-duplicate threshold sensitivity to establish whether B4 suppression
improves context efficiency or retains evidence. Such a diagnostic should use
predeclared overlapping/duplicate contexts and report retained IDs, suppressed
IDs and reasons, token reduction, and evidence retention separately from
ranking outcomes. It must not tune thresholds against relevance results.

## Provenance

- `benchmarks/results/scifact-b0/snapshot.json`
- `benchmarks/results/nfcorpus-b0-v1/snapshot.json`
- `benchmarks/results/fiqa-b0-v2/snapshot.json`
- `benchmarks/results/redundancy-audit-v1.json`
