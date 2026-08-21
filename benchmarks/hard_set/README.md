# Structured hard-negative benchmark

`structured-hard-negatives-v1.json` is a frozen, curated diagnostic benchmark
for explicit-constraint ranking failures. It contains 48 candidate-level cases:
12 query groups with one relevant candidate and three same-topic negatives each.

The four diagnostic categories are `wrong_version`, `wrong_identifier`,
`wrong_date`, and `wrong_numeric_value`. Every record stores its query and
candidate IDs/text, binary relevance judgment, category, and the expected
structured constraint. The JSON's `.sha256` sidecar protects the exact frozen
artifact; validation also rejects schema, ordering, ID, category, and group
inconsistencies.

| Category | Representative query group | Diagnostic rationale |
|---|---|---|
| `wrong_version` | `version-01` | Same deployment explanation, but a different software/model release. |
| `wrong_identifier` | `identifier-01` | Same incident-resolution language, but the incident or request ID differs. |
| `wrong_date` | `date-01` | Same operational event, but a different dated occurrence is described. |
| `wrong_numeric_value` | `numeric-01` | Same latency/policy context, but the threshold or percentage differs. |

This is deliberately curated/synthetic diagnostic data. It is not BEIR, does
not measure general retrieval performance, and must not be presented as a
quality improvement claim. It exists to make per-category B2 failure analysis
reproducible. No retrieval, neural inference, or fusion is needed to validate it.

### Selection limitation

In `structured-hard-negatives-v1`, the relevant candidate is deliberately at
the original retrieval rank of 1 for all 12 query groups. It is consequently a
deterministic pattern/schema regression fixture, not a discriminative profile-
selection benchmark: original order already has Top-1 accuracy and
unrestricted MRR of 1.0. Do not use this version to choose between B0/B1/B2/B3
profiles or to claim an improvement. A future comparative hard set must place
relevant candidates below rank 1 while preserving its stated constraints.
