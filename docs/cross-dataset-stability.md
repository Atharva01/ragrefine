# Cross-dataset stability (RRF-63)

This report compares the retained B0, B1-reference, and B2-L rankings using
the same frozen Top-50 candidate pool within each dataset. The machine-readable
report in `benchmarks/results/cross-dataset-stability-v2/` is the source of
truth; this page is its concise interpretation.

No retrieval, neural inference, rejected B2-P profile, or B3 profile was run
for this analysis. Runtime is deliberately not part of the quality ordering:
B1-reference uses a CUDA CrossEncoder while B2-L is the CPU standard-library
lexical ranker.

## Aggregate ordering

| Dataset | Candidate-pool Recall@50 | Quality ordering for nDCG@5, MRR@5, Precision@5, Recall@5 |
| --- | ---: | --- |
| SciFact | 0.7919 | B1-reference > B2-L > B0 |
| NFCorpus | 0.1851 | B1-reference > B2-L > B0 |
| FiQA | 0.4336 | B1-reference > B0 > B2-L |

The per-profile primary metrics are recorded in `summary.json`. Candidate-pool
Recall@50 is invariant within a dataset because all profiles reorder the same
frozen candidates; it is a ceiling on refinement, not a profile-quality
comparison metric.

## Interpretation

B1-reference improves all four primary aggregate metrics on all three
datasets. This is cross-dataset evidence for the retained neural reranker on
these three fixed candidate pools, not a claim that it improves every query:
the artifact records per-query regressions too.

B2-L improves all four primary aggregate metrics on SciFact and NFCorpus, but
regresses on all four on FiQA. Its observed gain is therefore
dataset-specific—not evidence for a general lexical-ranking improvement. In
particular, the previous SciFact-only conclusion must not be generalized to
FiQA or to unseen domains.

Representative largest nDCG@5 movements are persisted with ranked document
IDs, relevance labels, and text previews in `representative-cases.json`.

| Dataset | Profile | Improvement case | Regression case |
| --- | --- | --- | --- |
| SciFact | B1-reference | `1137` — TNFAIP3 tumor-suppressor claim | `384` — noncommunicable-disease burden claim |
| SciFact | B2-L | `1099` — statins/cholesterol claim | `1088` — Bcl2/tumor-progression claim |
| NFCorpus | B1-reference | `PLAIN-2051` — saturated fat | `PLAIN-2620` — phytates/cancer treatment |
| NFCorpus | B2-L | `PLAIN-2790` — titanium dioxide/IBD | `PLAIN-681` — betel nuts |
| FiQA | B1-reference | `10601` — Bitcoin cost-basis purchases | `2648` — unemployment insurance |
| FiQA | B2-L | `10601` — Bitcoin cost-basis purchases | `1310` — Bitcoin-collateral mortgage |

These cases are diagnostics, not causal explanations. The artifacts preserve
the evidence necessary to inspect each rank movement without rerunning a
profile.

## Permanent suite recommendation

Retain all three datasets:

- **SciFact** is a controlled scientific-domain reference with the highest
  frozen-pool availability and improvements for both retained profiles.
- **NFCorpus** retains value despite its low 0.1851 candidate-pool ceiling:
  it exposes candidate-generation limits that a post-retrieval refiner cannot
  overcome, while still testing ranking changes.
- **FiQA** is essential as a domain-diverse lexical-regression guard. Removing
  it would hide the observed B2-L failure mode.

## Reproduce

```powershell
uv run python -m benchmarks.beir.cross_dataset reproduce `
  --output-dir benchmarks/results/cross-dataset-stability-v2
```

The replay reads the exact checked upstream artifacts recorded in
`input-manifest.json` and requires logically identical aggregate and
representative-case results.
