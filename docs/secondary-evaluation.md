# Retained-profile secondary evaluation

RRF-62 evaluates B0, B1-reference, and B2-L on the qualified frozen NFCorpus
and FiQA snapshots. It does not regenerate retrieval and does not execute the
rejected B2-P or B3 profiles. Machine-readable artifacts under
`benchmarks/results/` are the source of truth.

## Frozen inputs

| Dataset | Queries | Candidate pool | Snapshot SHA-256 |
|---|---:|---:|---|
| NFCorpus | 323 | Top-50 | `92f56832566e361abd723e5519414a6ba6b821813b3e0bdb9611ee42692b9dbe` |
| FiQA | 648 | Top-50 | `a2fe59001a24c97f8213151cc66dbf2c285b98a57fbb4a8cccded1762d814e71` |

B1-reference uses
`cross-encoder/ms-marco-MiniLM-L6-v2` at revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a` through SentenceTransformers on
CUDA. B2-L uses the retained deterministic lexical ranker on CPU.

## Retrieval quality

| Dataset | Profile | nDCG@5 | MRR | Precision@5 | Recall@5 | Pool Recall@50 |
|---|---|---:|---:|---:|---:|---:|
| NFCorpus | B0 | 0.2904 | 0.4547 | 0.2421 | 0.0933 | 0.1851 |
| NFCorpus | B1-reference | 0.3559 | 0.5363 | 0.2929 | 0.1177 | 0.1851 |
| NFCorpus | B2-L | 0.3153 | 0.4812 | 0.2619 | 0.1012 | 0.1851 |
| FiQA | B0 | 0.2082 | 0.2661 | 0.0975 | 0.2280 | 0.4336 |
| FiQA | B1-reference | 0.3168 | 0.4089 | 0.1460 | 0.3296 | 0.4336 |
| FiQA | B2-L | 0.1931 | 0.2572 | 0.0929 | 0.2184 | 0.4336 |

Candidate-pool Recall@50 is unchanged because every profile reranks the same
frozen candidates. B1-reference improves all four Top-5 ranking metrics on both
datasets. B2-L improves all four on NFCorpus but regresses on all four on FiQA:

| FiQA B2-L regression vs B0 | Absolute delta |
|---|---:|
| nDCG@5 | -0.0152 |
| MRR | -0.0089 |
| Precision@5 | -0.0046 |
| Recall@5 | -0.0096 |

These are per-dataset observations, not a final cross-dataset stability
decision; that analysis belongs to RRF-63.

## Per-query outcomes

| Dataset | Profile | Metric | Wins | Losses | Unchanged |
|---|---|---|---:|---:|---:|
| NFCorpus | B1-reference | nDCG@5 | 119 | 41 | 163 |
| NFCorpus | B1-reference | MRR | 62 | 25 | 236 |
| NFCorpus | B1-reference | Precision@5 / Recall@5 | 86 | 24 | 213 |
| NFCorpus | B2-L | nDCG@5 | 94 | 52 | 177 |
| NFCorpus | B2-L | MRR | 48 | 33 | 242 |
| NFCorpus | B2-L | Precision@5 / Recall@5 | 69 | 38 | 216 |
| FiQA | B1-reference | nDCG@5 | 213 | 43 | 392 |
| FiQA | B1-reference | MRR | 180 | 39 | 429 |
| FiQA | B1-reference | Precision@5 / Recall@5 | 149 | 12 | 487 |
| FiQA | B2-L | nDCG@5 | 122 | 138 | 388 |
| FiQA | B2-L | MRR | 118 | 120 | 410 |
| FiQA | B2-L | Precision@5 / Recall@5 | 80 | 90 | 478 |

The complete per-query metrics, deltas, and win/loss labels are persisted in
each evaluation directory's `per-query.json`.

## Runtime observations

Runtime is deployment evidence only and is not used to explain retrieval
quality. B1-reference/CUDA and B2-L/CPU differ in both algorithm and device, so
they are not a pure hardware or backend comparison.

| Dataset | Profile | Device | Measured total |
|---|---|---|---:|
| NFCorpus | B1-reference | CUDA | 526.8 s |
| NFCorpus | B2-L | CPU | 3.13 s |
| FiQA | B1-reference | CUDA | 3,979.4 s |
| FiQA | B2-L | CPU | 4.49 s |

The FiQA B1 total includes one measured batch outlier of 2,983.1 seconds while
the GPU remained saturated. The outlier is retained in the artifact rather than
discarded; this single run is not sufficient for a stable latency conclusion.

## Artifacts and reproduction

- `benchmarks/results/nfcorpus-b1-reference-v1/`
- `benchmarks/results/nfcorpus-b2-lexical-v1/`
- `benchmarks/results/nfcorpus-secondary-evaluation-v1/`
- `benchmarks/results/fiqa-b1-reference-v1/`
- `benchmarks/results/fiqa-b2-lexical-v1/`
- `benchmarks/results/fiqa-secondary-evaluation-v1/`

Each ranking has a SHA-256 sidecar and environment metadata. Each secondary
evaluation persists an input manifest, aggregate summary, complete per-query
evidence, and reproduction result. Both reproductions reported logically
identical results without retrieval or reranking.
