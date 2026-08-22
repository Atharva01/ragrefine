# Haystack Ragas A/B report

- Input artifact: `benchmarks\results\haystack-ragas-v1\ragas-results.json`
- Scores artifact: `benchmarks\results\haystack-ragas-scores-v1\ragas-scores.json`
- Profile: B2-L lexical
- Paired queries scored: 13
- Metrics: context_precision, id_based_context_recall, faithfulness, factual_correctness

## Per-metric comparison

| Metric | Baseline | Refined | Mean Δ | CI95 (Δ) | W/L/T | Decision |
|---|---:|---:|---:|---:|---:|---|
| context_precision | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | 0/0/13 | inconclusive |
| factual_correctness | 0.6685 | 0.7177 | 0.0492 | [-0.0508, 0.1554] | 3/1/9 | inconclusive |
| faithfulness | 0.7297 | 0.8076 | 0.0779 | [-0.0148, 0.1864] | 3/2/8 | inconclusive |
| id_based_context_recall | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | 0/0/13 | inconclusive |

**Overall decision: inconclusive** — mixed or absent signals (0 supported improvements, 0 regressions)

## Context size

- Baseline words: 689
- Refined words: 692
- Delta words: 3

## Representative regressions (refined below baseline)
- factual_correctness: q-13 (Δ -0.3300)
- faithfulness: q-08 (Δ -0.1466), q-04 (Δ -0.0455)

## Verification

- OK: True
