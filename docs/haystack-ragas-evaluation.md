# Haystack Ragas Evaluation

RRF-93 establishes a controlled, end-to-end [Ragas](https://docs.ragas.io)
evaluation of the Haystack adaptation of `ragrefine`. It compares the B0
original Haystack context against `RagRefineComponent`-selected context drawn
from the identical retrieved Top-N pool, generating answers with DeepSeek
through its OpenAI-compatible endpoint.

This is an evaluation harness. It does **not** change first-stage retrieval and
does **not**, by itself, claim that refinement improves generation or context
quality. Any such claim requires the recorded comparison to support it.

## Fixed human-reviewed test set

`benchmarks/haystack/testset/ragas-testset-v1.json` freezes a small retrieval
corpus (15 documents) and 10 hand-reviewed QA pairs. Each pair records:

- a question;
- a reference answer grounded in one or more reference documents;
- the IDs of those reference documents.

The corpus intentionally includes same-topic distractors (for example, three
Python release notes and two Ragas-related documents) so that ranking and
context selection, not just first-stage retrieval, determine the selected
context. A SHA-256 sidecar protects the exact artifact, and
`benchmarks/haystack/testset.py` verifies the checksum and rejects malformed or
internally inconsistent data.

## Comparison

For every question the harness retrieves one frozen Top-N pool (BM25, in
memory) and reuses it for both branches:

```text
Top-N pool
    ├── baseline (B0): original retrieval order, truncated to Top-K
    └── refined: RagRefineComponent-selected Top-K from the same pool
```

The default refinement profile is the retained lightweight lexical profile
(`B2-L`), matching `benchmarks/haystack/ab.py`. `top_n` and `top_k` default to
5 and 3 and are configurable via the CLI.

Answers are generated with the fixed prompt template shared by the bypass and
refinement paths, using `OpenAIChatGenerator` pointed at
`https://api.deepseek.com/v1` with temperature 0.

## Metrics

The harness scores both branches with three LLM-only Ragas metrics, which need
only the DeepSeek chat endpoint (no embedding endpoint is required):

| Metric | Question answered |
| --- | --- |
| `faithfulness` | Is the answer factually consistent with the retrieved context? |
| `context_recall` | Does the retrieved context cover the reference answer? |
| `factual_correctness` | Is the answer factually correct against the reference? |

Aggregate and per-query scores are recorded for both branches. No metric is
interpreted as a probability beyond Ragas's own 0–1 definition.

## Running

Ragas is an optional dependency and is not part of fast CI.

```text
uv sync --extra ragas --extra haystack
```

```powershell
$env:DEEPSEEK_API_KEY = "..."
uv run python -m benchmarks.haystack.ragas_eval `
  --output-dir benchmarks/results/haystack-ragas-v1 `
  --model deepseek-chat `
  --top-n 5 --top-k 3
```

The command writes `ragas-results.json` into a **new** directory and fails
rather than overwrite existing results.

## Artifact and reproducibility

`ragas-results.json` records:

- the test-set checksum and shape;
- the model, endpoint, and generation settings;
- `top_n`, `top_k`, and the refinement profile;
- per-query paired context IDs, answers, generation latencies, and per-query
  Ragas scores for both branches;
- aggregate baseline and refined metrics;
- environment provenance (Python version and git commit).

Answers and metrics are non-deterministic by nature (LLM calls); ranking,
context selection, and the frozen test set are deterministic and reproducible.

## Out of scope

- changing or regenerating first-stage retrieval;
- running Ragas in fast CI;
- production deployment;
- any claim of improvement before the completed evaluation is recorded.
