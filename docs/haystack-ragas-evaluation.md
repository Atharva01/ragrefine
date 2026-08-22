# Haystack Ragas Evaluation

RRF-94 establishes the frozen, human-reviewed input contract for the
controlled end-to-end [Ragas](https://docs.ragas.io)
evaluation of the Haystack adaptation of `ragrefine`. It compares the B0
original Haystack context against `RagRefineComponent`-selected context drawn
from the identical retrieved Top-N pool, generating answers with DeepSeek
through its OpenAI-compatible endpoint.

This is an evaluation harness. It does **not** change first-stage retrieval and
does **not**, by itself, claim that refinement improves generation or context
quality. Any such claim requires the recorded comparison to support it.

## Fixed human-reviewed test set

`benchmarks/haystack/testset/ragas-testset-v2.json` freezes a small retrieval
corpus (21 documents) and 13 hand-reviewed QA pairs. Each pair records:

- a question;
- a reference answer grounded in one or more reference documents;
- the IDs of those reference documents.
- a category: factual, version, identifier, date, or numeric value.

The artifact records the human-review declaration and fixed B0-versus-B2-L
contract: Top-N 5, Top-K 3, no token limit, the shared prompt template,
`deepseek-chat`, temperature 0, a 512-token generation limit, and the three
declared Ragas metrics. The runner rejects changed model or Top-N/Top-K values
for this frozen artifact. The corpus intentionally includes same-topic and
structured-constraint distractors so that ranking and context selection, not
just first-stage retrieval, determine the selected context. A SHA-256 sidecar
protects the exact artifact, and
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

The refinement profile is the retained lightweight lexical profile (`B2-L`),
matching `benchmarks/haystack/ab.py`. The frozen contract fixes `top_n=5` and
`top_k=3`; a changed setting requires a new test-set and contract version.

Answers are generated with the fixed prompt template shared by the bypass and
refinement paths, using `OpenAIChatGenerator` pointed at
`https://api.deepseek.com/v1` with temperature 0 and a 512-token limit.

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

Ragas is an optional dependency and is not part of fast CI. DeepSeek is
configured **only through environment variables**: `DEEPSEEK_API_KEY` for the
key and `DEEPSEEK_MODEL` for the model name (defaulting to `deepseek-chat`);
the endpoint is the OpenAI-compatible `https://api.deepseek.com/v1`.

```text
uv sync --extra ragas --extra haystack
```

```powershell
$env:DEEPSEEK_API_KEY = "..."
$env:DEEPSEEK_MODEL = "deepseek-chat"
uv run python -m benchmarks.haystack.ragas_eval `
  --output-dir benchmarks/results/haystack-ragas-v1 `
  --top-n 5 --top-k 3
```

The command writes `ragas-results.json` (plus a matching
`ragas-results.json.sha256` sidecar used by the scoring/verify stage) into a
**new** directory and fails rather than overwrite existing results.

## Artifact and reproducibility

`ragas-results.json` records:

- the test-set checksum and shape;
- review provenance and the frozen evaluation configuration;
- the model, endpoint, prompt template, temperature, and generation limits;
- `top_n`, `top_k`, and the refinement profile;
- per-query paired context IDs, the shared-pool digest, the **exact prompts
  sent to the generator**, answers, the **full refinement trace**, retrieval/
  refinement/generation timing, and **per-arm generation failures** for both
  branches;
- a pairing summary: a query is scored only when both arms generated an
  answer, so aggregate baseline versus refined metrics stay comparable;
- per-query Ragas scores and aggregate baseline/refined metrics;
- environment provenance (Python version and git commit).

Answers and metrics are non-deterministic by nature (LLM calls); ranking,
context selection, and the frozen test set are deterministic and reproducible.
A failed arm is persisted with its typed failure and the prompt it attempted;
it never aborts the run and never exposes credentials.

## Scoring persisted outputs

`benchmarks.haystack.ragas_score` is the **offline scoring stage**: it consumes
a persisted `ragas-results.json` and the frozen test set and scores both arms
with Ragas without regenerating retrieval, context, or answers. Context text is
reconstructed from the checksum-verified test-set corpus by document ID, never
re-retrieved.

Metrics (per sample and aggregate, persisted for both arms):

| Metric | Ragas metric | Notes |
| --- | --- | --- |
| `context_precision` | Context Precision with Reference | LLM |
| `id_based_context_recall` | ID-based Context Recall | non-LLM, uses persisted IDs |
| `faithfulness` | Faithfulness | LLM |
| `factual_correctness` | Factual Correctness | LLM |
| `answer_relevancy` | Answer Relevancy | added **only** when an explicit pinned embedding provider is configured |

`answer_relevancy` requires an embedding endpoint, so it is added only when all
three of `RAGAS_EMBEDDING_MODEL`, `RAGAS_EMBEDDING_BASE_URL`, and
`RAGAS_EMBEDDING_API_KEY` are set; otherwise the metric is recorded as omitted.
DeepSeek judging is configured only through `DEEPSEEK_API_KEY` and
`DEEPSEEK_MODEL`.

```text
uv sync --extra ragas
```

```powershell
$env:DEEPSEEK_API_KEY = "..."
$env:DEEPSEEK_MODEL = "deepseek-chat"
uv run python -m benchmarks.haystack.ragas_score score `
  --input benchmarks/results/haystack-ragas-v1/ragas-results.json `
  --output-dir benchmarks/results/haystack-ragas-scores-v1
```

The command writes `ragas-scores.json` (plus a `ragas-scores.json.sha256`
sidecar) into a **new** directory, records the input artifact's SHA-256 and the
test-set checksum, and rejects an artifact that references unknown corpus IDs,
an incomplete paired row, or a different test set. **Metric errors are
preserved as failures and are never converted into scores**: a metric that
raises, or a sample that yields a non-finite value, is recorded under
`failures` and excluded from the aggregate. Judge prompt outputs (verdict
reasons) are persisted per sample under `judge_reasons`.

## Verification

The `verify` subcommand validates checksums and artifact consistency without
re-scoring and exits non-zero on any failure:

```powershell
uv run python -m benchmarks.haystack.ragas_score verify `
  --input benchmarks/results/haystack-ragas-v1/ragas-results.json `
  --scores benchmarks/results/haystack-ragas-scores-v1/ragas-scores.json
```

Checks performed:

- the frozen test-set SHA-256 sidecar;
- the input artifact structure and its link to the test set;
- the input artifact bytes against `ragas-results.json.sha256`;
- the scores artifact bytes against `ragas-scores.json.sha256`;
- the input→scores SHA-256 link (scores were produced from this exact input);
- per-metric coverage of exactly the paired queries for both arms.

## A/B analysis and decision

`benchmarks.haystack.ragas_report` consumes the input and scores artifacts and
produces the evidence-backed B0-versus-refined comparison:

```powershell
uv run python -m benchmarks.haystack.ragas_report `
  --input benchmarks/results/haystack-ragas-v1/ragas-results.json `
  --scores benchmarks/results/haystack-ragas-scores-v1/ragas-scores.json `
  --output-dir benchmarks/results/haystack-ragas-analysis-v1
```

Writes `ragas-report.json` (+ `.sha256` sidecar) and `ragas-report.md` into a
**new** directory. The report includes, per metric:

- aggregate baseline/refined scores and the mean paired delta;
- paired wins/losses/ties (tolerance `1e-9`);
- a deterministic paired bootstrap 95% confidence interval on the mean delta
  (10 000 resamples, fixed seed 97);
- context size changes (prompt word counts per arm);
- representative wins and representative regressions (up to 3 queries each,
  only actual positive/negative deltas);
- an explicit decision and rationale.

Each metric is classified as `supported_improvement`, `regression`, or
`inconclusive`: an improvement requires a positive mean paired delta, a 95%
CI excluding zero, and more wins than losses; a regression is the mirror
image. The overall decision is `supported_improvement` only when at least one
metric is a supported improvement and none is a regression, `regression` only
when at least one metric regresses and none improves, and `inconclusive`
otherwise. No broader claim is made.

The report embeds the `verify` output as the reproduction record, so the
retained analysis directory (artifacts + report + reproduction record) is
self-validating.

## Execution workflow

The declared experiment is executed in two steps. First a small **pilot**
validates prompts, artifact structure, metric behaviour, and cost:

```powershell
uv run python -m benchmarks.haystack.ragas_eval `
  --output-dir benchmarks/results/haystack-ragas-pilot-v1 `
  --max-queries 3
```

Then the **frozen full run** uses the declared immutable configuration
(`top_n 5`, `top_k 3`, no token limit, B2-L lexical profile,
`deepseek-chat`, temperature 0, 512-token generation limit, the shared prompt
template, and the fixed metrics). The retained artifact set lives at:

```text
benchmarks/results/haystack-ragas-v1/          paired harness artifact
benchmarks/results/haystack-ragas-scores-v1/   Ragas scores + judge reasons
benchmarks/results/haystack-ragas-analysis-v1/ comparison report + decision
```

## Out of scope

- changing or regenerating first-stage retrieval;
- running Ragas in fast CI;
- production deployment;
- any claim of improvement before the completed evaluation is recorded.
