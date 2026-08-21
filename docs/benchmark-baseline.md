# Frozen BEIR baseline

The BEIR baseline is benchmark-only. It is not installed by `uv sync` and is
not part of the fast CI workflow.

## Setup

```powershell
uv sync --extra beir
```

The configured datasets are SciFact, NFCorpus, and FiQA. Candidate generation
uses the pinned `sentence-transformers/msmarco-MiniLM-L6-cos-v5` revision in
`benchmarks/beir/config.py` and writes its original Top-50 ranking unchanged.

## Generate a snapshot

```powershell
uv run --extra beir python -m benchmarks.beir.run generate --dataset scifact --output-dir benchmarks/results/scifact-b0
```

This downloads the dataset/model when needed and persists `snapshot.json` plus
its SHA-256 sidecar. The snapshot stores query ID/text and each candidate's
document ID/text, original rank, and retrieval score. Later evaluation reads
this frozen artifact; it never regenerates retrieval.

Candidate generation defaults to the configured CPU device. When a CUDA-enabled
run is required, pass `--device cuda`; the selected device is persisted with the
model, revision, backend, and Top-N configuration in the snapshot and run
metadata.

## Evaluate and reproduce

```powershell
uv run --extra beir python -m benchmarks.beir.run evaluate --snapshot benchmarks/results/scifact-b0/snapshot.json --output-dir benchmarks/results/scifact-b0
uv run --extra beir python -m benchmarks.beir.run reproduce --snapshot benchmarks/results/scifact-b0/snapshot.json --output-dir benchmarks/results/scifact-b0
```

Evaluation writes machine-readable `config.json`, `metrics.json`, and
`environment.json`. Metrics are nDCG@5, MRR@5, Precision@5, Recall@5, and the
candidate-pool Recall@N ceiling. `reproduce` evaluates the same snapshot twice
and fails if the metrics differ.

These scripts establish a B0 retrieval baseline only. They do not apply, or
claim benefit from, any ragrefine algorithm.

## Characterize frozen-pool redundancy

```powershell
uv run python -m benchmarks.beir.analyze_redundancy `
  --snapshot benchmarks/results/scifact-b0/snapshot.json `
  --snapshot benchmarks/results/nfcorpus-b0-v1/snapshot.json `
  --snapshot benchmarks/results/fiqa-b0-v2/snapshot.json `
  --output benchmarks/results/redundancy-audit-v1.json
```

The audit does not retrieve, rerank, or read qrels. It measures exact and
predeclared near-duplicate frequency plus a deterministic word-token proxy;
see `docs/redundancy-audit.md` for the current interpretation.

## Analyze a persisted B1 ranking

The B1 comparison command reads an existing frozen B0 snapshot and B1 ranking;
it does not rerun retrieval or CrossEncoder inference. It verifies the shared
candidate pool, evaluates B1 twice for reproducibility, and writes aggregate and
per-query deltas, representative cases, latency, and the measured decision.

```powershell
uv run python -m benchmarks.beir.analyze_b1 --baseline benchmarks/results/scifact-b0/snapshot.json --ranking benchmarks/results/scifact-b1/b1-ranking.json --environment benchmarks/results/scifact-b1/b1-environment.json --latency benchmarks/results/scifact-b1/b1-latency.json --output benchmarks/results/scifact-b1/b1-analysis.json
```

## Run and reproduce B4 selection ablations

The B4 command consumes the persisted SciFact B1-reference and B2-L rankings.
It validates their checksums and shared frozen B0 provenance, then evaluates the
predeclared S0-S4 policies without retrieval or reranking.

```powershell
uv run python -m benchmarks.beir.b4 run `
  --b1-ranking benchmarks/results/scifact-b1-batched/b1-ranking.json `
  --b1-environment benchmarks/results/scifact-b1-batched/b1-environment.json `
  --b2-ranking benchmarks/results/scifact-b2-lexical/b2-lexical-ranking.json `
  --b2-environment benchmarks/results/scifact-b2-lexical/b2-lexical-environment.json `
  --output-dir benchmarks/results/scifact-b4-selection-v1

uv run python -m benchmarks.beir.b4 reproduce `
  --output-dir benchmarks/results/scifact-b4-selection-v1
```

The output directory contains the frozen configuration, input manifest and
checksums, aggregate summary, per-query selected IDs and exclusions, runtime,
environment metadata, and reproduction result. See
`docs/b4-context-selection.md` for the validated RRF-61 interpretation.

## Evaluate qualified secondary datasets

RRF-62 uses the frozen NFCorpus and FiQA snapshots created by RRF-59. For each
dataset, run only the retained B1-reference and B2-L profiles, then compare the
persisted rankings with B0. B1 uses the pinned MiniLM-L6 revision on CUDA; B2-L
uses the deterministic standard-library lexical ranker on CPU.

```powershell
uv run --extra rerank python -m benchmarks.beir.b1 `
  --snapshot <b0-directory>/snapshot.json `
  --output-dir <b1-reference-directory> `
  --model cross-encoder/ms-marco-MiniLM-L6-v2 `
  --revision 233902d25c440f23af6f7d6e94d2946bac0bee0a `
  --device cuda --batch-size 512 --queries-per-batch 24

uv run python -m benchmarks.beir.b2 `
  --snapshot <b0-directory>/snapshot.json `
  --output-dir <b2-lexical-directory> `
  --signal lexical

uv run python -m benchmarks.beir.secondary run `
  --baseline <b0-directory>/snapshot.json `
  --b1-ranking <b1-reference-directory>/b1-ranking.json `
  --b1-environment <b1-reference-directory>/b1-environment.json `
  --b2-ranking <b2-lexical-directory>/b2-lexical-ranking.json `
  --b2-environment <b2-lexical-directory>/b2-lexical-environment.json `
  --output-dir <secondary-evaluation-directory>

uv run python -m benchmarks.beir.secondary reproduce `
  --output-dir <secondary-evaluation-directory>
```

The analyzer accepts only the qualified snapshot checksums and validates that
both retained profiles preserve the full candidate population. See
`docs/secondary-evaluation.md` for the measured RRF-62 results.

## Analyze retained-profile stability across datasets

RRF-63 compares only the persisted B0, B1-reference, and B2-L rankings for
SciFact, NFCorpus, and FiQA. It verifies all sidecars and provenance first;
it does not run retrieval or a model.

```powershell
uv run python -m benchmarks.beir.cross_dataset run `
  --output-dir benchmarks/results/cross-dataset-stability-v2

uv run python -m benchmarks.beir.cross_dataset reproduce `
  --output-dir benchmarks/results/cross-dataset-stability-v2
```

See `docs/cross-dataset-stability.md` for the measured interpretation. The
result directory contains the source-of-truth summary, input checksums, and
representative per-query improvement/regression cases.

## Reproduce from a clean checkout

The large frozen snapshots and persisted rankings are distributed as the
immutable external `retained-v1` bundle rather than committed source files.
Before running any retained-profile analysis, extract that bundle into
`benchmarks/` and verify every declared file and checksum:

```powershell
uv run python -m benchmarks.beir.artifacts verify `
  --manifest benchmarks/artifacts/retained-v1.json `
  --artifact-root benchmarks

uv run python -m benchmarks.beir.cross_dataset run `
  --output-dir benchmarks/results/cross-dataset-reproduced-v1

uv run python -m benchmarks.beir.cross_dataset reproduce `
  --output-dir benchmarks/results/cross-dataset-reproduced-v1
```

The manifest inventory and exact bundle layout are in
`benchmarks/artifacts/README.md`. Verification and `reproduce` consume only
persisted artifacts; they never regenerate retrieval or run a CrossEncoder.
