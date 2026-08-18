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

## Evaluate and reproduce

```powershell
uv run --extra beir python -m benchmarks.beir.run evaluate --snapshot benchmarks/results/scifact-b0/snapshot.json --output-dir benchmarks/results/scifact-b0
uv run --extra beir python -m benchmarks.beir.run reproduce --snapshot benchmarks/results/scifact-b0/snapshot.json --output-dir benchmarks/results/scifact-b0
```

Evaluation writes machine-readable `config.json`, `metrics.json`, and
`environment.json`. Metrics are nDCG@5, MRR, Precision@5, Recall@5, and the
candidate-pool Recall@N ceiling. `reproduce` evaluates the same snapshot twice
and fails if the metrics differ.

These scripts establish a B0 retrieval baseline only. They do not apply, or
claim benefit from, any ragrefine algorithm.
