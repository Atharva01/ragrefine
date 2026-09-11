# ragrefine

**Evidence-first post-retrieval refinement for RAG systems.**

[![CI](https://github.com/Atharva01/ragrefine/actions/workflows/ci.yml/badge.svg)](https://github.com/Atharva01/ragrefine/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ragrefine.svg)](https://pypi.org/project/ragrefine/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`ragrefine` improves an existing retriever's candidate set before it reaches an
LLM. It reranks, fuses, deduplicates, and selects context while preserving the
original evidence and recording an auditable trace.

```text
retriever -> Top-N candidates -> ragrefine -> refined Top-K + trace -> LLM
```

The core library is framework- and retriever-agnostic. Haystack support is
available as an optional integration.

## Core deliverables

| Capability | Delivered behaviour |
|---|---|
| Candidate contract | Immutable candidates retain IDs, text, metadata, retrieval rank, and retrieval score |
| Ranking channels | Independent original, neural, lexical, and configurable pattern rankings over the same candidate pool |
| Rank fusion | Deterministic reciprocal-rank fusion without mixing incompatible raw-score scales |
| Context refinement | Exact/near-duplicate suppression and rank-preserving Top-K/token-budget selection |
| Traceability | Per-channel evidence, fusion contributions, stage timings, configuration, and explicit failure records |
| Neural reranking | Optional SentenceTransformers CrossEncoder adapter with batched inference |
| Haystack integration | Serializable component profiles and a retriever-to-generator pipeline builder |
| Evaluation | Frozen BEIR candidate pools, checksummed artifacts, reproducibility commands, and a paired Ragas harness |

Output candidates always originate from the supplied input pool; `ragrefine`
does not invent or rewrite evidence.

## Quick start

Python 3.12+ is required.

```bash
pip install ragrefine
```

Or, from this checkout:

```bash
uv sync
```

```python
from ragrefine import Candidate, CandidateSet, Refiner

candidates = CandidateSet(
    name="my-retriever",
    candidates=(
        Candidate(
            id="doc-1",
            text="Python 3.12 improves f-string parsing.",
            retrieval_rank=1,
            retrieval_score=0.91,
        ),
        Candidate(
            id="doc-2",
            text="Python 3.11 introduced exception groups.",
            retrieval_rank=2,
            retrieval_score=0.84,
        ),
    ),
)

result = Refiner().refine(
    query="Which Python version improves f-string parsing?",
    candidate_set=candidates,
    top_k=1,
)

print(result.candidates[0].candidate.id)
print(result.trace)
```

`Refiner()` preserves the original ranking. Explicit `RefinerConfig` settings
enable lexical, pattern, neural, or fused profiles. See the
[technical design](TECHNICAL_DESIGN.md) for the configuration contract.

## Haystack integration

```bash
uv sync --extra haystack
```

The optional integration provides document conversion, a serializable
`RagRefineComponent`, registered refiner profiles, prompt construction, and a
complete Haystack pipeline builder. The [RAG pipeline guide](docs/rag-pipeline-guide.md)
contains the runnable recipe and customization points.

## Evaluation summary

Post-retrieval comparisons reuse the same frozen first-stage Top-50 candidates.
Retrieval is not regenerated between profiles, and candidate-pool Recall@50 is
a fixed ceiling rather than a refinement gain.

| Experiment | SciFact result | Decision |
|---|---:|---|
| B0 — frozen dense baseline | nDCG@5 0.4592 | Baseline |
| B1-reference — MiniLM-L6/CUDA | nDCG@5 **0.6262** | Retained neural profile |
| B1-light — TinyBERT-L2/CPU | nDCG@5 **0.6112** | Retained measured profile |
| B2-L — lexical/CPU | nDCG@5 **0.5432** | Retained lightweight profile |
| B2-P — patterns/CPU | nDCG@5 0.4559 | Not retained |
| B3 — equal-weight fusion | Best profile nDCG@5 0.5715 | Not retained; below B1-reference |

The end-to-end Ragas A/B used 13 paired DeepSeek queries. It is statistically
inconclusive: the positive directional score differences have 95% paired
bootstrap confidence intervals spanning zero, and both context metrics are
saturated at 1.0. The run establishes neither an end-to-end benefit nor a
regression. See the [retained A/B report](benchmarks/results/haystack-ragas-analysis-v1/ragas-report.md).

Machine-readable artifacts are the source of truth. No experiment is described
as improved unless its recorded measurements support that claim.

## Documentation

Detailed material removed from this README remains in the repository's focused
documents:

- [Product requirements](AI_PRD.md) — supported scope and product boundary
- [Technical hypotheses](TECHNICAL_HYPOTHESIS.md) — testable claims and current evidence
- [Technical design](TECHNICAL_DESIGN.md) — architecture, configuration, selection, and tracing contracts
- [Evaluation contract](docs/evaluation-contract-v0.1.md) — frozen-pool rules, metrics, profiles, and reporting policy
- [Benchmark guide](docs/benchmark-baseline.md) — execution and reproduction commands
- [Retained artifact manifest](benchmarks/artifacts/README.md) — checksummed benchmark inputs and verification
- [Haystack pipeline guide](docs/rag-pipeline-guide.md) — integration and serialization
- [Ragas evaluation guide](docs/haystack-ragas-evaluation.md) — paired A/B design and analysis
- [v0.2 release record](docs/v0.2-release.md) — delivered scope, validation, and limitations

## Scope

`ragrefine` owns post-retrieval candidate refinement, context selection, and
tracing. It deliberately does not own ingestion, parsing, chunking, embeddings,
vector databases, first-stage retrieval, LLM generation, agents, or deployment.

## Development

```bash
uv sync --extra haystack --extra ragas
uv run ruff check .
uv run ruff format --check .
uv run mypy src/
uv run pytest
uv build
```

Heavy model and benchmark dependencies remain optional so the core package has
no runtime dependencies.

## License

[MIT](LICENSE)
