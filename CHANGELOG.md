# Changelog

All notable changes to `ragrefine` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/).

## [0.2.1] - 2026-09-12

### Added

- PyPI packaging metadata (`license`, `authors`, `classifiers`,
  `project.urls`) and a `py.typed` marker for downstream type checkers.
- `release.yml`: trusted-publishing GitHub Actions workflow that builds
  and publishes to PyPI on `v*.*.*` tag pushes.

## [0.2.0] - 2026-08-22

### Added

- Serializable refiner profiles: `RagRefineComponent` round-trips through
  `to_dict`/`from_dict` and Haystack pipeline YAML for the built-in `b0` and
  `b2-l-lexical` profiles; custom profiles via `register_refiner_factory`.
- `build_rag_pipeline`: end-to-end Haystack pipeline wiring (BM25 retriever →
  `RagRefineComponent` → prompt builder → chat generator), with a prompt-only
  mode when no API key is configured, and the `ChatPromptAdapter` helper.
- `docs/rag-pipeline-guide.md` (pipeline recipes, profiles, honest claims) and
  `docs/v0.2-release.md` (closure record).
- Full packaged integration under `ragrefine.integrations.haystack` (modular
  adapter, component, fixture, prompt, pipeline).

### Fixed

- `uv.lock` regenerated for the 0.2.0 version bump (CI `uv sync --locked`).
- Packaged Haystack integration synced with the checkout implementation and
  made mypy-strict clean.
- CI haystack wheel check now exercises `build_rag_pipeline` and profile YAML
  round-trips from the installed wheel.

### Evaluation

- Executed the frozen Ragas A/B (RRF-97): 13 paired queries, B2-L lexical
  profile, `deepseek-chat`. `faithfulness` 0.730 → 0.808 and
  `factual_correctness` 0.668 → 0.718 directionally favour the refined arm,
  but paired bootstrap 95% CIs include zero → **inconclusive**; no
  generation-quality improvement is claimed. Artifacts retained under
  `benchmarks/results/haystack-ragas-*/` with SHA-256 sidecars.

## [0.1.0] - 2026-08-21

### Added

- Core candidate domain contracts (`Candidate`, `CandidateSet`,
  `RefinedCandidate`) and the `Refiner` API with independent ranking
  channels (original, neural, lexical, pattern).
- Reciprocal Rank Fusion, exact/near deduplication, and rank-preserving
  Top-K/token context selection with structured tracing.
- Optional SentenceTransformers CrossEncoder reranking and optional Haystack
  integration (adapter, `RagRefineComponent`, prompt builder, frozen
  retrieval fixture).
- Frozen BEIR benchmark harness (SciFact, NFCorpus, FiQA), the curated
  hard-negative diagnostic set, and the v0.1 evaluation contract.
- Optional dependency groups: `[beir]`, `[rerank]`, `[haystack]`, `[ragas]`,
  `[dev]`.

### Evaluation

- B1-reference (MiniLM-L6/CUDA) and B1-light (TinyBERT-L2) retained over the
  frozen B0 baseline; B2-L (lexical/CPU) retained; B2-P (patterns) and all B3
  fusion profiles evaluated and **not retained** (measured evidence did not
  support a general improvement claim).
