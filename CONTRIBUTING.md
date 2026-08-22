# Contributing to ragrefine

Thanks for considering a contribution. `ragrefine` is an evidence-first
research library: implementation work is only half the job, and any claim
that a technique improves retrieval quality must be backed by recorded,
reproducible evaluation.

## Project orientation

- Read `AI_PRD.md`, `TECHNICAL_HYPOTHESIS.md`, and `TECHNICAL_DESIGN.md`
  before substantial changes; they are the source of truth.
- The core boundary is intentional: `query + retrieved candidates` in,
  `refined candidates + trace` out. Do not expand the core into ingestion,
  vector stores, generation, agents, or orchestration without an explicit
  design change.
- Work is ticket-scoped (Jira `RRF-*`); keep patches narrow and reviewable.
  Do not opportunistically refactor unrelated code.

## Setup

```bash
# Python 3.12+ with uv
uv sync --extra haystack          # base + optional Haystack integration
uv sync --extra ragas --extra beir # evaluation harnesses (optional)
```

## Validation gates

Run all of these before considering work complete:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/
uv run pytest
uv build            # when packaging changes
```

- Do not hide, skip, or weaken failing checks.
- Model downloads and live API calls are never part of the default test run.
- Heavier integration tests use `pytest.importorskip` and mocked seams so
  they are skipped without optional dependencies.

## Evaluation discipline

- Benchmark claims require recorded artifacts: frozen candidate pools, model
  identifiers/revisions, environment metadata, and checksums.
- Never fabricate or silently overwrite benchmark results; machine-readable
  artifacts are the source of truth.
- An experiment family that does not improve measured metrics is rejected —
  that is the expected outcome of the research process, not a failure.
- Do not describe experimental features as superior until evaluation
  demonstrates it.

## Submitting changes

1. Implement the smallest surface that satisfies the ticket/design.
2. Add unit, property, or integration tests that directly exercise the
   change and its invariants.
3. Run the validation gates above.
4. Update `CHANGELOG.md` for user-visible changes.
5. Open a pull request describing what changed, what was measured, and any
   deviations from the design.
