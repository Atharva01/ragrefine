# Implementation matrix — v0.1 release candidate

This matrix distinguishes implemented package behaviour from benchmark evidence.
Package code and tests establish that a capability exists; quality claims also
require the recorded immutable benchmark artifacts.

| Stage | Implemented capability | Package modules | Primary tests | Evidence / owner |
| --- | --- | --- | --- | --- |
| B0 | Preserve original order through one immutable candidate pool | `models.py`, `refiner.py`, `results.py` | `test_refiner.py`, `test_models.py` | Frozen SciFact/NFCorpus/FiQA snapshots; RRF-71–74 |
| B1 | Optional CrossEncoder reranking with raw-score provenance | `rerank/base.py`, `rerank/sentence_transformers.py`, `refiner.py` | `test_reranker.py`, `test_refiner.py` | Persisted B1-reference rankings and environments; RRF-31–33, RRF-75–76 |
| B2 | Deterministic lexical and configurable pattern ranking | `ranking/lexical.py`, `ranking/patterns.py`, `query/patterns.py` | `test_lexical.py`, `test_patterns.py`, `test_refiner.py` | B2-L/B2-P artifacts; B2-L retained, B2-P not retained; RRF-39–42, RRF-75 |
| B3 | Independent channel orchestration and optional rank fusion | `config.py`, `_channels.py`, `ranking/rrf.py`, `refiner.py` | `test_refiner.py`, `test_rrf.py` | Frozen B3 reports; no B3 profile retained; RRF-46–49, RRF-71–74 |
| B4 | Exact/near deduplication and greedy rank-preserving Top-K/token selection | `selection/deduplicate.py`, `selection/selector.py`, `selection/tokens.py`, `refiner.py` | `test_deduplicate.py`, `test_context_selector.py`, `test_refiner.py` | B4 context-selection artifacts; quality benefit is not claimed; RRF-56–57, RRF-73 |

## Ownership reconciliation

RRF-71–RRF-78 are the verified reconciliation tickets for the implementation
and documentation state represented here. Earlier RRF-67–RRF-70 remain
historical backlog items: their Jira acceptance criteria have not been
independently revalidated in this release-candidate pass, so this repository
does not change their status or recreate their work.

There is intentionally no `ragrefine.haystack` package or Haystack dependency.
The previous empty placeholder surface is absent. A real packaged integration
is deferred to Epic 6 and must add an explicit optional dependency, adapters,
and integration tests.

## Release state

The package version is `0.1.0rc1`. `0.1.0` is reserved for an intentional Epic
6 release decision after the remaining productization scope is complete.
