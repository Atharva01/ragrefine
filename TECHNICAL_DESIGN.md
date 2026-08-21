# ragrefine v0.1 technical design

## Core contract

```text
external retrieval -> one immutable CandidateSet -> Refiner -> RefinementResult + trace
```

Candidates preserve ID, text, metadata, retrieval rank, and retrieval score.
Output candidates are always derived from the supplied pool. Callers merge
multiple retrieval sources upstream; `Refiner` deliberately accepts one
`CandidateSet`.

## Refinement flow

```text
same input pool
  ├─ original order
  ├─ optional neural ranker
  ├─ optional lexical ranker
  └─ optional pattern ranker
          ↓
 one successful channel: direct ranking
 multiple successful channels: optional RRF
          ↓
 deduplication -> greedy rank-preserving Top-K/token-budget selection -> trace
```

Each enabled channel ranks the complete input pool. Required channel failures
abort; optional failures are recorded and omitted. Multiple successful channels
without configured fusion are an explicit configuration error. RRF operates on
ranks, not heterogeneous raw scores, with deterministic tie-breaking.

## Public configuration

`ChannelConfig(enabled, required, weight)` controls each channel.
`RefinerConfig` declares original/neural/lexical/pattern channels and whether
fusion is enabled. Existing constructor forms remain additive B0/B1-compatible
shortcuts. Optional heavyweight inference remains under `ragrefine[rerank]`.

## Selection and trace

Deduplication never rewrites text and keeps the highest-ranked representative.
After final ranking, selection greedily keeps candidates in rank order until
`top_k` or the token budget is reached. The trace records channel evidence,
fusion contributions, stage timing, configuration fingerprint, selection
records, and optional-channel failures.

## Evaluation boundary

BEIR and benchmark scripts live outside the package and import the package
implementations; they do not contain alternate refinement algorithms. The
artifact policy and reproducibility commands are documented in
`benchmarks/artifacts/README.md` and `docs/benchmark-baseline.md`.
