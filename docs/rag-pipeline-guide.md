# Building a RAG pipeline with ragrefine and Haystack

This guide shows how to build a real, serializable RAG pipeline with
`ragrefine` sitting between retrieval and generation. It uses the optional
Haystack integration (`uv sync --extra haystack` or
`uv pip install "ragrefine[haystack]"`).

## Pipeline shape

```text
InMemoryBM25Retriever -> RagRefineComponent -> PromptBuilder -> ChatPromptAdapter
                                                                  -> OpenAIChatGenerator
```

Refinement is strictly post-retrieval: `RagRefineComponent` receives the
retrieved `documents` list and returns the same `Document` objects re-ranked
and trimmed to the final context budget. It never touches indexing, retrieval,
document text, or generation.

## Quick start

```python
from haystack.document_stores.in_memory import InMemoryDocumentStore
from ragrefine.integrations.haystack import build_rag_pipeline, demo_documents

store = InMemoryDocumentStore(bm25_algorithm="BM25L")
store.write_documents(demo_documents())  # frozen demo corpus; use your own docs

pipeline = build_rag_pipeline(store, top_k=3, profile="b2-l-lexical")

query = "Which Python version improves f-string parsing?"
result = pipeline.run(
    {
        "retriever": {"query": query},
        "refine": {"query": query},
        "prompt": {"query": query},
    },
    include_outputs_from={"refine", "prompt"},
)
answer = result["generator"]["replies"][0].text
trace = result["refine"]["trace"]
```

With `DEEPSEEK_API_KEY` (and optional `DEEPSEEK_MODEL`) set, the pipeline ends
in a DeepSeek chat generator (OpenAI-compatible, temperature 0). Without a
key, the pipeline stops after the prompt builder and returns the prompt.

## Refiner profiles

`RagRefineComponent` serializes a **named profile**, so the whole pipeline
round-trips through `Pipeline.dumps()` / `Pipeline.loads()` (YAML):

| Profile | Behaviour | Evidence |
| --- | --- | --- |
| `b0` (default) | Original retrieval order — refinement is a no-op | B0 baseline |
| `b2-l-lexical` | Stdlib lexical-coverage ranking (CPU, no dependencies) | Retained B2-L profile (SciFact) |

Register a custom profile to make your own refiner serializable:

```python
from ragrefine import Refiner, RefinerConfig
from ragrefine.config import ChannelConfig
from ragrefine.ranking.lexical import LexicalRanker
from ragrefine.integrations.haystack import register_refiner_factory

register_refiner_factory(
    "my-lexical",
    lambda: Refiner(
        lexical_ranker=LexicalRanker(),
        config=RefinerConfig(
            original=ChannelConfig(enabled=False),
            lexical=ChannelConfig(enabled=True),
        ),
    ),
)
pipeline = build_rag_pipeline(store, profile="my-lexical")
```

The factory must be registered before the component (or pipeline YAML) is
loaded. An unregistered custom refiner fails loudly on serialization rather
than silently changing behaviour.

## Using your own store, retriever, and generator

`build_rag_pipeline` wires `InMemoryBM25Retriever` over the store you pass in;
swap in any retriever by building the pipeline yourself:

```python
from haystack import Pipeline
from ragrefine.integrations.haystack import (
    RagRefineComponent,
    prompt_builder,
    refiner_for_profile,
)

pipeline = Pipeline()
pipeline.add_component("retriever", your_retriever)
pipeline.add_component(
    "refine",
    RagRefineComponent(top_k=3, refiner=refiner_for_profile("b2-l-lexical")),
)
pipeline.add_component("prompt", prompt_builder())
pipeline.connect("retriever.documents", "refine.documents")
pipeline.connect("refine.documents", "prompt.documents")
```

For a non-chat generator, connect `prompt.prompt` directly to its `prompt`
input; for a chat generator, wrap the prompt with the provided
`ChatPromptAdapter` (`prompt.prompt -> chat.prompt -> generator.messages`).

## What refinement does and does not claim

- It re-ranks and trims the retrieved pool under `top_k` (and optionally
  `max_tokens`), always returning original documents with their IDs and
  metadata preserved; it never invents evidence.
- Measured retrieval evidence retains the **B1-reference** (neural, CUDA) and
  **B2-L** (lexical, CPU) profiles on the frozen BEIR harness. B3 fusion
  profiles were evaluated and rejected.
- The end-to-end **Ragas A/B is inconclusive**: on the frozen 13-query set,
  the refined arm scored directionally higher on faithfulness and factual
  correctness but the paired bootstrap confidence intervals include zero, so
  **no generation-quality improvement is claimed**. See
  `benchmarks/results/haystack-ragas-analysis-v1/ragas-report.md` and
  [docs/haystack-ragas-evaluation.md](haystack-ragas-evaluation.md).

## Serialization and reproduction

Every retained artifact carries a SHA-256 sidecar. `ragas_score verify`
validates the checksum chain (test set → harness artifact → scores artifact →
report). See `docs/v0.2-release.md` for the completion record.
