"""Opt-in DeepSeek/Haystack generator smoke test over frozen retrieved context."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Protocol

from haystack import Document
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from integrations.haystack.component import RagRefineComponent
from integrations.haystack.fixture import FrozenHaystackFixture
from integrations.haystack.prompt import prompt_builder

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


class ChatGenerator(Protocol):
    """Small test seam for the real Haystack chat generator."""

    def run(self, messages: list[ChatMessage]) -> dict[str, list[ChatMessage]]: ...


def run(
    output_dir: Path,
    *,
    api_key: str,
    model: str,
    generator: ChatGenerator | None = None,
) -> dict[str, object]:
    """Run one baseline/refined smoke comparison with identical generator settings."""
    if output_dir.exists():
        raise FileExistsError(f"generator smoke output already exists: {output_dir}")
    active_generator = generator or OpenAIChatGenerator(
        api_key=Secret.from_token(api_key),
        model=model,
        api_base_url=DEEPSEEK_BASE_URL,
        generation_kwargs={"temperature": 0},
        max_retries=0,
    )
    fixture = FrozenHaystackFixture(top_n=2)
    query = fixture.frozen_query("python")
    pool = fixture.retrieve(query)
    refined = RagRefineComponent(top_k=1).run(query, list(pool))
    results = {
        "baseline": _generate(active_generator, query, list(pool)),
        "refined": _generate(active_generator, query, refined["documents"]),
    }
    result: dict[str, object] = {
        "schema_version": "1.0",
        "model": model,
        "api_base_url": DEEPSEEK_BASE_URL,
        "generation_settings": {"temperature": 0},
        "query": query,
        "results": results,
        "refinement_trace": refined["trace"],
        "interpretation": (
            "Smoke integration only; this does not establish generation-quality "
            "improvement."
        ),
    }
    output_dir.mkdir(parents=True)
    (output_dir / "generator-smoke.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _generate(
    generator: ChatGenerator, query: str, documents: list[Document]
) -> dict[str, object]:
    prompt = prompt_builder().run(query=query, documents=documents)["prompt"]
    started = perf_counter()
    reply = generator.run(messages=[ChatMessage.from_user(prompt)])["replies"][0]
    return {
        "context": prompt,
        "answer": reply.text,
        "source_ids": [str(document.id) for document in documents],
        "latency_seconds": perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    )
    args = parser.parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        parser.error("DEEPSEEK_API_KEY must be set externally")
    print(
        json.dumps(
            run(args.output_dir, api_key=api_key, model=args.model), sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
