"""Tests for opt-in generator-smoke recording without external credentials."""

from pathlib import Path

import pytest

haystack = pytest.importorskip("haystack")
ChatMessage = haystack.dataclasses.ChatMessage

from benchmarks.haystack.generator_smoke import run  # noqa: E402


class _FakeGenerator:
    def run(self, messages: list[ChatMessage]) -> dict[str, list[ChatMessage]]:
        assert messages[0].text
        return {"replies": [ChatMessage.from_assistant("fixture answer")]}


def test_smoke_records_same_model_settings_context_sources_and_trace(
    tmp_path: Path,
) -> None:
    result = run(
        tmp_path / "smoke",
        api_key="not-used",
        model="deepseek-chat",
        generator=_FakeGenerator(),
    )

    assert result["model"] == "deepseek-chat"
    assert (
        result["results"]["baseline"]["source_ids"]
        != result["results"]["refined"]["source_ids"]
    )
    assert result["results"]["refined"]["answer"] == "fixture answer"
    assert result["refinement_trace"]["stages"]
