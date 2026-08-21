"""Prompt construction helpers for the optional Haystack integration."""

from haystack.components.builders import PromptBuilder

PROMPT_TEMPLATE = """Question: {{ query }}
Context:
{% for document in documents %}
[{{ document.id }}] {{ document.content }} | {{ document.meta }}
{% endfor %}"""


def prompt_builder() -> PromptBuilder:
    """Create the fixed prompt builder shared by bypass and refinement paths."""
    return PromptBuilder(template=PROMPT_TEMPLATE)
