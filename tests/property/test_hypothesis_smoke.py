"""Minimal Hypothesis execution coverage for the project test configuration."""

from hypothesis import given
from hypothesis import strategies as st


@given(st.lists(st.text(), max_size=10))
def test_hypothesis_generates_examples(values: list[str]) -> None:
    """Hypothesis-generated examples execute in the fast property-test suite."""
    assert tuple(values) == tuple(values)
