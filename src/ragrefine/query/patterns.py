"""Configurable, deterministic extraction of structured text patterns."""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ragrefine.errors import PatternError
from ragrefine.query.normalize import normalize_text

ValueNormalizer = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class PatternRule:
    """One user-configured regex rule for extracting a structured constraint."""

    label: str
    pattern: str
    normalizer: ValueNormalizer | None = None

    def __post_init__(self) -> None:
        """Validate labels and expressions before the rule can be used."""
        if not isinstance(self.label, str):
            raise PatternError("pattern rule label must be str")
        if not isinstance(self.pattern, str):
            raise PatternError("pattern rule must be a regular-expression string")
        if self.normalizer is not None and not callable(self.normalizer):
            raise PatternError("pattern rule normalizer must be callable")
        if not self.label.strip():
            raise PatternError("pattern rule label must not be empty")
        try:
            compiled = re.compile(self.pattern)
        except re.error as error:
            raise PatternError(f"invalid pattern for label {self.label!r}") from error
        if compiled.search("") is not None:
            raise PatternError(
                f"pattern for label {self.label!r} must not match empty text"
            )


@dataclass(frozen=True, slots=True)
class PatternMatch:
    """One exact pattern occurrence, retaining its original source span."""

    label: str
    value: str
    normalized_value: str
    start: int
    end: int


@dataclass(frozen=True, slots=True, init=False)
class PatternRegistry:
    """An immutable registry that extracts patterns in a stable source order."""

    _rules: tuple[PatternRule, ...]

    def __init__(self, rules: Sequence[PatternRule] = ()) -> None:
        configured_rules = tuple(rules)
        seen_rules: set[tuple[str, str]] = set()
        for rule in configured_rules:
            if not isinstance(rule, PatternRule):
                raise PatternError(
                    "pattern registry rules must be PatternRule instances"
                )
            rule_key = (rule.label, rule.pattern)
            if rule_key in seen_rules:
                raise PatternError(
                    "pattern registry contains duplicate label/pattern rules"
                )
            seen_rules.add(rule_key)
        object.__setattr__(self, "_rules", configured_rules)

    @property
    def rules(self) -> tuple[PatternRule, ...]:
        """Return the configured rules in their deterministic registry order."""
        return self._rules

    def extract(self, text: str) -> tuple[PatternMatch, ...]:
        """Extract all rule matches while preserving source text values and spans."""
        matches: list[tuple[int, int, int, int, PatternMatch]] = []
        for rule_index, rule in enumerate(self._rules):
            for match_index, match in enumerate(re.finditer(rule.pattern, text)):
                value = match.group()
                normalized_value = self._normalize_value(rule, value)
                matches.append(
                    (
                        match.start(),
                        match.end(),
                        rule_index,
                        match_index,
                        PatternMatch(
                            label=rule.label,
                            value=value,
                            normalized_value=normalized_value,
                            start=match.start(),
                            end=match.end(),
                        ),
                    )
                )
        matches.sort(key=lambda item: item[:4])
        return tuple(item[4] for item in matches)

    @staticmethod
    def _normalize_value(rule: PatternRule, value: str) -> str:
        normalizer = rule.normalizer or normalize_text
        try:
            normalized_value = normalizer(value)
        except Exception as error:
            raise PatternError(
                f"value normalization failed for pattern label {rule.label!r}"
            ) from error
        if not isinstance(normalized_value, str):
            raise PatternError(
                f"value normalizer for pattern label {rule.label!r} must return str"
            )
        return normalized_value


def example_pattern_rules() -> tuple[PatternRule, ...]:
    """Return opt-in representative rules for common structured constraints.

    Applications choose whether to use or replace these examples; core pattern
    ranking never assumes an application-specific taxonomy.
    """
    return (
        PatternRule("version", r"(?i)\bv?\d+(?:\.\d+){1,3}\b"),
        PatternRule("code", r"(?i)\b(?:http[-_ ]?)?\d{3}\b"),
        PatternRule("date", r"\b\d{4}-\d{2}-\d{2}\b"),
        PatternRule("percentage", r"\b\d+(?:\.\d+)?\s*%"),
        PatternRule("number", r"\b\d+(?:\.\d+)?\b"),
    )
