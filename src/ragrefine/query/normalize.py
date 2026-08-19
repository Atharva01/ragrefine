"""Deterministic, dependency-free text normalization for query analysis."""

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[^\W_]+(?:[-_][^\W_]+)*", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """Return a NFKC-normalized, case-folded, whitespace-normalized string.

    The returned value is suitable for deterministic comparison.  It does not
    stem, lemmatize, remove stop words, rewrite text, or mutate the caller's
    original query/candidate text.
    """
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def normalized_terms(text: str) -> tuple[str, ...]:
    """Extract unique normalized lexical terms in their first-seen order.

    Hyphenated and underscored technical identifiers remain one term.  The
    uniqueness matches lexical coverage's set-based definition while ordering
    makes evidence deterministic and human-readable.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for match in _TOKEN.finditer(normalize_text(text)):
        term = match.group().replace("_", "-")
        if term not in seen:
            terms.append(term)
            seen.add(term)
    return tuple(terms)
