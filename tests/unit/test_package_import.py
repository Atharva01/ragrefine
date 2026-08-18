"""Smoke tests for the package scaffold."""


def test_package_imports() -> None:
    """The installed package can be imported without optional dependencies."""
    import ragrefine

    assert ragrefine.__name__ == "ragrefine"
