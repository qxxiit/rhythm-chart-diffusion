"""Smoke tests — just verify the package imports."""


def test_import_src() -> None:
    """src/ package imports without errors."""
    import src  # noqa: F401


def test_python_version() -> None:
    """We're running Python 3.10+."""
    import sys
    assert sys.version_info >= (3, 10), f"Python 3.10+ required, got {sys.version_info}"
