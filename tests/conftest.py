"""Pytest configuration for ssm-simulators test suite."""

import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-notebooks",
        action="store_true",
        default=False,
        help="Run notebook tests (skipped by default)",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "notebooks: mark test as a notebook test (skipped unless --run-notebooks is passed)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip notebook tests unless --run-notebooks is passed."""
    if config.getoption("--run-notebooks"):
        # --run-notebooks given: don't skip notebook tests
        return

    skip_notebooks = pytest.mark.skip(reason="need --run-notebooks option to run")
    for item in items:
        if "notebooks" in item.keywords:
            item.add_marker(skip_notebooks)
