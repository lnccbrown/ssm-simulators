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
    parser.addoption(
        "--run-statistical",
        action="store_true",
        default=False,
        help="Run statistical equivalence tests (skipped by default, ~30s)",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "notebooks: mark test as a notebook test (skipped unless --run-notebooks is passed)",
    )
    config.addinivalue_line(
        "markers",
        "statistical: mark test as a statistical equivalence test (skipped unless --run-statistical is passed)",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (large sample sizes, may take several seconds)",
    )
    config.addinivalue_line(
        "markers",
        "rng_validation: mark test as an RNG validation test",
    )


def pytest_collection_modifyitems(config, items):
    """Skip notebook and statistical tests unless the corresponding flag is passed."""
    run_notebooks = config.getoption("--run-notebooks")
    run_statistical = config.getoption("--run-statistical")

    skip_notebooks = pytest.mark.skip(reason="need --run-notebooks option to run")
    skip_statistical = pytest.mark.skip(reason="need --run-statistical option to run")

    for item in items:
        if not run_notebooks and "notebooks" in item.keywords:
            item.add_marker(skip_notebooks)
        if not run_statistical and "statistical" in item.keywords:
            item.add_marker(skip_statistical)
