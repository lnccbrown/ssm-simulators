"""Shared fixtures for the MLBA tests: the declared model specification."""

from __future__ import annotations

import pytest
from mlba_reference import MLBA_SPEC


@pytest.fixture(scope="session")
def declarations() -> dict:
    return MLBA_SPEC


@pytest.fixture(scope="session")
def declared_models(declarations) -> dict:
    return {model["name"]: model for model in declarations["models"]}
