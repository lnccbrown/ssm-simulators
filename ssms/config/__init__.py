"""Configuration module for SSM simulators.

This module provides access to model configurations, boundary and drift function
configurations, and various generator configurations used throughout the SSMS package.
It centralizes all configuration-related functionality to ensure consistent
parameter settings across simulations.
"""

import copy

# New registry-based imports
from .boundary_registry import register_boundary, get_boundary_registry
from .drift_registry import register_drift, get_drift_registry
from .model_registry import (
    register_model_config,
    register_model_config_factory,
    get_model_registry,
)

# Old imports for backward compatibility
from ssms.config._modelconfig import get_model_config

from .generator_config.data_generator_config import (
    get_default_generator_config,
    get_defective_detector_config,
    get_kde_simulation_filters,
    get_lan_config,  # Deprecated alias for get_lan_kde_config
    get_lan_kde_config,
    get_ratio_estimator_config,
)
from .kde_constants import KDE_NO_DISPLACE_T  # noqa: F401
from .model_config_builder import ModelConfigBuilder


def boundary_config_to_function_params(config: dict) -> dict:
    """
    Convert boundary configuration to function parameters.

    Parameters
    ----------
    config: dict
        Dictionary containing the boundary configuration

    Returns
    -------
    dict
        Dictionary with adjusted key names so that they match function parameters names
        directly.
    """
    return {f"boundary_{k}": v for k, v in config.items()}


class CopyOnAccessDict(dict):
    """A dict that returns a deep copy of the value on lookup."""

    def __getitem__(self, key):
        return copy.deepcopy(super().__getitem__(key))


model_config = CopyOnAccessDict(get_model_config())

__all__ = [
    # Registry functions (new, recommended)
    "register_boundary",
    "get_boundary_registry",
    "register_drift",
    "get_drift_registry",
    "register_model_config",
    "register_model_config_factory",
    "get_model_registry",
    "ModelConfigBuilder",
    # Legacy access (kept for convenience)
    "model_config",
    "boundary_config_to_function_params",
    # Generator configs (get_default_generator_config is the primary API)
    "get_default_generator_config",
    "get_kde_simulation_filters",
    # Approach-specific configs (for advanced users)
    "get_lan_kde_config",
    "get_lan_config",  # Deprecated alias for get_lan_kde_config
    "get_defective_detector_config",
    "get_ratio_estimator_config",
]
