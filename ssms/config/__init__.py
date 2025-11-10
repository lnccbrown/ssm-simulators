"""Configuration module for SSM simulators.

This module provides access to model configurations, boundary and drift function
configurations, and various generator configurations used throughout the SSMS package.
It centralizes all configuration-related functionality to ensure consistent
parameter settings across simulations.
"""

import copy

from ssms.config._modelconfig import get_model_config

from ._modelconfig.base import boundary_config, drift_config
from .generator_config.data_generator_config import (
    data_generator_config,  # TODO: remove from interface in v1.0.0
    get_cpn_only_config,
    get_default_generator_config,
    get_defective_detector_config,
    get_kde_simulation_filters,
    get_lan_config,
    get_opn_only_config,
    get_ratio_estimator_config,
)
from .kde_constants import KDE_NO_DISPLACE_T  # noqa: F401
from .config_builder import ConfigBuilder


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
    "model_config",
    "boundary_config",
    "drift_config",
    "boundary_config_to_function_params",
    "get_lan_config",
    "get_opn_only_config",
    "get_cpn_only_config",
    "get_kde_simulation_filters",
    "get_defective_detector_config",
    "get_ratio_estimator_config",
    "get_default_generator_config",
    "data_generator_config",  # TODO: remove from interface in v1.0.0
    "ConfigBuilder",
]
