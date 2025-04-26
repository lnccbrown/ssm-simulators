"""Configuration module for SSM simulators.

This module provides access to model configurations, boundary and drift function
configurations, and various generator configurations used throughout the SSMS package.
It centralizes all configuration-related functionality to ensure consistent
parameter settings across simulations.
"""

from .config import (
    boundary_config_to_function_params,
    model_config,
)

from .generator_config.data_generator_config import (
    get_lan_config,
    get_opn_only_config,
    get_cpn_only_config,
    get_kde_simulation_filters,
    get_defective_detector_config,
)

from ._modelconfig.base import boundary_config, drift_config
from .kde_constants import KDE_NO_DISPLACE_T  # noqa: F401

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
]
