"""Configuration module for SSM simulators.

This module provides access to model configurations, boundary and drift function
configurations, and various generator configurations used throughout the SSMS package.
It centralizes all configuration-related functionality to ensure consistent
parameter settings across simulations.
"""

from .config import (  # noqa: D104
    boundary_config,
    boundary_config_to_function_params,
    drift_config,
    kde_simulation_filters,
    model_config,
    DeprecatedDict,
    get_kde_simulation_filters,
    get_opn_only_config,
    get_cpn_only_config,
    get_lan_config,
    get_ratio_estimator_config,
    get_defective_detector_config,
    get_snpe_config,
    get_default_generator_config,
    data_generator_config,
)

from .kde_constants import KDE_NO_DISPLACE_T

__all__ = [
    "DeprecatedDict",
    "KDE_NO_DISPLACE_T",
    "boundary_config",
    "boundary_config_to_function_params",
    "data_generator_config", # TODO: deprecate
    "drift_config",
    "get_cpn_only_config",
    "get_default_generator_config",
    "get_defective_detector_config",
    "get_kde_simulation_filters",
    "get_lan_config",
    "get_opn_only_config",
    "get_ratio_estimator_config",
    "get_snpe_config",
    "model_config"
]
