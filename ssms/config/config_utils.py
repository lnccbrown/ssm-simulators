"""Utilities for handling generator config with nested structure.

This module provides utilities for working with the nested generator_config
structure.

Nested structure (REQUIRED):
    {
        "pipeline": {"n_parameter_sets": 100, "n_subruns": 10, ...},
        "estimator": {"type": "kde", "bandwidth": 0.1, ...},
        "training": {"mixture_probabilities": [0.8, 0.1, 0.1], ...},
        "simulator": {"delta_t": 0.001, "max_t": 20.0, ...},
        "output": {"folder": "...", "pickle_protocol": 4, ...},
    }

Note: Only nested configs are supported. Flat configs are no longer accepted.
"""

from typing import Any


def get_nested_config(config: dict, section: str, key: str, default: Any = None) -> Any:
    """Get a value from nested config structure.

    Args:
        config: Generator configuration dictionary (must use nested structure)
        section: Nested section name ("pipeline", "estimator", "training", "simulator", "output")
        key: Key name within the section
        default: Default value if key not found

    Returns:
        Value from nested structure if available, else default

    Examples:
        >>> config = {"pipeline": {"n_parameter_sets": 100}}
        >>> get_nested_config(config, "pipeline", "n_parameter_sets")
        100

        >>> get_nested_config(config, "pipeline", "missing_key", default=42)
        42
    """
    # Get value from nested structure
    if section in config and isinstance(config[section], dict):
        if key in config[section]:
            return config[section][key]

    # Return default if not found
    return default


def has_nested_structure(config: dict) -> bool:
    """Check if config uses the required nested structure.

    Args:
        config: Generator configuration dictionary

    Returns:
        True if config has nested sections (required format), False otherwise

    Note:
        Only nested configs are supported. This function validates that configs
        have the correct structure with at least one of the required sections.
    """
    nested_sections = {"pipeline", "estimator", "training", "simulator", "output"}
    return any(section in config for section in nested_sections)
