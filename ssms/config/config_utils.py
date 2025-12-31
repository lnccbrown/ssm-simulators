"""Utilities for handling generator config with nested structure.

This module provides utilities for working with the nested generator_config
structure and converting legacy flat configs to the new format.

Nested structure (REQUIRED):
    {
        "pipeline": {"n_parameter_sets": 100, "n_subruns": 10, ...},
        "estimator": {"type": "kde", "bandwidth": 0.1, ...},
        "training": {"mixture_probabilities": [0.8, 0.1, 0.1], ...},
        "simulator": {"delta_t": 0.001, "max_t": 20.0, ...},
        "output": {"folder": "...", "pickle_protocol": 4, ...},
    }

Note: Flat configs are no longer supported. Use convert_flat_to_nested()
to migrate legacy configs.
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
        Flat configs are no longer supported. This function is used to validate
        that configs have the correct nested structure.
    """
    nested_sections = {"pipeline", "estimator", "training", "simulator", "output"}
    return any(section in config for section in nested_sections)


def convert_flat_to_nested(flat_config: dict) -> dict:
    """Convert a flat config structure to nested structure.

    Args:
        flat_config: Generator configuration using flat structure

    Returns:
        Generator configuration using nested structure

    Example:
        >>> flat = {"n_parameter_sets": 100, "estimator_type": "kde"}
        >>> nested = convert_flat_to_nested(flat)
        >>> nested
        {"pipeline": {"n_parameter_sets": 100}, "estimator": {"type": "kde"}}
    """
    nested = {
        "pipeline": {},
        "estimator": {},
        "training": {},
        "simulator": {},
        "output": {},
    }

    # Pipeline settings
    pipeline_keys = [
        "n_parameter_sets",
        "n_subruns",
        "n_cpus",
        "n_parameter_sets_rejected",
    ]
    for key in pipeline_keys:
        if key in flat_config:
            nested["pipeline"][key] = flat_config[key]

    # Estimator settings
    if "estimator_type" in flat_config:
        nested["estimator"]["type"] = flat_config["estimator_type"]
    if "kde_bandwidth" in flat_config:
        nested["estimator"]["bandwidth"] = flat_config["kde_bandwidth"]
    if "kde_displace_t" in flat_config:
        nested["estimator"]["displace_t"] = flat_config["kde_displace_t"]
    if "use_pyddm_pdf" in flat_config:
        nested["estimator"]["use_pyddm_pdf"] = flat_config["use_pyddm_pdf"]

    # Training settings
    # Handle both kde_data_mixture_probabilities and data_mixture_probabilities
    if "data_mixture_probabilities" in flat_config:
        nested["training"]["mixture_probabilities"] = flat_config[
            "data_mixture_probabilities"
        ]
    elif "data_mixture_probabilities" in flat_config:
        nested["training"]["mixture_probabilities"] = flat_config[
            "data_mixture_probabilities"
        ]

    if "n_training_samples_by_parameter_set" in flat_config:
        nested["training"]["n_samples_per_param"] = flat_config[
            "n_training_samples_by_parameter_set"
        ]
    if "separate_response_channels" in flat_config:
        nested["training"]["separate_response_channels"] = flat_config[
            "separate_response_channels"
        ]
    if "negative_rt_log_likelihood" in flat_config:
        nested["training"]["negative_rt_log_likelihood"] = flat_config[
            "negative_rt_log_likelihood"
        ]
    # Backward compatibility: support old name
    if "negative_rt_cutoff" in flat_config:
        nested["training"]["negative_rt_log_likelihood"] = flat_config[
            "negative_rt_cutoff"
        ]

    # Additional training settings
    if "n_subdatasets" in flat_config:
        nested["training"]["n_subdatasets"] = flat_config["n_subdatasets"]
    if "n_trials_per_dataset" in flat_config:
        nested["training"]["n_trials_per_dataset"] = flat_config["n_trials_per_dataset"]

    # Simulator settings
    simulator_keys = ["delta_t", "max_t", "n_samples", "smooth_unif"]
    for key in simulator_keys:
        if key in flat_config:
            nested["simulator"][key] = flat_config[key]

    # Simulation filters
    if "simulation_filters" in flat_config:
        nested["simulator"]["filters"] = flat_config["simulation_filters"]

    # Output settings
    if "output_folder" in flat_config:
        nested["output"]["folder"] = flat_config["output_folder"]
    if "pickleprotocol" in flat_config:
        nested["output"]["pickle_protocol"] = flat_config["pickleprotocol"]
    if "nbins" in flat_config:
        nested["output"]["nbins"] = flat_config["nbins"]

    # Model name (keep at top level)
    if "model" in flat_config:
        nested["model"] = flat_config["model"]

    # Bin settings (keep at top level for now)
    if "bin_pointwise" in flat_config:
        nested["bin_pointwise"] = flat_config["bin_pointwise"]

    return nested
