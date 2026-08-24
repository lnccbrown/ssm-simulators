"""Model configuration module for SSM simulators."""

from .ddm import (
    get_ddm_config,
    get_ddm_legacy_config,
)
from .conflict import (
    get_conflict_ds_config,
    get_conflict_ds_angle_config,
    get_conflict_dsstimflex_config,
    get_conflict_dsstimflex_angle_config,
    get_conflict_stimflex_config,
    get_conflict_stimflex_angle_config,
    get_conflict_stimflexrel1_config,
    get_conflict_stimflexrel1_angle_config,
    get_conflict_stimflexrel1_leak_config,
    get_conflict_stimflexrel1_leak2_config,
)
from .full_ddm import (
    get_full_ddm_config,
    get_full_ddm_rv_config,
)
from .lca import (
    get_lca_3_config,
    get_lca_4_config,
    get_lca_no_bias_3_config,
    get_lca_no_bias_4_config,
    get_lca_no_bias_angle_3_config,
    get_lca_no_bias_angle_4_config,
    get_lca_no_z_3_config,
    get_lca_no_z_4_config,
    get_lca_no_z_angle_3_config,
    get_lca_no_z_angle_4_config,
)
from .levy import get_levy_angle_config, get_levy_config
from .tradeoff import (
    get_tradeoff_angle_no_bias_config,
    get_tradeoff_conflict_gamma_no_bias_config,
    get_tradeoff_no_bias_config,
    get_tradeoff_weibull_no_bias_config,
)

from .angle import get_angle_config
from .weibull import get_weibull_config
from .ddm_par2 import (
    get_ddm_par2_angle_no_bias_config,
    get_ddm_par2_config,
    get_ddm_par2_conflict_gamma_no_bias_config,
    get_ddm_par2_no_bias_config,
    get_ddm_par2_weibull_no_bias_config,
)
from .ddm_random import (
    get_ddm_rayleight_config,
    get_ddm_sdv_config,
    get_ddm_st_config,
    get_ddm_truncnormt_config,
)
from .ddm_seq2 import (
    get_ddm_seq2_angle_no_bias_config,
    get_ddm_seq2_config,
    get_ddm_seq2_conflict_gamma_no_bias_config,
    get_ddm_seq2_no_bias_config,
    get_ddm_seq2_weibull_no_bias_config,
)
from .dev_rlwm_lba import (
    get_dev_rlwm_lba_pw_v1_config,
    get_dev_rlwm_lba_race_v1_config,
    get_dev_rlwm_lba_race_v2_config,
)
from .gamma_drift import (
    get_gamma_drift_angle_config,
    get_gamma_drift_config,
)
from .inv_temp_softmax import (
    get_inv_temp_softmax_2_config,
    get_inv_temp_softmax_3_config,
    get_inv_temp_softmax_4_config,
)
from .lba import (
    get_lba2_config,
    get_lba3_config,
    get_lba4_config,
    get_lba_3_vs_constraint_config,
    get_lba_angle_3_config,
    get_lba_angle_3_vs_constraint_config,
)
from .mic2 import (
    get_ddm_mic2_adj_angle_no_bias_config,
    get_ddm_mic2_adj_config,
    get_ddm_mic2_adj_conflict_gamma_no_bias_config,
    get_ddm_mic2_adj_no_bias_config,
    get_ddm_mic2_adj_weibull_no_bias_config,
    get_ddm_mic2_leak_angle_no_bias_config,
    get_ddm_mic2_leak_config,
    get_ddm_mic2_leak_conflict_gamma_no_bias_config,
    get_ddm_mic2_leak_no_bias_config,
    get_ddm_mic2_leak_weibull_no_bias_config,
    get_ddm_mic2_ornstein_angle_no_bias_config,
    get_ddm_mic2_ornstein_config,
    get_ddm_mic2_ornstein_conflict_gamma_no_bias_config,
    get_ddm_mic2_ornstein_no_bias_config,
    get_ddm_mic2_ornstein_weibull_no_bias_config,
)
from .mic2.multinoise import (
    get_ddm_mic2_multinoise_angle_no_bias_config,
    get_ddm_mic2_multinoise_conflict_gamma_no_bias_config,
    get_ddm_mic2_multinoise_no_bias_config,
    get_ddm_mic2_multinoise_weibull_no_bias_config,
)
from .ornstein import (
    get_ornstein_angle_config,
    get_ornstein_config,
)
from .race import (
    get_race_2_config,
    get_race_3_config,
    get_race_4_config,
    get_race_no_bias_2_config,
    get_race_no_bias_3_config,
    get_race_no_bias_4_config,
    get_race_no_bias_angle_2_config,
    get_race_no_bias_angle_3_config,
    get_race_no_bias_angle_4_config,
    get_race_no_z_2_config,
    get_race_no_z_3_config,
    get_race_no_z_4_config,
    get_race_no_z_angle_2_config,
    get_race_no_z_angle_3_config,
    get_race_no_z_angle_4_config,
)
from .racing_diffusion import (
    get_racing_diffusion_3_config,
)
from .poisson_race import get_poisson_race_config
from .shrink import (
    get_shrink_spot_config,
    get_shrink_spot_extended_config,
    get_shrink_spot_simple_config,
    get_shrink_spot_simple_extended_config,
)

from .addm import get_addm_config
from .validation import get_invalid_configs


def _rt_choice(config: dict) -> dict:
    """Mark an explicitly selected config as a legacy RT/choice producer."""
    config["observation_schema_version"] = 1
    config["observation_schema_profile"] = "legacy_rt_choice"
    return config


def _normalize_param_bounds(config: dict) -> dict:
    """Normalize param_bounds to param_bounds_dict format.

    Converts the param_bounds field (which can be either list or dict format)
    to a standardized dict format stored in param_bounds_dict.

    Args:
        config: Model configuration dict

    Returns:
        Modified config with param_bounds_dict added
    """
    if "param_bounds_dict" in config:
        # Already normalized
        return config

    if "param_bounds" not in config:
        # No param_bounds to normalize
        return config

    if isinstance(config["param_bounds"], list):
        # Convert list format to dict: [[low1, low2, ...], [high1, high2, ...]]
        bounds_lower, bounds_upper = config["param_bounds"]
        config["param_bounds_dict"] = {
            param: (lower, upper)
            for param, lower, upper in zip(config["params"], bounds_lower, bounds_upper)
        }
    elif isinstance(config["param_bounds"], dict):
        # Already dict format, just copy
        config["param_bounds_dict"] = config["param_bounds"]

    return config


def get_model_config():
    """Accessor for model configurations.

    Returns
    -------
    dict
        Dictionary containing all model configurations.

    Note:
        All returned configs are normalized to include param_bounds_dict,
        which is the dict format of param_bounds for easier parameter sampling.
    """
    # TODO: Refactor to load these lazily
    configs = {
        "ddm": _rt_choice(get_ddm_config()),
        "ddm_st": _rt_choice(get_ddm_st_config()),
        "ddm_truncnormt": _rt_choice(get_ddm_truncnormt_config()),
        "ddm_rayleight": _rt_choice(get_ddm_rayleight_config()),
        "ddm_sdv": _rt_choice(get_ddm_sdv_config()),
        "ddm_par2": _rt_choice(get_ddm_par2_config()),
        "ddm_par2_no_bias": _rt_choice(get_ddm_par2_no_bias_config()),
        "ddm_par2_conflict_gamma_no_bias": _rt_choice(
            get_ddm_par2_conflict_gamma_no_bias_config()
        ),
        "ddm_par2_angle_no_bias": _rt_choice(get_ddm_par2_angle_no_bias_config()),
        "ddm_par2_weibull_no_bias": _rt_choice(get_ddm_par2_weibull_no_bias_config()),
        "ddm_seq2": _rt_choice(get_ddm_seq2_config()),
        "ddm_seq2_no_bias": _rt_choice(get_ddm_seq2_no_bias_config()),
        "ddm_seq2_conflict_gamma_no_bias": _rt_choice(
            get_ddm_seq2_conflict_gamma_no_bias_config()
        ),
        "ddm_seq2_angle_no_bias": _rt_choice(get_ddm_seq2_angle_no_bias_config()),
        "ddm_seq2_weibull_no_bias": _rt_choice(get_ddm_seq2_weibull_no_bias_config()),
        "ddm_mic2_adj": _rt_choice(get_ddm_mic2_adj_config()),
        "ddm_mic2_adj_no_bias": _rt_choice(get_ddm_mic2_adj_no_bias_config()),
        "ddm_mic2_adj_conflict_gamma_no_bias": _rt_choice(
            get_ddm_mic2_adj_conflict_gamma_no_bias_config()
        ),
        "ddm_mic2_adj_angle_no_bias": _rt_choice(
            get_ddm_mic2_adj_angle_no_bias_config()
        ),
        "ddm_mic2_adj_weibull_no_bias": _rt_choice(
            get_ddm_mic2_adj_weibull_no_bias_config()
        ),
        "ddm_mic2_ornstein": _rt_choice(get_ddm_mic2_ornstein_config()),
        "ddm_mic2_ornstein_no_bias": _rt_choice(get_ddm_mic2_ornstein_no_bias_config()),
        "ddm_mic2_ornstein_no_bias_no_lowdim_noise": (
            _rt_choice(get_ddm_mic2_ornstein_no_bias_config())
        ),
        "ddm_mic2_ornstein_conflict_gamma_no_bias": (
            _rt_choice(get_ddm_mic2_ornstein_conflict_gamma_no_bias_config())
        ),
        "ddm_mic2_ornstein_conflict_gamma_no_bias_no_lowdim_noise": (
            _rt_choice(get_ddm_mic2_ornstein_conflict_gamma_no_bias_config())
        ),
        "ddm_mic2_ornstein_angle_no_bias": _rt_choice(
            get_ddm_mic2_ornstein_angle_no_bias_config()
        ),
        "ddm_mic2_ornstein_angle_no_bias_no_lowdim_noise": (
            _rt_choice(get_ddm_mic2_ornstein_angle_no_bias_config())
        ),
        "ddm_mic2_ornstein_weibull_no_bias": _rt_choice(
            get_ddm_mic2_ornstein_weibull_no_bias_config()
        ),
        "ddm_mic2_ornstein_weibull_no_bias_no_lowdim_noise": (
            _rt_choice(get_ddm_mic2_ornstein_weibull_no_bias_config())
        ),
        "ddm_mic2_leak": _rt_choice(get_ddm_mic2_leak_config()),
        "ddm_mic2_leak_no_bias": _rt_choice(get_ddm_mic2_leak_no_bias_config()),
        "ddm_mic2_leak_no_bias_no_lowdim_noise": _rt_choice(
            get_ddm_mic2_leak_no_bias_config()
        ),
        "ddm_mic2_leak_conflict_gamma_no_bias": _rt_choice(
            get_ddm_mic2_leak_conflict_gamma_no_bias_config()
        ),
        "ddm_mic2_leak_conflict_gamma_no_bias_no_lowdim_noise": (
            _rt_choice(get_ddm_mic2_leak_conflict_gamma_no_bias_config())
        ),
        "ddm_mic2_leak_angle_no_bias": _rt_choice(
            get_ddm_mic2_leak_angle_no_bias_config()
        ),
        "ddm_mic2_leak_angle_no_bias_no_lowdim_noise": (
            _rt_choice(get_ddm_mic2_leak_angle_no_bias_config())
        ),
        "ddm_mic2_leak_weibull_no_bias": _rt_choice(
            get_ddm_mic2_leak_weibull_no_bias_config()
        ),
        "ddm_mic2_leak_weibull_no_bias_no_lowdim_noise": (
            _rt_choice(get_ddm_mic2_leak_weibull_no_bias_config())
        ),
        "ddm_mic2_multinoise_no_bias": _rt_choice(
            get_ddm_mic2_multinoise_no_bias_config()
        ),
        "ddm_mic2_multinoise_conflict_gamma_no_bias": (
            _rt_choice(get_ddm_mic2_multinoise_conflict_gamma_no_bias_config())
        ),
        "ddm_mic2_multinoise_angle_no_bias": _rt_choice(
            get_ddm_mic2_multinoise_angle_no_bias_config()
        ),
        "ddm_mic2_multinoise_weibull_no_bias": _rt_choice(
            get_ddm_mic2_multinoise_weibull_no_bias_config()
        ),
        "addm": _rt_choice(get_addm_config()),
        "full_ddm": _rt_choice(get_full_ddm_config()),
        "full_ddm_rv": _rt_choice(get_full_ddm_rv_config()),
        "levy": _rt_choice(get_levy_config()),
        "levy_angle": _rt_choice(get_levy_angle_config()),
        "angle": _rt_choice(get_angle_config()),
        "weibull": _rt_choice(get_weibull_config()),
        "gamma_drift": _rt_choice(get_gamma_drift_config()),
        "inv_temp_softmax_2": get_inv_temp_softmax_2_config(),
        "inv_temp_softmax_3": get_inv_temp_softmax_3_config(),
        "inv_temp_softmax_4": get_inv_temp_softmax_4_config(),
        "shrink_spot": _rt_choice(get_shrink_spot_config()),
        "shrink_spot_extended": _rt_choice(get_shrink_spot_extended_config()),
        "shrink_spot_simple": _rt_choice(get_shrink_spot_simple_config()),
        "shrink_spot_simple_extended": _rt_choice(
            get_shrink_spot_simple_extended_config()
        ),
        "gamma_drift_angle": _rt_choice(get_gamma_drift_angle_config()),
        "conflict_ds": _rt_choice(get_conflict_ds_config()),
        "conflict_ds_angle": _rt_choice(get_conflict_ds_angle_config()),
        "conflict_dsstimflex": _rt_choice(get_conflict_dsstimflex_config()),
        "conflict_dsstimflex_angle": _rt_choice(get_conflict_dsstimflex_angle_config()),
        "conflict_stimflex": _rt_choice(get_conflict_stimflex_config()),
        "conflict_stimflex_angle": _rt_choice(get_conflict_stimflex_angle_config()),
        "conflict_stimflexrel1": _rt_choice(get_conflict_stimflexrel1_config()),
        "conflict_stimflexrel1_angle": _rt_choice(
            get_conflict_stimflexrel1_angle_config()
        ),
        "conflict_stimflexrel1_leak": _rt_choice(
            get_conflict_stimflexrel1_leak_config()
        ),
        "conflict_stimflexrel1_leak2": _rt_choice(
            get_conflict_stimflexrel1_leak2_config()
        ),
        "ornstein": _rt_choice(get_ornstein_config()),
        "ornstein_angle": _rt_choice(get_ornstein_angle_config()),
        "race_2": _rt_choice(get_race_2_config()),
        "race_no_bias_2": _rt_choice(get_race_no_bias_2_config()),
        "race_no_z_2": _rt_choice(get_race_no_z_2_config()),
        "race_no_bias_angle_2": _rt_choice(get_race_no_bias_angle_2_config()),
        "race_no_z_angle_2": _rt_choice(get_race_no_z_angle_2_config()),
        "race_3": _rt_choice(get_race_3_config()),
        "race_no_bias_3": _rt_choice(get_race_no_bias_3_config()),
        "race_no_z_3": _rt_choice(get_race_no_z_3_config()),
        "race_no_bias_angle_3": _rt_choice(get_race_no_bias_angle_3_config()),
        "race_no_z_angle_3": _rt_choice(get_race_no_z_angle_3_config()),
        "race_4": _rt_choice(get_race_4_config()),
        "race_no_bias_4": _rt_choice(get_race_no_bias_4_config()),
        "race_no_z_4": _rt_choice(get_race_no_z_4_config()),
        "race_no_bias_angle_4": _rt_choice(get_race_no_bias_angle_4_config()),
        "race_no_z_angle_4": _rt_choice(get_race_no_z_angle_4_config()),
        "racing_diffusion_3": _rt_choice(get_racing_diffusion_3_config()),
        "poisson_race": _rt_choice(get_poisson_race_config()),
        "dev_rlwm_lba_pw_v1": _rt_choice(get_dev_rlwm_lba_pw_v1_config()),
        "dev_rlwm_lba_race_v1": _rt_choice(get_dev_rlwm_lba_race_v1_config()),
        "dev_rlwm_lba_race_v2": _rt_choice(get_dev_rlwm_lba_race_v2_config()),
        "lba2": _rt_choice(get_lba2_config()),
        "lba3": _rt_choice(get_lba3_config()),
        "lba4": _rt_choice(get_lba4_config()),
        "lba_3_vs_constraint": _rt_choice(get_lba_3_vs_constraint_config()),
        "lba_angle_3_vs_constraint": _rt_choice(get_lba_angle_3_vs_constraint_config()),
        "lba_angle_3": _rt_choice(get_lba_angle_3_config()),
        "lca_3": _rt_choice(get_lca_3_config()),
        "lca_no_bias_3": _rt_choice(get_lca_no_bias_3_config()),
        "lca_no_z_3": _rt_choice(get_lca_no_z_3_config()),
        "lca_no_bias_angle_3": _rt_choice(get_lca_no_bias_angle_3_config()),
        "lca_no_z_angle_3": _rt_choice(get_lca_no_z_angle_3_config()),
        "lca_4": _rt_choice(get_lca_4_config()),
        "lca_no_bias_4": _rt_choice(get_lca_no_bias_4_config()),
        "lca_no_z_4": _rt_choice(get_lca_no_z_4_config()),
        "lca_no_bias_angle_4": _rt_choice(get_lca_no_bias_angle_4_config()),
        "lca_no_z_angle_4": _rt_choice(get_lca_no_z_angle_4_config()),
        "tradeoff_no_bias": _rt_choice(get_tradeoff_no_bias_config()),
        "tradeoff_angle_no_bias": _rt_choice(get_tradeoff_angle_no_bias_config()),
        "tradeoff_weibull_no_bias": _rt_choice(get_tradeoff_weibull_no_bias_config()),
        "tradeoff_conflict_gamma_no_bias": _rt_choice(
            get_tradeoff_conflict_gamma_no_bias_config()
        ),
        "weibull_cdf": _rt_choice(get_weibull_config()),
        "full_ddm2": _rt_choice(get_full_ddm_config()),
        "ddm_legacy": _rt_choice(get_ddm_legacy_config()),
    }

    # Normalize all configs to include param_bounds_dict
    return {name: _normalize_param_bounds(cfg) for name, cfg in configs.items()}


__all__ = [
    "get_model_config",
    "get_ddm_config",
    "get_angle_config",
    "get_weibull_config",
    "get_full_ddm_config",
    "get_ddm_st_config",
    "get_ddm_truncnormt_config",
    "get_ddm_rayleight_config",
    "get_ddm_sdv_config",
    "get_ddm_par2_config",
    "get_ddm_par2_no_bias_config",
    "get_ddm_par2_conflict_gamma_no_bias_config",
    "get_ddm_par2_angle_no_bias_config",
    "get_ddm_par2_weibull_no_bias_config",
    "get_ddm_seq2_config",
    "get_ddm_seq2_no_bias_config",
    "get_ddm_seq2_conflict_gamma_no_bias_config",
    "get_ddm_seq2_angle_no_bias_config",
    "get_ddm_seq2_weibull_no_bias_config",
    "get_ddm_mic2_adj_config",
    "get_ddm_mic2_adj_no_bias_config",
    "get_ddm_mic2_adj_conflict_gamma_no_bias_config",
    "get_ddm_mic2_adj_angle_no_bias_config",
    "get_ddm_mic2_adj_weibull_no_bias_config",
    "get_ddm_mic2_ornstein_config",
    "get_ddm_mic2_ornstein_no_bias_config",
    "get_ddm_mic2_ornstein_conflict_gamma_no_bias_config",
    "get_ddm_mic2_ornstein_angle_no_bias_config",
    "get_ddm_mic2_ornstein_weibull_no_bias_config",
    "get_ddm_mic2_leak_config",
    "get_ddm_mic2_leak_no_bias_config",
    "get_ddm_mic2_leak_conflict_gamma_no_bias_config",
    "get_ddm_mic2_leak_angle_no_bias_config",
    "get_ddm_mic2_leak_weibull_no_bias_config",
    "get_ddm_mic2_multinoise_no_bias_config",
    "get_ddm_mic2_multinoise_conflict_gamma_no_bias_config",
    "get_ddm_mic2_multinoise_angle_no_bias_config",
    "get_ddm_mic2_multinoise_weibull_no_bias_config",
    "get_poisson_race_config",
    "get_addm_config",
    "get_inv_temp_softmax_2_config",
    "get_inv_temp_softmax_3_config",
    "get_inv_temp_softmax_4_config",
]

# Validate


def _validate_configs():
    """Validate all configurations for parameter name consistency."""
    # Import locally to avoid circular imports
    from .base import boundary_config, drift_config

    _ALL_CONFIGS = {
        "model_configs": get_model_config(),
        "drift_configs": drift_config,
        "boundary_configs": boundary_config,
    }
    invalid_configs = {
        key: get_invalid_configs(configs) for key, configs in _ALL_CONFIGS.items()
    }
    if any(invalid_configs.values()):
        raise ValueError(f"Invalid parameter names detected: {invalid_configs}")


_validate_configs()
