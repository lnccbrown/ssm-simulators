"""Configuration dictionary for simulators.

Variables:
---------
model_config: dict
    Dictionary containing all the information about the models
"""

import cssm

from ssms import boundary_functions as bf
from ssms import drift_functions as df

from ssms.config._modelconfig import get_model_config
from ssms.config._modelconfig.tradeoff import (
    get_tradeoff_no_bias_config,
    get_tradeoff_angle_no_bias_config,
    get_tradeoff_weibull_no_bias_config,
    get_tradeoff_conflict_gamma_no_bias_config,
)
from ssms.config._modelconfig.ddm import (
    get_ddm_config,
)
from ssms.config._modelconfig.full_ddm import (
    get_full_ddm_config,
    get_full_ddm_rv_config,
)
from ssms.config._modelconfig.levy import get_levy_config, get_levy_angle_config
from ssms.config._modelconfig.ds_conflict_drift import (
    get_ds_conflict_drift_config,
    get_ds_conflict_drift_angle_config,
    get_ds_conflict_stimflexons_drift_config,
    get_ds_conflict_stimflexons_drift_angle_config,
)
from ssms.config._modelconfig.lca import (
    get_lca_3_config,
    get_lca_no_bias_3_config,
    get_lca_no_bias_angle_3_config,
    get_lca_no_z_3_config,
    get_lca_no_z_angle_3_config,
    get_lca_4_config,
    get_lca_no_bias_4_config,
    get_lca_no_z_4_config,
    get_lca_no_bias_angle_4_config,
    get_lca_no_z_angle_4_config,
)
from ._modelconfig.angle import get_angle_config
from ._modelconfig.weibull import get_weibull_config

from ssms.config._modelconfig.lba import (
    get_lba2_config,
    get_lba3_config,
    get_lba_3_vs_constraint_config,
    get_lba_angle_3_vs_constraint_config,
    get_lba_angle_3_config,
)

from ssms.config._modelconfig.shrink import (
    get_shrink_spot_config,
    get_shrink_spot_extended_config,
    get_shrink_spot_simple_config,
    get_shrink_spot_simple_extended_config,
)

from ssms.config._modelconfig.race import (
    get_race_2_config,
    get_race_no_bias_2_config,
    get_race_no_z_2_config,
    get_race_no_bias_angle_2_config,
    get_race_no_z_angle_2_config,
    get_race_3_config,
    get_race_no_bias_3_config,
    get_race_no_z_3_config,
    get_race_no_bias_angle_3_config,
    get_race_no_z_angle_3_config,
    get_race_4_config,
    get_race_no_bias_4_config,
    get_race_no_z_4_config,
    get_race_no_bias_angle_4_config,
    get_race_no_z_angle_4_config,
)

from ssms.config._modelconfig.dev_rlwm_lba import (
    get_dev_rlwm_lba_pw_v1_config,
    get_dev_rlwm_lba_race_v1_config,
    get_dev_rlwm_lba_race_v2_config,
)


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


model_config_getter = get_model_config()
# Configuration dictionary for simulators
model_config = {
    "ddm": get_ddm_config(),
    "ddm_legacy": {
        "name": "ddm_legacy",
        "params": ["v", "a", "z", "t"],
        "param_bounds": [[-3.0, 0.3, 0.1, 0.0], [3.0, 2.5, 0.9, 2.0]],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 4,
        "default_params": [0.0, 1.0, 0.5, 1e-3],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm,
    },
    "full_ddm": get_full_ddm_config(),
    "full_ddm_rv": get_full_ddm_rv_config(),
    "levy": get_levy_config(),
    "levy_angle": get_levy_angle_config(),
    "angle": get_angle_config(),
    "weibull": get_weibull_config(),
    "ddm_st": model_config_getter["ddm_st"],
    "ddm_truncnormt": model_config_getter["ddm_truncnormt"],
    "ddm_rayleight": model_config_getter["ddm_rayleight"],
    "ddm_sdv": model_config_getter["ddm_sdv"],
    "gamma_drift": {
        "name": "gamma_drift",
        "params": ["v", "a", "z", "t", "shape", "scale", "c"],
        "param_bounds": [
            [-3.0, 0.3, 0.1, 1e-3, 2.0, 0.01, -3.0],
            [3.0, 3.0, 0.9, 2.0, 10.0, 1.0, 3.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "drift_name": "gamma_drift",
        "drift_fun": df.gamma_drift,
        "n_params": 7,
        "default_params": [0.0, 1.0, 0.5, 0.25, 5.0, 0.5, 1.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
    },
    "shrink_spot": get_shrink_spot_config(),
    "shrink_spot_extended": get_shrink_spot_extended_config(),
    "shrink_spot_simple": get_shrink_spot_simple_config(),
    "shrink_spot_simple_extended": get_shrink_spot_simple_extended_config(),
    "gamma_drift_angle": {
        "name": "gamma_drift_angle",
        "params": ["v", "a", "z", "t", "theta", "shape", "scale", "c"],
        "param_bounds": [
            [-3.0, 0.3, 0.1, 1e-3, -0.1, 2.0, 0.01, -3.0],
            [3.0, 3.0, 0.9, 2.0, 1.3, 10.0, 1.0, 3.0],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "drift_name": "gamma_drift",
        "drift_fun": df.gamma_drift,
        "n_params": 8,
        "default_params": [0.0, 1.0, 0.5, 0.25, 0.0, 5.0, 0.5, 1.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
    },
    "ds_conflict_drift": get_ds_conflict_drift_config(),
    "ds_conflict_drift_angle": get_ds_conflict_drift_angle_config(),
    "ds_conflict_stimflexons_drift": get_ds_conflict_stimflexons_drift_config(),
    "ds_conflict_stimflexons_drift_angle": get_ds_conflict_stimflexons_drift_angle_config(),
    "ornstein": {
        "name": "ornstein",
        "params": ["v", "a", "z", "g", "t"],
        "param_bounds": [[-2.0, 0.3, 0.1, -1.0, 1e-3], [2.0, 3.0, 0.9, 1.0, 2]],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.0, 1.0, 0.5, 0.0, 1e-3],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ornstein_uhlenbeck,
    },
    "ornstein_angle": {
        "name": "ornstein_angle",
        "params": ["v", "a", "z", "g", "t", "theta"],
        "param_bounds": [
            [-2.0, 0.3, 0.1, -1.0, 1e-3, -0.1],
            [2.0, 3.0, 0.9, 1.0, 2, 1.3],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 6,
        "default_params": [0.0, 1.0, 0.5, 0.0, 1e-3, 0.1],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ornstein_uhlenbeck,
    },
    "race_2": get_race_2_config(),
    "race_no_bias_2": get_race_no_bias_2_config(),
    "race_no_z_2": get_race_no_z_2_config(),
    "race_no_bias_angle_2": get_race_no_bias_angle_2_config(),
    "race_no_z_angle_2": get_race_no_z_angle_2_config(),
    "race_3": get_race_3_config(),
    "race_no_bias_3": get_race_no_bias_3_config(),
    "race_no_z_3": get_race_no_z_3_config(),
    "race_no_bias_angle_3": get_race_no_bias_angle_3_config(),
    "race_no_z_angle_3": get_race_no_z_angle_3_config(),
    "race_4": get_race_4_config(),
    "race_no_bias_4": get_race_no_bias_4_config(),
    "race_no_z_4": get_race_no_z_4_config(),
    "race_no_bias_angle_4": get_race_no_bias_angle_4_config(),
    "race_no_z_angle_4": get_race_no_z_angle_4_config(),
    "dev_rlwm_lba_pw_v1": get_dev_rlwm_lba_pw_v1_config(),
    "dev_rlwm_lba_race_v1": get_dev_rlwm_lba_race_v1_config(),
    "dev_rlwm_lba_race_v2": get_dev_rlwm_lba_race_v2_config(),
    "lba2": get_lba2_config(),
    "lba3": get_lba3_config(),
    "lba_3_vs_constraint": get_lba_3_vs_constraint_config(),
    "lba_angle_3_vs_constraint": get_lba_angle_3_vs_constraint_config(),
    "lba_angle_3": get_lba_angle_3_config(),
    "lca_3": get_lca_3_config(),
    "lca_no_bias_3": get_lca_no_bias_3_config(),
    "lca_no_z_3": get_lca_no_z_3_config(),
    "lca_no_bias_angle_3": get_lca_no_bias_angle_3_config(),
    "lca_no_z_angle_3": get_lca_no_z_angle_3_config(),
    "lca_4": get_lca_4_config(),
    "lca_no_bias_4": get_lca_no_bias_4_config(),
    "lca_no_z_4": get_lca_no_z_4_config(),
    "lca_no_bias_angle_4": get_lca_no_bias_angle_4_config(),
    "lca_no_z_angle_4": get_lca_no_z_angle_4_config(),
    "ddm_par2": model_config_getter["ddm_par2"],
    "ddm_par2_no_bias": model_config_getter["ddm_par2_no_bias"],
    "ddm_par2_conflict_gamma_no_bias": model_config_getter[
        "ddm_par2_conflict_gamma_no_bias"
    ],
    "ddm_par2_angle_no_bias": model_config_getter["ddm_par2_angle_no_bias"],
    "ddm_par2_weibull_no_bias": model_config_getter["ddm_par2_weibull_no_bias"],
    "ddm_seq2": model_config_getter["ddm_seq2"],
    "ddm_seq2_no_bias": model_config_getter["ddm_seq2_no_bias"],
    "ddm_seq2_conflict_gamma_no_bias": model_config_getter[
        "ddm_seq2_conflict_gamma_no_bias"
    ],
    "ddm_seq2_angle_no_bias": model_config_getter["ddm_seq2_angle_no_bias"],
    "ddm_seq2_weibull_no_bias": model_config_getter["ddm_seq2_weibull_no_bias"],
    "ddm_mic2_adj": model_config_getter["ddm_mic2_adj"],
    "ddm_mic2_adj_no_bias": model_config_getter["ddm_mic2_adj_no_bias"],
    "ddm_mic2_adj_conflict_gamma_no_bias": model_config_getter[
        "ddm_mic2_adj_conflict_gamma_no_bias"
    ],
    "ddm_mic2_adj_angle_no_bias": model_config_getter["ddm_mic2_adj_angle_no_bias"],
    "ddm_mic2_adj_weibull_no_bias": model_config_getter["ddm_mic2_adj_weibull_no_bias"],
    "ddm_mic2_ornstein": model_config_getter["ddm_mic2_ornstein"],
    "ddm_mic2_ornstein_no_bias": model_config_getter["ddm_mic2_ornstein_no_bias"],
    "ddm_mic2_ornstein_conflict_gamma_no_bias": model_config_getter[
        "ddm_mic2_ornstein_conflict_gamma_no_bias"
    ],
    "ddm_mic2_ornstein_angle_no_bias": model_config_getter[
        "ddm_mic2_ornstein_angle_no_bias"
    ],
    "ddm_mic2_ornstein_weibull_no_bias": model_config_getter[
        "ddm_mic2_ornstein_weibull_no_bias"
    ],
    "ddm_mic2_multinoise_no_bias": model_config_getter["ddm_mic2_multinoise_no_bias"],
    "ddm_mic2_multinoise_conflict_gamma_no_bias": model_config_getter[
        "ddm_mic2_multinoise_conflict_gamma_no_bias"
    ],
    "ddm_mic2_multinoise_angle_no_bias": model_config_getter[
        "ddm_mic2_multinoise_angle_no_bias"
    ],
    "ddm_mic2_multinoise_weibull_no_bias": model_config_getter[
        "ddm_mic2_multinoise_weibull_no_bias"
    ],
    "ddm_mic2_leak": model_config_getter["ddm_mic2_leak"],
    "ddm_mic2_leak_no_bias": model_config_getter["ddm_mic2_leak_no_bias"],
    "ddm_mic2_leak_conflict_gamma_no_bias": model_config_getter[
        "ddm_mic2_leak_conflict_gamma_no_bias"
    ],
    "ddm_mic2_leak_angle_no_bias": model_config_getter["ddm_mic2_leak_angle_no_bias"],
    "ddm_mic2_leak_weibull_no_bias": model_config_getter[
        "ddm_mic2_leak_weibull_no_bias"
    ],
    "tradeoff_no_bias": get_tradeoff_no_bias_config(),
    "tradeoff_angle_no_bias": get_tradeoff_angle_no_bias_config(),
    "tradeoff_weibull_no_bias": get_tradeoff_weibull_no_bias_config(),
    "tradeoff_conflict_gamma_no_bias": get_tradeoff_conflict_gamma_no_bias_config(),
}

model_config["weibull_cdf"] = get_weibull_config()
model_config["full_ddm2"] = get_full_ddm_config()
model_config["ddm_mic2_ornstein_no_bias_no_lowdim_noise"] = model_config[
    "ddm_mic2_ornstein_no_bias"
].copy()
model_config["ddm_mic2_ornstein_angle_no_bias_no_lowdim_noise"] = model_config[
    "ddm_mic2_ornstein_angle_no_bias"
].copy()
model_config["ddm_mic2_ornstein_weibull_no_bias_no_lowdim_noise"] = model_config[
    "ddm_mic2_ornstein_weibull_no_bias"
].copy()
model_config["ddm_mic2_ornstein_conflict_gamma_no_bias_no_lowdim_noise"] = model_config[
    "ddm_mic2_ornstein_conflict_gamma_no_bias"
].copy()
model_config["ddm_mic2_leak_no_bias_no_lowdim_noise"] = model_config[
    "ddm_mic2_leak_no_bias"
].copy()
model_config["ddm_mic2_leak_angle_no_bias_no_lowdim_noise"] = model_config[
    "ddm_mic2_leak_angle_no_bias"
].copy()
model_config["ddm_mic2_leak_weibull_no_bias_no_lowdim_noise"] = model_config[
    "ddm_mic2_leak_weibull_no_bias"
].copy()
model_config["ddm_mic2_leak_conflict_gamma_no_bias_no_lowdim_noise"] = model_config[
    "ddm_mic2_leak_conflict_gamma_no_bias"
].copy()
