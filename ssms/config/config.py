"""Configuration dictionary for simulators.

Variables:
---------
model_config: dict
    Dictionary containing all the information about the models

kde_simulation_filters: dict
    Dictionary containing the filters for the KDE simulations

data_generator_config: dict
    Dictionary containing information for data generator settings.
    Supposed to serve as a starting point and example, which the user then
    modifies to their needs.
"""

import cssm

from ssms.basic_simulators import boundary_functions as bf
from ssms.basic_simulators import drift_functions as df

from ssms.config._modelconfig import get_model_config
from ssms.config._modelconfig.full_ddm import (
    get_full_ddm_config,
    get_full_ddm_rv_config,
)
from ssms.config._modelconfig.levy import get_levy_config, get_levy_angle_config
from ._modelconfig.angle import get_angle_config
from ._modelconfig.weibull import get_weibull_config


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
    "ddm": model_config_getter["ddm"],
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
    "shrink_spot": {
        "name": "shrink_spot",
        "params": [
            "a",
            "z",
            "t",
            "ptarget",
            "pouter",
            "pinner",
            "r",
            "sda",
        ],
        "param_bounds": [
            [0.3, 0.1, 1e-3, 2.0, -5.5, -5.5, 1e-2, 1],
            [3.0, 0.9, 2.0, 5.5, 5.5, 5.5, 0.05, 3],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "drift_name": "attend_drift",
        "drift_fun": df.attend_drift,
        "n_params": 8,
        "default_params": [0.7, 0.5, 0.25, 2.0, -2.0, -2.0, 0.01, 1],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
    },
    "shrink_spot_extended": {
        "name": "shrink_spot",
        "params": [
            "a",
            "z",
            "t",
            "ptarget",
            "pouter",
            "pinner",
            "r",
            "sda",
        ],
        "param_bounds": [
            [0.3, 0.1, 1e-3, 2.0, -5.5, -5.5, 0.01, 1],
            [3.0, 0.9, 2.0, 5.5, 5.5, 5.5, 1.0, 3],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "drift_name": "attend_drift",
        "drift_fun": df.attend_drift,
        "n_params": 8,
        "default_params": [0.7, 0.5, 0.25, 2.0, -2.0, -2.0, 0.01, 1],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
    },
    "shrink_spot_simple": {
        "name": "shrink_spot_simple",
        "params": [
            "a",
            "z",
            "t",
            "ptarget",
            "pouter",
            "r",
            "sda",
        ],
        "param_bounds": [
            [0.3, 0.1, 1e-3, 2.0, -5.5, 0.01, 1],
            [3.0, 0.9, 2.0, 5.5, 5.5, 0.05, 3],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "drift_name": "attend_drift_simple",
        "drift_fun": df.attend_drift_simple,
        "n_params": 7,
        "default_params": [0.7, 0.5, 0.25, 2.0, -2.0, 0.01, 1],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
    },
    "shrink_spot_simple_extended": {
        "name": "shrink_spot_simple_extended",
        "params": [
            "a",
            "z",
            "t",
            "ptarget",
            "pouter",
            "r",
            "sda",
        ],
        "param_bounds": [
            [0.3, 0.1, 1e-3, 2.0, -5.5, 0.01, 1],
            [3.0, 0.9, 2.0, 5.5, 5.5, 1.0, 3],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "drift_name": "attend_drift_simple",
        "drift_fun": df.attend_drift_simple,
        "n_params": 7,
        "default_params": [0.7, 0.5, 0.25, 2.0, -2.0, 0.01, 1],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
    },
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
        "n_params": 7,
        "default_params": [0.0, 1.0, 0.5, 0.25, 0.0, 5.0, 0.5, 1.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
    },
    "ds_conflict_drift": {
        "name": "ds_conflict_drift",
        "params": [
            "a",
            "z",
            "t",
            "tinit",
            "dinit",
            "tslope",
            "dslope",
            "tfixedp",
            "tcoh",
            "dcoh",
        ],
        "param_bounds": [
            [0.3, 0.1, 1e-3, 0.0, 0.0, 0.01, 0.01, 0.0, -1.0, -1.0],
            [3.0, 0.9, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "drift_name": "ds_conflict_drift",
        "drift_fun": df.ds_conflict_drift,
        "n_params": 10,
        "default_params": [2.0, 0.5, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 0.5, -0.5],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
    },
    "ds_conflict_drift_angle": {
        "name": "ds_conflict_drift_angle",
        "params": [
            "a",
            "z",
            "t",
            "tinit",
            "dinit",
            "tslope",
            "dslope",
            "tfixedp",
            "tcoh",
            "dcoh",
            "theta",
        ],
        "param_bounds": [
            [0.3, 0.1, 1e-3, 0.0, 0.0, 0.01, 0.01, 0.0, -1.0, -1.0, 0.0],
            [3.0, 0.9, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.3],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "drift_name": "ds_conflict_drift",
        "drift_fun": df.ds_conflict_drift,
        "n_params": 10,
        "default_params": [2.0, 0.5, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 0.5, -0.5, 0.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
    },
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
    "lba2": {
        "name": "lba2",
        "params": ["A", "b", "v0", "v1"],
        "param_bounds": [[0.0, 0.0, 0.0, 0.1], [1.0, 1.0, 1.0, 1.1]],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 4,
        "default_params": [0.3, 0.5, 0.5, 0.5],
        "nchoices": 2,
        "choices": [0, 1],
        "n_particles": 2,
        "simulator": cssm.lba_vanilla,
    },
    "lba3": {
        "name": "lba3",
        "params": ["A", "b", "v0", "v1", "v2"],
        "param_bounds": [[0.0, 0.0, 0.0, 0.1, 0.1], [1.0, 1.0, 1.0, 1.1, 0.50]],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.3, 0.5, 0.25, 0.5, 0.25],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.lba_vanilla,
    },
    "lba_3_vs_constraint": {
        # conventional analytical LBA with constraints on vs (sum of all v = 1)
        "name": "lba_3_vs_constraint",
        "params": ["v0", "v1", "v2", "a", "z"],
        "param_bounds": [[0.0, 0.0, 0.0, 0.1, 0.1], [1.0, 1.0, 1.0, 1.1, 0.50]],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.5, 0.3, 0.2, 0.5, 0.2],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.lba_vanilla,
    },
    "lba_angle_3_vs_constraint": {
        # conventional analytical LBA with angle with constraints on vs (sum of all v=1)
        "name": "lba_angle_3_vs_constraint",
        "params": ["v0", "v1", "v2", "a", "z", "theta"],
        "param_bounds": [[0.0, 0.0, 0.0, 0.1, 0.0, 0], [1.0, 1.0, 1.0, 1.1, 0.5, 1.3]],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 6,
        "default_params": [0.5, 0.3, 0.2, 0.5, 0.2, 0.0],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.lba_angle,
    },
    "lba_angle_3": {
        # conventional analytical LBA with angle without any constraints on vs
        "name": "lba_angle_3",
        "params": ["v0", "v1", "v2", "a", "z", "theta"],
        "param_bounds": [[0.0, 0.0, 0.0, 0.1, 0.0, 0], [6.0, 6.0, 6.0, 1.1, 0.5, 1.3]],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 6,
        "default_params": [0.5, 0.3, 0.2, 0.5, 0.2, 0.0],
        "nchoices": 3,
        "n_particles": 3,
        "simulator": cssm.lba_angle,
    },
    "dev_rlwm_lba_pw_v1": {
        "name": "dev_rlwm_lba_pw_v1",
        "params": [
            "v_RL_0",
            "v_RL_1",
            "v_RL_2",
            "v_WM_0",
            "v_WM_1",
            "v_WM_2",
            "a",
            "z",
            "t_WM",
        ],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.01],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 0.5, 0.5],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 9,
        "default_params": [0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.5, 0.2, 0.1],
        "nchoices": 3,
        "n_particles": 3,
        "simulator": cssm.rlwm_lba_pw_v1,
    },
    "dev_rlwm_lba_race_v1": {
        # RLWM_Race_LBA_3 without ndt; sum of all v_RL = 1 and sum of all v_WM = 1
        "name": "dev_rlwm_lba_race_v1",
        "params": [
            "v_RL_0",
            "v_RL_1",
            "v_RL_2",
            "v_WM_0",
            "v_WM_1",
            "v_WM_2",
            "a",
            "z",
        ],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 0.5],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 8,
        "default_params": [0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.5, 0.2],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.rlwm_lba_race,
    },
    "dev_rlwm_lba_race_v2": {
        # RLWM_Race_LBA_3 without ndt; no constraints on the sum of v_RL and v_WM.
        "name": "dev_rlwm_lba_race_v2",
        "params": [
            "v_RL_0",
            "v_RL_1",
            "v_RL_2",
            "v_WM_0",
            "v_WM_1",
            "v_WM_2",
            "a",
            "z",
        ],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 8,
        "default_params": [0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.5, 0.2],
        "nchoices": 3,
        "n_particles": 3,
        "simulator": cssm.rlwm_lba_race,
    },
    "race_2": {
        "name": "race_2",
        "params": ["v0", "v1", "a", "z0", "z1", "t"],
        "param_bounds": [
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [2.5, 2.5, 3.0, 0.9, 0.9, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 6,
        "default_params": [0.0, 0.0, 2.0, 0.5, 0.5, 1e-3],
        "nchoices": 2,
        "choices": [0, 1],
        "n_particles": 2,
        "simulator": cssm.race_model,
    },
    "race_no_bias_2": {
        "name": "race_no_bias_2",
        "params": ["v0", "v1", "a", "z", "t"],
        "param_bounds": [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [2.5, 2.5, 3.0, 0.9, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.0, 0.0, 2.0, 0.5, 1e-3],
        "nchoices": 2,
        "choices": [0, 1],
        "n_particles": 2,
        "simulator": cssm.race_model,
    },
    "race_no_z_2": {
        "name": "race_no_z_2",
        "params": ["v0", "v1", "a", "t"],
        "param_bounds": [
            [0.0, 0.0, 1.0, 0.0],
            [2.5, 2.5, 3.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 4,
        "default_params": [0.0, 0.0, 2.0, 1e-3],
        "nchoices": 2,
        "choices": [0, 1],
        "n_particles": 2,
        "simulator": cssm.race_model,
    },
    "race_no_bias_angle_2": {
        "name": "race_no_bias_angle_2",
        "params": ["v0", "v1", "a", "z", "t", "theta"],
        "param_bounds": [
            [0.0, 0.0, 1.0, 0.0, 0.0, -0.1],
            [2.5, 2.5, 3.0, 0.9, 2.0, 1.45],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 6,
        "default_params": [0.0, 0.0, 2.0, 0.5, 1e-3, 0.0],
        "nchoices": 2,
        "choices": [0, 1],
        "n_particles": 2,
        "simulator": cssm.race_model,
    },
    "race_no_z_angle_2": {
        "name": "race_no_z_angle_2",
        "params": ["v0", "v1", "a", "t", "theta"],
        "param_bounds": [
            [0.0, 0.0, 1.0, 0.0, -0.1],
            [2.5, 2.5, 3.0, 2.0, 1.45],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 5,
        "default_params": [0.0, 0.0, 2.0, 1e-3, 0.0],
        "nchoices": 2,
        "choices": [0, 1],
        "n_particles": 2,
        "simulator": cssm.race_model,
    },
    "race_3": {
        "name": "race_3",
        "params": ["v0", "v1", "v2", "a", "z0", "z1", "z2", "t"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [2.5, 2.5, 2.5, 3.0, 0.9, 0.9, 0.9, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 8,
        "default_params": [0.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.5, 1e-3],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.race_model,
    },
    "race_no_bias_3": {
        "name": "race_no_bias_3",
        "params": ["v0", "v1", "v2", "a", "z", "t"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [2.5, 2.5, 2.5, 3.0, 0.9, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 6,
        "n_particles": 3,
        "default_params": [0.0, 0.0, 0.0, 2.0, 0.5, 1e-3],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "simulator": cssm.race_model,
    },
    "race_no_z_3": {
        "name": "race_no_z_3",
        "params": ["v0", "v1", "v2", "a", "t"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [2.5, 2.5, 2.5, 3.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.0, 0.0, 0.0, 2.0, 1e-3],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.race_model,
    },
    "race_no_bias_angle_3": {
        "name": "race_no_bias_angle_3",
        "params": ["v0", "v1", "v2", "a", "z", "t", "theta"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -0.1],
            [2.5, 2.5, 2.5, 3.0, 0.9, 2.0, 1.45],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 7,
        "default_params": [0.0, 0.0, 0.0, 2.0, 0.5, 1e-3, 0.0],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.race_model,
    },
    "race_no_z_angle_3": {
        "name": "race_no_z_angle_3",
        "params": ["v0", "v1", "v2", "a", "t", "theta"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, 0.0, -0.1],
            [2.5, 2.5, 2.5, 3.0, 2.0, 1.45],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 6,
        "default_params": [0.0, 0.0, 0.0, 2.0, 1e-3, 0.0],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.race_model,
    },
    "race_4": {
        "name": "race_4",
        "params": ["v0", "v1", "v2", "v3", "a", "z0", "z1", "z2", "z3", "t"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.5, 2.5, 2.5, 2.5, 3.0, 0.9, 0.9, 0.9, 0.9, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 10,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.5, 0.5, 1e-3],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 4,
        "simulator": cssm.race_model,
    },
    "race_no_bias_4": {
        "name": "race_no_bias_4",
        "params": ["v0", "v1", "v2", "v3", "a", "z", "t"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [2.5, 2.5, 2.5, 2.5, 3.0, 0.9, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 7,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 1e-3],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 4,
        "simulator": cssm.race_model,
    },
    "race_no_z_4": {
        "name": "race_no_z_4",
        "params": ["v0", "v1", "v2", "v3", "a", "t"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [2.5, 2.5, 2.5, 2.5, 3.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 6,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 1e-3],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 4,
        "simulator": cssm.race_model,
    },
    "race_no_bias_angle_4": {
        "name": "race_no_bias_angle_4",
        "params": ["v0", "v1", "v2", "v3", "a", "z", "t", "theta"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -0.1],
            [2.5, 2.5, 2.5, 2.5, 3.0, 0.9, 2.0, 1.45],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 8,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 1e-3, 0.0],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 4,
        "simulator": cssm.race_model,
    },
    "race_no_z_angle_4": {
        "name": "race_no_z_angle_4",
        "params": ["v0", "v1", "v2", "v3", "a", "t", "theta"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1],
            [2.5, 2.5, 2.5, 2.5, 3.0, 2.0, 1.45],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 7,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 1e-3, 0.0],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 4,
        "simulator": cssm.race_model,
    },
    "lca_3": {
        "name": "lca_3",
        "params": ["v0", "v1", "v2", "a", "z0", "z1", "z2", "g", "b", "t"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0],
            [2.5, 2.5, 2.5, 3.0, 0.9, 0.9, 0.9, 1.0, 1.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 10,
        "default_params": [0.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.5, 0.0, 0.0, 1e-3],
        "nchoices": 3,
        "n_particles": 3,
        "simulator": cssm.lca,
    },
    "lca_no_bias_3": {
        "name": "lca_no_bias_3",
        "params": ["v0", "v1", "v2", "a", "z", "g", "b", "t"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, 0.0],
            [2.5, 2.5, 2.5, 3.0, 0.9, 1.0, 1.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 8,
        "default_params": [0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1e-3],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.lca,
    },
    "lca_no_z_3": {
        "name": "lca_no_z_3",
        "params": ["v0", "v1", "v2", "a", "g", "b", "t"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0],
            [2.5, 2.5, 2.5, 3.0, 1.0, 1.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 7,
        "default_params": [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1e-3],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.lca,
    },
    "lca_no_bias_angle_3": {
        "name": "lca_no_bias_angle_3",
        "params": ["v0", "v1", "v2", "a", "z", "g", "b", "t", "theta"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, 0.0, -1.0],
            [2.5, 2.5, 2.5, 3.0, 0.9, 1.0, 1.0, 2.0, 1.45],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 9,
        "default_params": [0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1e-3, 0.0],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.lca,
    },
    "lca_no_z_angle_3": {
        "name": "lca_no_z_angle_3",
        "params": ["v0", "v1", "v2", "a", "g", "b", "t", "theta"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0, -1.0],
            [2.5, 2.5, 2.5, 3.0, 1.0, 1.0, 2.0, 1.45],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 8,
        "default_params": [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1e-3, 0.0],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.lca,
    },
    "lca_4": {
        "name": "lca_4",
        "params": [
            "v0",
            "v1",
            "v2",
            "v3",
            "a",
            "z0",
            "z1",
            "z2",
            "z3",
            "g",
            "b",
            "t",
        ],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0],
            [2.5, 2.5, 2.5, 2.5, 3.0, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 12,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 1e-3],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 4,
        "simulator": cssm.lca,
    },
    "lca_no_bias_4": {
        "name": "lca_no_bias_4",
        "params": ["v0", "v1", "v2", "v3", "a", "z", "g", "b", "t"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, 0.0],
            [2.5, 2.5, 2.5, 2.5, 3.0, 0.9, 1.0, 1.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 9,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1e-3],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 4,
        "simulator": cssm.lca,
    },
    "lca_no_z_4": {
        "name": "lca_no_z_4",
        "params": ["v0", "v1", "v2", "v3", "a", "g", "b", "t"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0],
            [2.5, 2.5, 2.5, 2.5, 3.0, 1.0, 1.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 8,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1e-3],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 4,
        "simulator": cssm.lca,
    },
    "lca_no_bias_angle_4": {
        "name": "lca_no_bias_angle_4",
        "params": ["v0", "v1", "v2", "v3", "a", "z", "g", "b", "t", "theta"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, 0.0, -0.1],
            [2.5, 2.5, 2.5, 2.5, 3.0, 0.9, 1.0, 1.0, 2.0, 1.45],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 10,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1e-3, 0.0],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 4,
        "simulator": cssm.lca,
    },
    "lca_no_z_angle_4": {
        "name": "lca_no_z_angle_4",
        "params": ["v0", "v1", "v2", "v3", "a", "g", "b", "t", "theta"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0, -0.1],
            [2.5, 2.5, 2.5, 2.5, 3.0, 1.0, 1.0, 2.0, 1.45],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 9,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1e-3, 0.0],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 4,
        "simulator": cssm.lca,
    },
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
    # -----
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
    # -----
    "ddm_mic2_leak": model_config_getter["ddm_mic2_leak"],
    "ddm_mic2_leak_no_bias": model_config_getter["ddm_mic2_leak_no_bias"],
    "ddm_mic2_leak_conflict_gamma_no_bias": model_config_getter[
        "ddm_mic2_leak_conflict_gamma_no_bias"
    ],
    "ddm_mic2_leak_angle_no_bias": model_config_getter["ddm_mic2_leak_angle_no_bias"],
    "ddm_mic2_leak_weibull_no_bias": model_config_getter[
        "ddm_mic2_leak_weibull_no_bias"
    ],
    # -----
    "tradeoff_no_bias": {
        "name": "tradeoff_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "d", "t"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.0, 0.0],
            [4.0, 4.0, 4.0, 2.5, 1.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 6,
        "default_params": [0.0, 0.0, 0.0, 1.0, 0.5, 1.0],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_tradeoff,
    },
    "tradeoff_angle_no_bias": {
        "name": "tradeoff_angle_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "d", "t", "theta"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.0, 0.0, -0.1],
            [4.0, 4.0, 4.0, 2.5, 1.0, 2.0, 1.0],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "boundary_multiplicative": False,
        "n_params": 7,
        "default_params": [0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_tradeoff,
    },
    "tradeoff_weibull_no_bias": {
        "name": "tradeoff_weibull_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "d", "t", "alpha", "beta"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.0, 0.0, 0.31, 0.31],
            [4.0, 4.0, 4.0, 2.5, 1.0, 2.0, 4.99, 6.99],
        ],
        "boundary_name": "weibull_cdf",
        "boundary": bf.weibull_cdf,
        "boundary_multiplicative": True,
        "n_params": 8,
        "default_params": [0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 2.5, 3.5],
        "nchoices": 4,
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_tradeoff,
    },
    "tradeoff_conflict_gamma_no_bias": {
        "name": "tradeoff_conflict_gamma_no_bias",
        "params": [
            "vh",
            "vl1",
            "vl2",
            "d",
            "t",
            "a",
            "theta",
            "scale",
            "alphagamma",
            "scalegamma",
        ],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.0, 0.0, 0.3, 0.0, 0.0, 1.1, 0.5],
            [4.0, 4.0, 4.0, 1.0, 2.0, 2.5, 0.5, 5.0, 5.0, 5.0],
        ],
        "boundary_name": "conflict_gamma",
        "boundary": bf.conflict_gamma,
        "boundary_multiplicative": True,
        "n_params": 10,
        "default_params": [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 2, 2],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_tradeoff,
    },
    # "glob": {
    #     "name": "glob",
    #     "params": ["v", "a", "z", "alphar", "g", "t", "theta"],
    #     "param_bounds": [
    #         [-3.0, 0.3, 0.15, 1.0, -1.0, 1e-5, 0.0],
    #         [3.0, 2.0, 0.85, 2.0, 1.0, 2.0, 1.45],
    #     ],
    #     "n_params": 7,
    #     "default_params": [0.0, 1.0, 0.5, 2.0, 0.0, 1.0, 2.5, 3.5],
    #     "hddm_include": ["z", "alphar", "g", "theta"],
    #     "nchoices": 2,
    #     "boundary_name": "angle",
    #     "boundary": bf.angle,
    #     "boundary_multiplicative": False,
    #     "components": {
    #         "names": ["g", "alphar", "theta"],
    #         "off_values": np.float32(np.array([0, 1, 0])),
    #         "probabilities": np.array([1 / 3, 1 / 3, 1 / 3]),
    #         "labels": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    #         "n_components": 3,
    #     },
    # },
}

model_config["weibull_cdf"] = model_config["weibull"].copy()
model_config["full_ddm2"] = model_config["full_ddm"].copy()
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

#### DATASET GENERATOR CONFIGS --------------------------

kde_simulation_filters = {
    "mode": 20,  # != (if mode is max_rt)
    "choice_cnt": 0,  # > (each choice receive at least 10 samples )
    "mean_rt": 17,  # < (mean_rt is smaller than specified value
    "std": 0,  # > (std is positive for each choice)
    "mode_cnt_rel": 0.95,  # < (mode can't be large proportion of all samples)
}

data_generator_config = {
    "opn_only": {
        "output_folder": "data/cpn_only/",
        "model": "ddm",  # should be ['ddm'],
        "n_samples": 100000,  # eventually should be {'low': 100000, 'high': 100000},
        "n_parameter_sets": 10000,
        "n_parameter_sets_rejected": 100,
        "n_training_samples_by_parameter_set": 1000,
        "max_t": 20.0,
        "delta_t": 0.001,
        "pickleprotocol": 4,
        "n_cpus": "all",
        "negative_rt_cutoff": -66.77497,
        "n_subruns": 10,
        "smooth_unif": False,
    },
    "cpn_only": {
        "output_folder": "data/cpn_only/",
        "model": "ddm",  # should be ['ddm'],
        "n_samples": 100000,  # eventually should be {'low': 100000, 'high': 100000},
        "n_parameter_sets": 10000,
        "n_parameter_sets_rejected": 100,
        "n_training_samples_by_parameter_set": 1000,
        "max_t": 20.0,
        "delta_t": 0.001,
        "pickleprotocol": 4,
        "n_cpus": "all",
        "negative_rt_cutoff": -66.77497,
        "n_subruns": 10,
        "smooth_unif": False,
    },
    "lan": {
        "output_folder": "data/lan_mlp/",
        "model": "ddm",  # should be ['ddm'],
        "nbins": 0,
        "n_samples": 100000,  # eventually should be {'low': 100000, 'high': 100000},
        "n_parameter_sets": 10000,
        "n_parameter_sets_rejected": 100,
        "n_training_samples_by_parameter_set": 1000,
        "max_t": 20.0,
        "delta_t": 0.001,
        "pickleprotocol": 4,
        "n_cpus": "all",
        "kde_data_mixture_probabilities": [0.8, 0.1, 0.1],
        "simulation_filters": kde_simulation_filters,
        "negative_rt_cutoff": -66.77497,
        "n_subruns": 10,
        "bin_pointwise": False,
        "separate_response_channels": False,
        "smooth_unif": True,
        "kde_displace_t": False,
    },
    "defective_detector": {
        "output_folder": "data/defective_detector/",
        "model": "ddm",
        "nbins": 0,
        "n_samples": {"low": 100000, "high": 100000},
        "n_parameter_sets": 100000,
        "n_parameter_sets_rejected": 100,
        "n_training_samples_by_parameter_set": 1000,
        "max_t": 20.0,
        "delta_t": 0.001,
        "pickleprotocol": 4,
        "n_cpus": "all",
        "n_subdatasets": 12,
        "n_trials_per_dataset": 10000,  # EVEN NUMBER ! AF-TODO: Saveguard against odd
        "kde_data_mixture_probabilities": [0.8, 0.1, 0.1],
        "simulation_filters": kde_simulation_filters,
        "negative_rt_cutoff": -66.77497,
        "n_subruns": 10,
        "bin_pointwise": False,
        "separate_response_channels": False,
    },
}
