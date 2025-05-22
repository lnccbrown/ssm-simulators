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

import functools

import cssm
import numpy as np
import scipy.stats as sps  # type: ignore

from ssms.basic_simulators import boundary_functions as bf
from ssms.basic_simulators import drift_functions as df


def boundary_config_to_function_params(boundary_config: dict) -> dict:
    """
    Convert boundary configuration to function parameters.

    Parameters
    ----------
    boundary_config: dict
        Dictionary containing the boundary configuration

    Returns
    -------
    dict
        Dictionary with adjusted key names so that they match function parameters names
        directly.
    """
    return {"boundary_" + k: boundary_config[k] for k in boundary_config.keys()}


boundary_config = {
    "constant": {
        "fun": bf.constant,
        "params": [],
        "multiplicative": True,
    },
    "angle": {
        "fun": bf.angle,
        "params": ["theta"],
        "multiplicative": False,
    },
    "weibull_cdf": {
        "fun": bf.weibull_cdf,
        "params": ["alpha", "beta"],
        "multiplicative": True,
    },
    "generalized_logistic": {
        "fun": bf.generalized_logistic,
        "params": ["B", "M", "v"],
        "multiplicative": True,
    },
    "conflict_gamma": {
        "fun": bf.conflict_gamma,
        "params": ["theta", "scale", "alpha_gamma", "scale_gamma"],
        "multiplicative": False,
    },
}

drift_config = {
    "constant": {
        "fun": df.constant,
        "params": [],
    },
    "gamma_drift": {
        "fun": df.gamma_drift,
        "params": ["shape", "scale", "c"],
    },
    "ds_conflict_drift": {
        "fun": df.ds_conflict_drift,
        "params": ["tinit", "dinit", "tslope", "dslope", "tfixedp", "tcoh", "dcoh"],
    },
    "attend_drift": {
        "fun": df.attend_drift,
        "params": ["ptarget", "pouter", "pinner", "r", "sda"],
    },
    "attend_drift_simple": {
        "fun": df.attend_drift_simple,
        "params": ["ptarget", "pouter", "r", "sda"],
    },
}

# Configuration dictionary for simulators
model_config = {
    "ddm": {
        "name": "ddm",
        "params": ["v", "a", "z", "t"],
        "param_bounds": [[-3.0, 0.3, 0.1, 0.0], [3.0, 2.5, 0.9, 2.0]],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "boundary_params": [],
        "n_params": 4,
        "default_params": [0.0, 1.0, 0.5, 1e-3],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound,
    },
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
    "angle": {
        "name": "angle",
        "params": ["v", "a", "z", "t", "theta"],
        "param_bounds": [[-3.0, 0.3, 0.1, 1e-3, -0.1], [3.0, 3.0, 0.9, 2.0, 1.3]],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 5,
        "default_params": [0.0, 1.0, 0.5, 1e-3, 0.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound,
    },
    "weibull": {
        "name": "weibull",
        "params": ["v", "a", "z", "t", "alpha", "beta"],
        "param_bounds": [
            [-2.5, 0.3, 0.2, 1e-3, 0.31, 0.31],
            [2.5, 2.5, 0.8, 2.0, 4.99, 6.99],
        ],
        "boundary_name": "weibull_cdf",
        "boundary": bf.weibull_cdf,
        "n_params": 6,
        "default_params": [0.0, 1.0, 0.5, 1e-3, 3.0, 3.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound,
    },
    "levy": {
        "name": "levy",
        "params": ["v", "a", "z", "alpha", "t"],
        "param_bounds": [[-3.0, 0.3, 0.1, 1.0, 1e-3], [3.0, 3.0, 0.9, 2.0, 2]],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.0, 1.0, 0.5, 1.5, 0.1],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.levy_flexbound,
    },
    "levy_angle": {
        "name": "levy_angle",
        "params": ["v", "a", "z", "alpha", "t", "theta"],
        "param_bounds": [
            [-3.0, 0.3, 0.1, 1.0, 1e-3, -0.1],
            [3.0, 3.0, 0.9, 2.0, 2, 1.3],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 6,
        "default_params": [0.0, 1.0, 0.5, 1.5, 0.1, 0.01],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.levy_flexbound,
    },
    "full_ddm": {
        "name": "full_ddm",
        "params": ["v", "a", "z", "t", "sz", "sv", "st"],
        "param_bounds": [
            [-3.0, 0.3, 0.3, 0.25, 1e-3, 1e-3, 1e-3],
            [3.0, 2.5, 0.7, 2.25, 0.2, 2.0, 0.25],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 7,
        "default_params": [0.0, 1.0, 0.5, 0.25, 1e-3, 1e-3, 1e-3],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.full_ddm,
    },
    "full_ddm_rv": {
        "name": "full_ddm_rv",
        "params": ["v", "a", "z", "t", "sz", "sv", "st"],
        "param_bounds": {
            "v": (-3.0, 3.0),
            "a": (0.3, 2.5),
            "z": (0.3, 0.7),
            "t": (0.25, 2.25),
            "sz": (1e-3, 0.2),
            "sv": (1e-3, 2.0),
            "st": (1e-3, "t"),
        },
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 6,
        "default_params": [0.0, 1.0, 0.5, 0.25, 1e-3, 1e-3, 1e-3],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.full_ddm_rv,
        "simulator_fixed_params": {},
        "simulator_param_mappings": {
            "t_dist": lambda st: functools.partial(
                sps.uniform.rvs, loc=(-1) * st, scale=2 * st
            ),
            "v_dist": lambda sv: functools.partial(
                sps.norm.rvs,
                loc=0,
                scale=sv,
            ),
            "z_dist": lambda sz: functools.partial(
                sps.uniform.rvs, loc=(-1) * sz, scale=2 * sz
            ),
        },
    },
    "ddm_st": {
        "name": "ddm_st",
        "params": ["v", "a", "z", "t", "st"],
        "param_bounds": [
            [-3.0, 0.3, 0.3, 0.25, 1e-3],
            [3.0, 2.5, 0.7, 2.25, 0.25],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.0, 1.0, 0.5, 0.25, 1e-3],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.full_ddm_rv,
        "simulator_fixed_params": {
            "z_dist": functools.partial(sps.norm.rvs, loc=0, scale=0),
            "v_dist": functools.partial(sps.norm.rvs, loc=0, scale=0),
        },
        "simulator_param_mappings": {
            "t_dist": lambda st: functools.partial(
                sps.uniform.rvs, loc=(-1) * st, scale=2 * st
            ),
        },
    },
    "ddm_truncnormt": {
        "name": "ddm_truncnormt",
        "params": ["v", "a", "z", "mt", "st"],
        "param_bounds": [
            [-3.0, 0.3, 0.3, 0.05, 1e-3],
            [3.0, 2.5, 0.7, 2.25, 0.5],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.0, 1.0, 0.5, 0.25, 1e-3],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.full_ddm_rv,
        "simulator_fixed_params": {
            "z_dist": functools.partial(sps.norm.rvs, loc=0, scale=0),
            "v_dist": functools.partial(sps.norm.rvs, loc=0, scale=0),
            "t": 0.0,
        },
        "simulator_param_mappings": {
            "t_dist": lambda mt, st: functools.partial(
                sps.truncnorm.rvs,
                a=(-1) * np.divide(mt, st),
                b=np.inf,
                loc=mt,
                scale=st,
            ),
        },
    },
    "ddm_rayleight": {
        "name": "ddm_rayleight",
        "params": ["v", "a", "z", "st"],
        "param_bounds": [
            [-3.0, 0.3, 0.3, 1e-3],
            [3.0, 2.5, 0.7, 1.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.0, 1.0, 0.5, 0.25, 0.2],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.full_ddm_rv,
        "simulator_fixed_params": {
            "z_dist": functools.partial(sps.norm.rvs, loc=0, scale=0),
            "v_dist": functools.partial(sps.norm.rvs, loc=0, scale=0),
            "t": 0.0,
        },
        "simulator_param_mappings": {
            "t_dist": lambda st: functools.partial(
                sps.rayleigh.rvs,
                loc=0,
                scale=st,
            ),
        },
    },
    "ddm_sdv": {
        "name": "ddm_sdv",
        "params": ["v", "a", "z", "t", "sv"],
        "param_bounds": [[-3.0, 0.3, 0.1, 1e-3, 1e-3], [3.0, 2.5, 0.9, 2.0, 2.5]],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.0, 1.0, 0.5, 1e-3, 1e-3],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.full_ddm_rv,
        "simulator_fixed_params": {
            "z_dist": functools.partial(sps.norm.rvs, loc=0, scale=0),
            "t_dist": functools.partial(sps.norm.rvs, loc=0, scale=0),
        },
        "simulator_param_mappings": {
            "v_dist": lambda sv: functools.partial(
                sps.norm.rvs,
                loc=0,
                scale=sv,
            ),
        },
    },
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
    "ddm_par2": {
        "name": "ddm_par2",
        "params": ["vh", "vl1", "vl2", "a", "zh", "zl1", "zl2", "t"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.2, 0.2, 0.2, 0.0],
            [4.0, 4.0, 4.0, 2.5, 0.8, 0.8, 0.8, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 8,
        "default_params": [0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 1.0],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_par2,
    },
    "ddm_par2_no_bias": {
        "name": "ddm_par2_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "t"],
        "param_bounds": [[-4.0, -4.0, -4.0, 0.3, 0.0], [4.0, 4.0, 4.0, 2.5, 2.0]],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.0, 0.0, 0.0, 1.0, 1.0],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_par2,
    },
    "ddm_par2_conflict_gamma_no_bias": {
        "name": "ddm_par2_conflict_gamma_no_bias",
        "params": [
            "vh",
            "vl1",
            "vl2",
            "t",
            "a",
            "theta",
            "scale",
            "alpha_gamma",
            "scale_gamma",
        ],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.0, 0.3, 0.0, 0.0, 1.1, 0.5],
            [4.0, 4.0, 4.0, 2.0, 2.5, 0.5, 5.0, 5.0, 5.0],
        ],
        "boundary_name": "conflict_gamma",
        "boundary": bf.conflict_gamma,
        "n_params": 9,
        "default_params": [0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 1.0, 2, 2],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_par2,
    },
    "ddm_par2_angle_no_bias": {
        "name": "ddm_par2_angle_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "t", "theta"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.0, -0.1],
            [4.0, 4.0, 4.0, 2.5, 2.0, 1.0],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "boundary_multiplicative": False,
        "n_params": 6,
        "default_params": [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_par2,
    },
    "ddm_par2_weibull_no_bias": {
        "name": "ddm_par2_weibull_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "t", "alpha", "beta"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.0, 0.31, 0.31],
            [4.0, 4.0, 4.0, 2.5, 2.0, 4.99, 6.99],
        ],
        "boundary_name": "weibull_cdf",
        "boundary": bf.weibull_cdf,
        "boundary_multiplicative": True,
        "n_params": 7,
        "default_params": [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 3.0],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_par2,
    },
    "ddm_seq2": {
        "name": "ddm_seq2",
        "params": ["vh", "vl1", "vl2", "a", "zh", "zl1", "zl2", "t"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.2, 0.2, 0.2, 0.0],
            [4.0, 4.0, 4.0, 2.5, 0.8, 0.8, 0.8, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 8,
        "default_params": [0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 1.0],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_seq2,
    },
    "ddm_seq2_no_bias": {
        "name": "ddm_seq2_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "t"],
        "param_bounds": [[-4.0, -4.0, -4.0, 0.3, 0.0], [4.0, 4.0, 4.0, 2.5, 2.0]],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.0, 0.0, 0.0, 1.0, 1.0],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_seq2,
    },
    "ddm_seq2_conflict_gamma_no_bias": {
        "name": "ddm_seq2_conflict_gamma_no_bias",
        "params": [
            "vh",
            "vl1",
            "vl2",
            "t",
            "a",
            "theta",
            "scale",
            "alpha_gamma",
            "scale_gamma",
        ],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.0, 0.3, 0.0, 0.0, 1.1, 0.5],
            [4.0, 4.0, 4.0, 2.0, 2.5, 0.5, 5.0, 5.0, 5.0],
        ],
        "boundary_name": "conflict_gamma",
        "boundary": bf.conflict_gamma,
        "n_params": 9,
        "default_params": [0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 1.0, 2, 2],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_seq2,
    },
    "ddm_seq2_angle_no_bias": {
        "name": "ddm_seq2_angle_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "t", "theta"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.0, -0.1],
            [4.0, 4.0, 4.0, 2.5, 2.0, 1.0],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "boundary_multiplicative": False,
        "n_params": 6,
        "default_params": [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_seq2,
    },
    "ddm_seq2_weibull_no_bias": {
        "name": "ddm_seq2_weibull_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "t", "alpha", "beta"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.0, 0.31, 0.31],
            [4.0, 4.0, 4.0, 2.5, 2.0, 4.99, 6.99],
        ],
        "boundary_name": "weibull_cdf",
        "boundary": bf.weibull_cdf,
        "boundary_multiplicative": True,
        "n_params": 7,
        "default_params": [0.0, 0.0, 0.0, 1.0, 1.0, 2.5, 3.5],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_seq2,
    },
    "ddm_mic2_adj": {
        "name": "ddm_mic2_adj",
        "params": ["vh", "vl1", "vl2", "a", "zh", "zl1", "zl2", "d", "t"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.2, 0.2, 0.2, 0.0, 0.0],
            [4.0, 4.0, 4.0, 2.5, 0.8, 0.8, 0.8, 1.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 9,
        "default_params": [0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_mic2_ornstein,
    },
    "ddm_mic2_adj_no_bias": {
        "name": "ddm_mic2_adj_no_bias",
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
        "simulator": cssm.ddm_flexbound_mic2_ornstein,
    },
    "ddm_mic2_adj_conflict_gamma_no_bias": {
        "name": "ddm_mic2_adj_conflict_gamma_no_bias",
        "params": [
            "vh",
            "vl1",
            "vl2",
            "d",
            "t",
            "a",
            "theta",
            "scale",
            "alpha_gamma",
            "scale_gamma",
        ],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.0, 0.0, 0.3, 0.0, 0.0, 1.1, 0.5],
            [4.0, 4.0, 4.0, 1.0, 2.0, 2.5, 0.5, 5.0, 5.0, 5.0],
        ],
        "boundary_name": "conflict_gamma",
        "boundary": bf.conflict_gamma,
        "n_params": 10,
        "default_params": [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 2, 2],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_mic2_ornstein,
    },
    "ddm_mic2_adj_angle_no_bias": {
        "name": "ddm_mic2_adj_angle_no_bias",
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
        "simulator": cssm.ddm_flexbound_mic2_ornstein,
    },
    "ddm_mic2_adj_weibull_no_bias": {
        "name": "ddm_mic2_adj_weibull_no_bias",
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
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_mic2_ornstein,
    },
    "ddm_mic2_ornstein": {
        "name": "ddm_mic2_ornstein",
        "params": ["vh", "vl1", "vl2", "a", "zh", "zl1", "zl2", "d", "g", "t"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0],
            [4.0, 4.0, 4.0, 2.5, 0.8, 0.8, 0.8, 1.0, 3.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 10,
        "default_params": [0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.5, 0.5],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_mic2_ornstein,
    },
    "ddm_mic2_ornstein_no_bias": {
        "name": "ddm_mic2_ornstein_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "d", "g", "t"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.0, 0.0, 0.0],
            [4.0, 4.0, 4.0, 2.5, 1.0, 3.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 7,
        "default_params": [0.0, 0.0, 0.0, 1.0, 0.5, 1.5, 1.0],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_mic2_ornstein,
    },
    "ddm_mic2_ornstein_conflict_gamma_no_bias": {
        "name": "ddm_mic2_ornstein_conflict_gamma_no_bias",
        "params": [
            "vh",
            "vl1",
            "vl2",
            "d",
            "g",
            "t",
            "a",
            "theta",
            "scale",
            "alpha_gamma",
            "scale_gamma",
        ],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 1.1, 0.5],
            [4.0, 4.0, 4.0, 1.0, 3.0, 2.0, 2.5, 0.5, 5.0, 5.0, 5.0],
        ],
        "boundary_name": "conflict_gamma",
        "boundary": bf.conflict_gamma,
        "n_params": 11,
        "default_params": [0.0, 0.0, 0.0, 0.5, 1.5, 1.0, 1.0, 1.0, 1.0, 2, 2],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_mic2_ornstein,
    },
    "ddm_mic2_ornstein_angle_no_bias": {
        "name": "ddm_mic2_ornstein_angle_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "d", "g", "t", "theta"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.0, 0.0, 0.0, -0.1],
            [4.0, 4.0, 4.0, 2.5, 1.0, 3.0, 2.0, 1.0],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "boundary_multiplicative": False,
        "n_params": 8,
        "default_params": [0.0, 0.0, 0.0, 1.0, 0.5, 1.5, 1.0, 0.0],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_mic2_ornstein,
    },
    "ddm_mic2_ornstein_weibull_no_bias": {
        "name": "ddm_mic2_ornstein_weibull_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "d", "g", "t", "alpha", "beta"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.0, 0.0, 0.0, 0.31, 0.31],
            [4.0, 4.0, 4.0, 2.5, 1.0, 3.0, 2.0, 4.99, 6.99],
        ],
        "boundary_name": "weibull_cdf",
        "boundary": bf.weibull_cdf,
        "boundary_multiplicative": True,
        "n_params": 9,
        "default_params": [0.0, 0.0, 0.0, 1.0, 0.5, 1.5, 1.0, 2.5, 3.5],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_mic2_ornstein,
    },
    # -----
    "ddm_mic2_multinoise_no_bias": {
        "name": "ddm_mic2_multinoise_no_bias",
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
        "simulator": cssm.ddm_flexbound_mic2_multinoise,
    },
    "ddm_mic2_multinoise_conflict_gamma_no_bias": {
        "name": "ddm_mic2_multinoise_conflict_gamma_no_bias",
        "params": [
            "vh",
            "vl1",
            "vl2",
            "d",
            "t",
            "a",
            "theta",
            "scale",
            "alpha_gamma",
            "scale_gamma",
        ],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.0, 0.0, 0.3, 0.0, 0.0, 1.1, 0.5],
            [4.0, 4.0, 4.0, 1.0, 2.0, 2.5, 0.5, 5.0, 5.0, 5.0],
        ],
        "boundary_name": "conflict_gamma",
        "boundary": bf.conflict_gamma,
        "n_params": 10,
        "default_params": [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 2, 2],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_mic2_multinoise,
    },
    "ddm_mic2_multinoise_angle_no_bias": {
        "name": "ddm_mic2_multinoise_angle_no_bias",
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
        "simulator": cssm.ddm_flexbound_mic2_multinoise,
    },
    "ddm_mic2_multinoise_weibull_no_bias": {
        "name": "ddm_mic2_multinoise_weibull_no_bias",
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
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_mic2_multinoise,
    },
    # -----
    "ddm_mic2_leak": {
        "name": "ddm_mic2_leak",
        "params": ["vh", "vl1", "vl2", "a", "zh", "zl1", "zl2", "d", "t"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.2, 0.2, 0.2, 0.0, 0.0],
            [4.0, 4.0, 4.0, 2.5, 0.8, 0.8, 0.8, 1.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 9,
        "default_params": [0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_mic2_ornstein,
    },
    "ddm_mic2_leak_no_bias": {
        "name": "ddm_mic2_leak_no_bias",
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
        "simulator": cssm.ddm_flexbound_mic2_ornstein,
    },
    "ddm_mic2_leak_conflict_gamma_no_bias": {
        "name": "ddm_mic2_leak_conflict_gamma_no_bias",
        "params": [
            "vh",
            "vl1",
            "vl2",
            "d",
            "t",
            "a",
            "theta",
            "scale",
            "alpha_gamma",
            "scale_gamma",
        ],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.0, 0.0, 0.3, 0.0, 0.0, 1.1, 0.5],
            [4.0, 4.0, 4.0, 1.0, 2.0, 2.5, 0.5, 5.0, 5.0, 5.0],
        ],
        "boundary_name": "conflict_gamma",
        "boundary": bf.conflict_gamma,
        "n_params": 10,
        "default_params": [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 2, 2],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_mic2_ornstein,
    },
    "ddm_mic2_leak_angle_no_bias": {
        "name": "ddm_mic2_leak_angle_no_bias",
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
        "simulator": cssm.ddm_flexbound_mic2_ornstein,
    },
    "ddm_mic2_leak_weibull_no_bias": {
        "name": "ddm_mic2_leak_weibull_no_bias",
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
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_mic2_ornstein,
    },
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

# TODO: Move to a separate file
"""Data generator configuration.

Convenience functions for getting default configurations for data generation.
"""


class DeprecatedDict(dict):
    """
    A pseudo-dictionary that raises a DeprecationWarning when accessed.
    This is used to indicate that the configuration dictionary is deprecated
    and should not be used directly.

    Parameters
    ----------
    lookup_func : callable, optional
        A function that takes a key and returns the corresponding value.
    alternative : str, optional
        A string indicating the alternative method to use instead of this
        configuration dictionary."""

    def __init__(self, lookup_func=None, alternative="get_default_generator_config"):
        self._lookup_func = lookup_func
        self._alternative = alternative

    def __getitem__(self, key):
        message = f"Accessing this configuration dict is deprecated and will be removed in a future version. Use `{self._alternative}` instead."
        from warnings import warn

        warn(
            message,
            DeprecationWarning,
            stacklevel=2,
        )
        if self._lookup_func is None or not callable(self._lookup_func):
            raise ValueError("A valid callable lookup_func must be provided.")
        return self._lookup_func(key)


def get_kde_simulation_filters() -> dict:
    return {
        "mode": 20,  # != (if mode is max_rt)
        "choice_cnt": 0,  # > (each choice receive at least 10 samples )
        "mean_rt": 17,  # < (mean_rt is smaller than specified value
        "std": 0,  # > (std is positive for each choice)
        "mode_cnt_rel": 0.95,  # < (mode can't be large proportion of all samples)
    }


def get_opn_only_config() -> dict:
    return {
        "output_folder": "data/cpn_only/",
        "model": "ddm",  # should be ['ddm'],
        "n_samples": 100_000,  # eventually should be {'low': 100000, 'high': 100000},
        "n_parameter_sets": 10_000,
        "n_parameter_sets_rejected": 100,
        "n_training_samples_by_parameter_set": 1_000,
        "max_t": 20.0,
        "delta_t": 0.001,
        "pickleprotocol": 4,
        "n_cpus": "all",
        "negative_rt_cutoff": -66.77497,
        "n_subruns": 10,
        "smooth_unif": False,
    }


def get_cpn_only_config() -> dict:
    return {
        "output_folder": "data/cpn_only/",
        "model": "ddm",  # should be ['ddm'],
        "n_samples": 100_000,  # eventually should be {'low': 100000, 'high': 100000},
        "n_parameter_sets": 10_000,
        "n_parameter_sets_rejected": 100,
        "n_training_samples_by_parameter_set": 1_000,
        "max_t": 20.0,
        "delta_t": 0.001,
        "pickleprotocol": 4,
        "n_cpus": "all",
        "negative_rt_cutoff": -66.77497,
        "n_subruns": 10,
        "smooth_unif": False,
    }


def get_lan_config() -> dict:
    return {
        "output_folder": "data/lan_mlp/",
        "model": "ddm",  # should be ['ddm'],
        "nbins": 0,
        "n_samples": 100_000,  # eventually should be {'low': 100000, 'high': 100000},
        "n_parameter_sets": 10_000,
        "n_parameter_sets_rejected": 100,
        "n_training_samples_by_parameter_set": 1_000,
        "max_t": 20.0,
        "delta_t": 0.001,
        "pickleprotocol": 4,
        "n_cpus": "all",
        "kde_data_mixture_probabilities": [0.8, 0.1, 0.1],
        "simulation_filters": get_kde_simulation_filters(),
        "negative_rt_cutoff": -66.77497,
        "n_subruns": 10,
        "bin_pointwise": False,
        "separate_response_channels": False,
        "smooth_unif": True,
        "kde_displace_t": False,
    }


def get_ratio_estimator_config() -> dict:
    return {
        "output_folder": "data/ratio/",
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
        "simulation_filters": get_kde_simulation_filters(),
        "negative_rt_cutoff": -66.77497,
        "n_subruns": 10,
        "bin_pointwise": False,
        "separate_response_channels": False,
    }


def get_defective_detector_config() -> dict:
    return {
        "output_folder": "data/defective_detector/",
        "model": "ddm",
        "nbins": 0,
        "n_samples": {"low": 100_000, "high": 100_000},
        "n_parameter_sets": 100_000,
        "n_parameter_sets_rejected": 100,
        "n_training_samples_by_parameter_set": 1_000,
        "max_t": 20.0,
        "delta_t": 0.001,
        "pickleprotocol": 4,
        "n_cpus": "all",
        "n_subdatasets": 12,
        "n_trials_per_dataset": 10000,  # EVEN NUMBER ! AF-TODO: Saveguard against odd
        "kde_data_mixture_probabilities": [0.8, 0.1, 0.1],
        "simulation_filters": get_kde_simulation_filters(),
        "negative_rt_cutoff": -66.77497,
        "n_subruns": 10,
        "bin_pointwise": False,
        "separate_response_channels": False,
    }


def get_snpe_config() -> dict:
    return {
        "output_folder": "data/snpe_training/",
        "model": "ddm",  # should be ['ddm'],
        "n_samples": 5000,  # eventually should be {'low': 100000, 'high': 100000},
        "n_parameter_sets": 10000,
        "max_t": 20.0,
        "delta_t": 0.001,
        "pickleprotocol": 4,
        "n_cpus": "all",
        "n_subruns": 10,
        "separate_response_channels": False,
    }


def get_default_generator_config(approach) -> dict:
    """
    Dynamically retrieve the data generator configuration for the given approach.

    Parameters
    ----------
    approach : str
        The approach corresponding to the desired data generator configuration.
        Valid options include:
        - "opn_only"
        - "cpn_only"
        - "lan"
        - "ratio_estimator"
        - "defective_detector"
        - "snpe"

    Returns
    -------
    dict
        The configuration dictionary for the specified approach.

    Raises
    ------
    KeyError
        If the approach is not found in the available configurations.
    """
    config_functions = {
        "opn_only": get_opn_only_config,
        "cpn_only": get_cpn_only_config,
        "lan": get_lan_config,
        "ratio_estimator": get_ratio_estimator_config,
        "defective_detector": get_defective_detector_config,
        "snpe": get_snpe_config,
    }

    if approach not in config_functions:
        raise KeyError(
            f"'{approach}' is not a valid data generator configuration approach."
        )

    return config_functions[approach]()


# TODO: Add for compatibility with lanfactory's test_end_to_end.py test. Delete when
#       lanfactory uses get_default_generator_config.
data_generator_config = DeprecatedDict(
    get_default_generator_config, "get_default_generator_config"
)
