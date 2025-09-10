"""Full DDM model configuration."""

import functools
import scipy.stats as sps  # type: ignore

import cssm
from ssms.basic_simulators import boundary_functions as bf
from ssms.config.registry import register_model


@register_model("full_ddm")
def get_full_ddm_config():
    """Get the configuration for the Full DDM model."""
    return {
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
    }


@register_model("full_ddm_rv")
def get_full_ddm_rv_config():
    """Get configuration for full DDM with random variables."""
    return {
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
    }
