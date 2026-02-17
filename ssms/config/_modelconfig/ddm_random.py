"""Configuration for DDM models with random variables."""

import functools
import numpy as np
import scipy.stats as sps

import cssm
from ssms.basic_simulators import boundary_functions as bf
from ssms.transforms import ApplyMapping, LambdaAdaptation


def get_ddm_st_config():
    """Get configuration for DDM with random non-decision time."""
    return {
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
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update(
                            {
                                "z_dist": cfg["simulator_fixed_params"]["z_dist"],
                                "v_dist": cfg["simulator_fixed_params"]["v_dist"],
                            }
                        )
                        or theta
                    ),
                    name="set_fixed_params",
                ),
                ApplyMapping("st", "t_dist", "t_dist"),
            ],
        },
    }


def get_ddm_truncnormt_config():
    """Get configuration for DDM with truncated normal non-decision time."""
    return {
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
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update(
                            {
                                "z_dist": cfg["simulator_fixed_params"]["z_dist"],
                                "v_dist": cfg["simulator_fixed_params"]["v_dist"],
                                "t": np.array([0], dtype=np.float32),
                            }
                        )
                        or theta
                    ),
                    name="set_fixed_params_and_zero_t",
                ),
                ApplyMapping("mt", "t_dist", "t_dist", additional_sources=["st"]),
            ],
        },
    }


def get_ddm_rayleight_config():
    """Get configuration for DDM with Rayleigh non-decision time."""
    return {
        "name": "ddm_rayleight",
        "params": ["v", "a", "z", "st"],
        "param_bounds": [
            [-3.0, 0.3, 0.3, 1e-3],
            [3.0, 2.5, 0.7, 1.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 4,
        "default_params": [0.0, 1.0, 0.5, 0.2],
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
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update(
                            {
                                "z_dist": cfg["simulator_fixed_params"]["z_dist"],
                                "v_dist": cfg["simulator_fixed_params"]["v_dist"],
                                "t": (
                                    np.ones(n) * cfg["simulator_fixed_params"]["t"]
                                ).astype(np.float32),
                            }
                        )
                        or theta
                    ),
                    name="set_fixed_params_and_t",
                ),
                ApplyMapping("st", "t_dist", "t_dist"),
            ],
        },
    }


def get_ddm_sdv_config():
    """Get configuration for DDM with random drift rate."""
    return {
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
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update(
                            {
                                "z_dist": cfg["simulator_fixed_params"]["z_dist"],
                                "t_dist": cfg["simulator_fixed_params"]["t_dist"],
                            }
                        )
                        or theta
                    ),
                    name="set_fixed_dists",
                ),
                ApplyMapping("sv", "v_dist", "v_dist"),
            ],
        },
    }
