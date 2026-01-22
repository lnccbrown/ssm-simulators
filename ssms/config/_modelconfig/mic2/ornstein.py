"""Configuration for DDM mic2 models."""

import cssm
from ssms.basic_simulators import boundary_functions as bf
from ssms.transforms import LambdaAdaptation

import numpy as np


def get_ddm_mic2_ornstein_config():
    """Get configuration for DDM mic2 ornstein model."""
    return {
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
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update(
                            {
                                "s_pre_high_level_choice": np.tile(
                                    np.array([1.0], dtype=np.float32), n
                                )
                            }
                        )
                        or theta
                    ),
                    name="add_ornstein_params",
                )
            ],
        },
    }


def get_ddm_mic2_ornstein_no_bias_config():
    """Get configuration for DDM mic2 ornstein no bias model."""
    return {
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
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update(
                            {
                                "zh": np.tile(np.array([0.5], dtype=np.float32), n),
                                "zl1": np.tile(np.array([0.5], dtype=np.float32), n),
                                "zl2": np.tile(np.array([0.5], dtype=np.float32), n),
                            }
                        )
                        or theta
                    ),
                    name="add_z_defaults",
                ),
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update(
                            {
                                "s_pre_high_level_choice": np.tile(
                                    np.array([1.0], dtype=np.float32), n
                                )
                            }
                        )
                        or theta
                    ),
                    name="add_ornstein_params",
                ),
            ],
        },
    }


def get_ddm_mic2_ornstein_conflict_gamma_no_bias_config():
    """Get configuration for DDM mic2 ornstein conflict gamma no bias model."""
    return {
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
            "alphaGamma",
            "scaleGamma",
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
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update(
                            {
                                "zh": np.tile(np.array([0.5], dtype=np.float32), n),
                                "zl1": np.tile(np.array([0.5], dtype=np.float32), n),
                                "zl2": np.tile(np.array([0.5], dtype=np.float32), n),
                            }
                        )
                        or theta
                    ),
                    name="add_z_defaults",
                ),
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update(
                            {
                                "s_pre_high_level_choice": np.tile(
                                    np.array([1.0], dtype=np.float32), n
                                )
                            }
                        )
                        or theta
                    ),
                    name="add_ornstein_params",
                ),
            ],
        },
    }


def get_ddm_mic2_ornstein_angle_no_bias_config():
    """Get configuration for DDM mic2 ornstein angle no bias model."""
    return {
        "name": "ddm_mic2_ornstein_angle_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "d", "g", "t", "theta"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.0, 0.0, 0.0, -0.1],
            [4.0, 4.0, 4.0, 2.5, 1.0, 3.0, 2.0, 1.0],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 8,
        "default_params": [0.0, 0.0, 0.0, 1.0, 0.5, 1.5, 1.0, 0.0],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_mic2_ornstein,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update(
                            {
                                "zh": np.tile(np.array([0.5], dtype=np.float32), n),
                                "zl1": np.tile(np.array([0.5], dtype=np.float32), n),
                                "zl2": np.tile(np.array([0.5], dtype=np.float32), n),
                            }
                        )
                        or theta
                    ),
                    name="add_z_defaults",
                ),
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update(
                            {
                                "s_pre_high_level_choice": np.tile(
                                    np.array([1.0], dtype=np.float32), n
                                )
                            }
                        )
                        or theta
                    ),
                    name="add_ornstein_params",
                ),
            ],
        },
    }


def get_ddm_mic2_ornstein_weibull_no_bias_config():
    """Get configuration for DDM mic2 ornstein weibull no bias model."""
    return {
        "name": "ddm_mic2_ornstein_weibull_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "d", "g", "t", "alpha", "beta"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.0, 0.0, 0.0, 0.31, 0.31],
            [4.0, 4.0, 4.0, 2.5, 1.0, 3.0, 2.0, 4.99, 6.99],
        ],
        "boundary_name": "weibull_cdf",
        "boundary": bf.weibull_cdf,
        "n_params": 9,
        "default_params": [0.0, 0.0, 0.0, 1.0, 0.5, 1.5, 1.0, 2.5, 3.5],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_mic2_ornstein,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update(
                            {
                                "zh": np.tile(np.array([0.5], dtype=np.float32), n),
                                "zl1": np.tile(np.array([0.5], dtype=np.float32), n),
                                "zl2": np.tile(np.array([0.5], dtype=np.float32), n),
                            }
                        )
                        or theta
                    ),
                    name="add_z_defaults",
                ),
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update(
                            {
                                "s_pre_high_level_choice": np.tile(
                                    np.array([1.0], dtype=np.float32), n
                                )
                            }
                        )
                        or theta
                    ),
                    name="add_ornstein_params",
                ),
            ],
        },
    }
