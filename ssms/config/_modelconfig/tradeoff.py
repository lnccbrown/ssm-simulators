"""Configuration for tradeoff models."""

import cssm
from ssms.basic_simulators import boundary_functions as bf


def get_tradeoff_no_bias_config():
    """Get configuration for tradeoff model without bias."""
    return {
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
    }


def get_tradeoff_angle_no_bias_config():
    """Get configuration for tradeoff model with angle boundary and no bias."""
    return {
        "name": "tradeoff_angle_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "d", "t", "theta"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.0, 0.0, -0.1],
            [4.0, 4.0, 4.0, 2.5, 1.0, 2.0, 1.0],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 7,
        "default_params": [0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_tradeoff,
    }


def get_tradeoff_weibull_no_bias_config():
    """Get configuration for tradeoff model with Weibull boundary and no bias."""
    return {
        "name": "tradeoff_weibull_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "d", "t", "alpha", "beta"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.0, 0.0, 0.31, 0.31],
            [4.0, 4.0, 4.0, 2.5, 1.0, 2.0, 4.99, 6.99],
        ],
        "boundary_name": "weibull_cdf",
        "boundary": bf.weibull_cdf,
        "n_params": 8,
        "default_params": [0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 2.5, 3.5],
        "nchoices": 4,
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_tradeoff,
    }


def get_tradeoff_conflict_gamma_no_bias_config():
    """Get configuration for tradeoff model with conflict gamma boundary and no bias."""
    return {
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
        "n_params": 10,
        "default_params": [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 2, 2],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_tradeoff,
    }
