"""Leak model configurations."""

import cssm
from ssms.basic_simulators import boundary_functions as bf
from ssms.config.registry import register_model


@register_model("ddm_mic2_leak")
def get_ddm_mic2_leak_config():
    """Get configuration for DDM mic2 leak model."""
    return {
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
    }


@register_model("ddm_mic2_leak_no_bias")
def get_ddm_mic2_leak_no_bias_config():
    """Get configuration for DDM mic2 leak no bias model."""
    return {
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
    }


@register_model("ddm_mic2_leak_conflict_gamma_no_bias")
def get_ddm_mic2_leak_conflict_gamma_no_bias_config():
    """Get configuration for DDM mic2 leak conflict gamma no bias model."""
    return {
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
    }


@register_model("ddm_mic2_leak_angle_no_bias")
def get_ddm_mic2_leak_angle_no_bias_config():
    """Get configuration for DDM mic2 leak angle no bias model."""
    return {
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
    }


@register_model("ddm_mic2_leak_weibull_no_bias")
def get_ddm_mic2_leak_weibull_no_bias_config():
    """Get configuration for DDM mic2 leak weibull no bias model."""
    return {
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
    }
