"""Configuration for DDM seq2 models."""

import cssm
from ssms.basic_simulators import boundary_functions as bf
from ssms.config.registry import register_model


@register_model("ddm_seq2")
def get_ddm_seq2_config():
    """Get the configuration for the DDM seq2 model."""
    return {
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
    }


@register_model("ddm_seq2_no_bias")
def get_ddm_seq2_no_bias_config():
    """Get the configuration for the DDM seq2 no bias model."""
    return {
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
    }


@register_model("ddm_seq2_conflict_gamma_no_bias")
def get_ddm_seq2_conflict_gamma_no_bias_config():
    """Get the configuration for the DDM seq2 conflict gamma no bias model."""
    return {
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
    }


@register_model("ddm_seq2_angle_no_bias")
def get_ddm_seq2_angle_no_bias_config():
    """Get the configuration for the DDM seq2 angle no bias model."""
    return {
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
    }


@register_model("ddm_seq2_weibull_no_bias")
def get_ddm_seq2_weibull_no_bias_config():
    """Get the configuration for the DDM seq2 weibull no bias model."""
    return {
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
    }
