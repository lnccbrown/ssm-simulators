"""Weibull model configuration."""

from cssm import ddm_flexbound
from .boundary_functions import weibull_cdf


def get_weibull_config():
    """Get the configuration for the Weibull model."""
    return {
        "name": "weibull",
        "params": ["v", "a", "z", "t", "alpha", "beta"],
        "param_bounds": [
            [-2.5, 0.3, 0.2, 1e-3, 0.31, 0.31],
            [2.5, 2.5, 0.8, 2.0, 4.99, 6.99],
        ],
        "boundary_name": "weibull_cdf",
        "boundary": weibull_cdf,
        "n_params": 6,
        "default_params": [0.0, 1.0, 0.5, 1e-3, 3.0, 3.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": ddm_flexbound,
    }
