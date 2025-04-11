"""Configuration for levy models."""

import cssm
from ssms.basic_simulators import boundary_functions as bf


def get_levy_config():
    """Get configuration for levy model."""
    return {
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
    }


def get_levy_angle_config():
    """Get configuration for levy model with angle boundary."""
    return {
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
    }
