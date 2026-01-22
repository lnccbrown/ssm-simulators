"""Ornstein models configurations."""

import cssm
from ssms.basic_simulators import boundary_functions as bf


def get_ornstein_config():
    """Get the configuration for the Ornstein model."""
    return {
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
        "parameter_transforms": {
            "sampling": [],
            "simulation": [],
        },
    }


def get_ornstein_angle_config():
    """Get the configuration for the Ornstein angle model."""
    return {
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
        "parameter_transforms": {
            "sampling": [],
            "simulation": [],
        },
    }
