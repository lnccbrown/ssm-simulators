"""DDM model configuration."""

import cssm
from .base import boundary_config


def get_ddm_config():
    """Get the configuration for the DDM model."""
    return {
        "name": "ddm",
        "params": ["v", "a", "z", "t"],
        "param_bounds": [[-3.0, 0.3, 0.1, 0.0], [3.0, 2.5, 0.9, 2.0]],
        "boundary_name": "constant",
        "boundary": boundary_config["constant"]["fun"],
        "boundary_params": [],
        "n_params": 4,
        "default_params": [0.0, 1.0, 0.5, 1e-3],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound,
    }
