"""Angle model configuration."""

import cssm
from ssms.basic_simulators import boundary_functions as bf
from ssms.config.registry import register_model

@register_model("angle")
def get_angle_config():
    """Get the configuration for the Angle model."""
    return {
        "name": "angle",
        "params": ["v", "a", "z", "t", "theta"],
        "param_bounds": [[-3.0, 0.3, 0.1, 1e-3, -0.1], [3.0, 3.0, 0.9, 2.0, 1.3]],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 5,
        "default_params": [0.0, 1.0, 0.5, 1e-3, 0.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound,
    }
