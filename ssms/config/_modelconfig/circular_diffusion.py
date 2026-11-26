"""Centered planar diffusion with fixed drift and a circular absorbing boundary."""

from ssms.basic_simulators import boundary_functions as bf
from ssms.basic_simulators.circular_diffusion import (
    circular_diffusion,
    circular_diffusion_observation_metadata,
)


def get_circular_diffusion_config() -> dict:
    """Register one circular response in radians (not a discrete choice)."""
    return {
        "name": "circular_diffusion",
        "params": ["vx", "vy", "a", "t"],
        "param_bounds": [[-3.0, -3.0, 0.1, 0.0], [3.0, 3.0, 3.0, 2.0]],
        "default_params": [0.0, 0.0, 1.0, 0.2],
        "n_params": 4,
        "boundary_name": "constant",
        "boundary": bf.constant,
        "nchoices": 0,
        "choices": [],
        **circular_diffusion_observation_metadata(),
        "n_particles": 1,
        "tags": ["python_simulator"],
        "simulator": circular_diffusion,
        "parameter_transforms": {"sampling": [], "simulation": []},
    }
