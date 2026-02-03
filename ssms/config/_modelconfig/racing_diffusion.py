"""Racing Diffusion model configurations."""

import cssm
from ssms.basic_simulators import boundary_functions as bf
from ssms.transforms import (
    ColumnStackParameters,
    ExpandDimension,
)


def get_racing_diffusion_3_config():
    """Get configuration for racing diffusion model with 3 choices."""
    return {
        "name": "racing_diffusion_3",
        "params": ["v0", "v1", "v2", "A", "b", "t"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
            [2.5, 2.5, 2.5, 1.0, 3.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 6,
        "default_params": [1.0, 1.0, 1.0, 0.5, 1.5, 1e-3],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.racing_diffusion_model,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                ColumnStackParameters(["v0", "v1", "v2"], "v", delete_sources=False),
                ExpandDimension(["t", "A", "b"]),
            ],
        },
    }
