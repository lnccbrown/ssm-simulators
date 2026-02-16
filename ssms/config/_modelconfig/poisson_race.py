"""Poisson race model configurations."""

import cssm
from ssms.basic_simulators import boundary_functions as bf
from ssms.transforms import (
    ColumnStackParameters,
    ExpandDimension,
)


def get_poisson_race_config():
    """Get configuration for 2-choice Poisson race model."""
    return {
        "name": "poisson_race",
        "params": ["r1", "r2", "k1", "k2", "t"],
        "param_bounds": [
            [1e-3, 1e-3, 1.0, 1.0, 0.0],
            [20.0, 20.0, 20.0, 20.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [1.0, 1.0, 2.0, 2.0, 1e-3],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 2,
        "simulator": cssm.poisson_race,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                ColumnStackParameters(["r1", "r2"], "r", delete_sources=False),
                ColumnStackParameters(["k1", "k2"], "k", delete_sources=False),
                ExpandDimension(["t"]),
            ],
        },
    }
