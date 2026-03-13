"""ADDM model configuration."""

import cssm
from ssms.basic_simulators import boundary_functions as bf


def get_addm_constant_config(): 
    """Get the configuration for the ADDM model."""
    return { 
        "name": "addm",
        "params": ["eta", "kappa", "a", "z", "t"], 
        "param_bounds": [[0.0, 0.0, -2.5, -2.5, 0.01], [3.0, 3.0, 2.5, 2.5, 1.0]],
        "boundary_name": "constant", 
        "boundary": bf.constant,
        "boundary_params": [],
        "n_params": 5,
        "default_params": [0.5, 0.5, 2.0, 0.5, 0.1],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.addm_constant,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [],
        },
    }
