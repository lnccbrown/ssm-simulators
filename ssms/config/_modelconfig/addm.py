"""ADDM model configuration."""

import cssm
from ssms.basic_simulators import boundary_functions as bf


def get_addm_config(): 
    """Get the configuration for the ADDM model."""
    return {
        "name": "addm",
        # Canonical aDDM contract, matching HSSM's sampled columns
        # [eta, kappa, a, b, x0, t]. x0 is the ABSOLUTE start (was the mislabeled
        # 'z'); sigma rides the noise slot. Bounds mirror hssm.aDDMConfig.
        "params": ["eta", "kappa", "a", "b", "x0", "t"],
        "param_bounds": [[0.0, 0.0, 0.1, 0.0, -1.0, 0.0], [1.0, 5.0, 3.0, 3.0, 1.0, 2.0]],
        "boundary_name": "addm_collapse",
        "boundary": bf.addm_collapse,
        "boundary_params": [],
        "n_params": 6,
        "default_params": [0.5, 0.5, 2.0, 0.2, 0.0, 0.1],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.addm,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [],
        },
    }
