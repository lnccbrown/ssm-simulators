"""DDM model configuration."""

import cssm
from .boundary_functions import constant


def _get_base_ddm_config():
    return {
        "name": "ddm",
        "params": ["v", "a", "z", "t"],
        "param_bounds": [[-3.0, 0.3, 0.1, 0.0], [3.0, 2.5, 0.9, 2.0]],
        "boundary_name": "constant",
        "boundary": constant,
        "boundary_params": [],
        "n_params": 4,
        "default_params": [0.0, 1.0, 0.5, 1e-3],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
    }


def get_ddm_config():
    """Get the configuration for the DDM model."""
    base_config = _get_base_ddm_config()
    base_config["simulator"] = cssm.ddm_flexbound
    return base_config


def get_ddm_legacy_config():
    """Get the configuration for the legacy DDM model."""
    base_config = _get_base_ddm_config()
    base_config["simulator"] = cssm.ddm
    return base_config
