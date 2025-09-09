"""Shrink model configurations."""

import cssm
from ssms import boundary_functions as bf
from ssms import drift_functions as df


def get_shrink_spot_config():
    """Get configuration for shrink spot model."""
    return {
        "name": "shrink_spot",
        "params": [
            "a",
            "z",
            "t",
            "ptarget",
            "pouter",
            "pinner",
            "r",
            "sda",
        ],
        "param_bounds": [
            [0.3, 0.1, 1e-3, 2.0, -5.5, -5.5, 1e-2, 1],
            [3.0, 0.9, 2.0, 5.5, 5.5, 5.5, 0.05, 3],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "drift_name": "attend_drift",
        "drift_fun": df.attend_drift,
        "n_params": 8,
        "default_params": [0.7, 0.5, 0.25, 2.0, -2.0, -2.0, 0.01, 1],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
    }


def get_shrink_spot_extended_config():
    """Get configuration for extended shrink spot model."""
    return {
        "name": "shrink_spot",
        "params": [
            "a",
            "z",
            "t",
            "ptarget",
            "pouter",
            "pinner",
            "r",
            "sda",
        ],
        "param_bounds": [
            [0.3, 0.1, 1e-3, 2.0, -5.5, -5.5, 0.01, 1],
            [3.0, 0.9, 2.0, 5.5, 5.5, 5.5, 1.0, 3],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "drift_name": "attend_drift",
        "drift_fun": df.attend_drift,
        "n_params": 8,
        "default_params": [0.7, 0.5, 0.25, 2.0, -2.0, -2.0, 0.01, 1],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
    }


def get_shrink_spot_simple_config():
    """Get configuration for simple shrink spot model."""
    return {
        "name": "shrink_spot_simple",
        "params": [
            "a",
            "z",
            "t",
            "ptarget",
            "pouter",
            "r",
            "sda",
        ],
        "param_bounds": [
            [0.3, 0.1, 1e-3, 2.0, -5.5, 0.01, 1],
            [3.0, 0.9, 2.0, 5.5, 5.5, 0.05, 3],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "drift_name": "attend_drift_simple",
        "drift_fun": df.attend_drift_simple,
        "n_params": 7,
        "default_params": [0.7, 0.5, 0.25, 2.0, -2.0, 0.01, 1],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
    }


def get_shrink_spot_simple_extended_config():
    """Get configuration for extended simple shrink spot model."""
    return {
        "name": "shrink_spot_simple_extended",
        "params": [
            "a",
            "z",
            "t",
            "ptarget",
            "pouter",
            "r",
            "sda",
        ],
        "param_bounds": [
            [0.3, 0.1, 1e-3, 2.0, -5.5, 0.01, 1],
            [3.0, 0.9, 2.0, 5.5, 5.5, 1.0, 3],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "drift_name": "attend_drift_simple",
        "drift_fun": df.attend_drift_simple,
        "n_params": 7,
        "default_params": [0.7, 0.5, 0.25, 2.0, -2.0, 0.01, 1],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
    }
