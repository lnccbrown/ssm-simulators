"""Gamma drift model configuration."""

import cssm
from ssms.basic_simulators import boundary_functions as bf, drift_functions as df


def get_gamma_drift_config():
    """Get the configuration for the Gamma drift model."""
    return {
        "name": "gamma_drift",
        "params": ["v", "a", "z", "t", "shape", "scale", "c"],
        "param_bounds": [
            [-3.0, 0.3, 0.1, 1e-3, 2.0, 0.01, -3.0],
            [3.0, 3.0, 0.9, 2.0, 10.0, 1.0, 3.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "drift_name": "gamma_drift",
        "drift_fun": df.gamma_drift,
        "n_params": 7,
        "default_params": [0.0, 1.0, 0.5, 0.25, 5.0, 0.5, 1.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
    }


def get_gamma_drift_angle_config():
    """Get the configuration for the Gamma drift angle model."""
    return {
        "name": "gamma_drift_angle",
        "params": ["v", "a", "z", "t", "theta", "shape", "scale", "c"],
        "param_bounds": [
            [-3.0, 0.3, 0.1, 1e-3, -0.1, 2.0, 0.01, -3.0],
            [3.0, 3.0, 0.9, 2.0, 1.3, 10.0, 1.0, 3.0],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "drift_name": "gamma_drift",
        "drift_fun": df.gamma_drift,
        "n_params": 7,
        "default_params": [0.0, 1.0, 0.5, 0.25, 0.0, 5.0, 0.5, 1.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
    }
