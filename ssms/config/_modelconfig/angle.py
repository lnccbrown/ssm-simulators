"""Angle model configuration."""

import cssm
from ssms.basic_simulators import boundary_functions as bf


def get_angle_extended_config():
    """Get the configuration for the Angle model with an extended drift range.

    Identical to `get_angle_config` except that `v` spans (-6, 6) rather than
    (-3, 3). Same simulator, same boundary, same parameters in the same order --
    the two configs should stay trivially diffable, so change both together.

    The wider box exists because the published `angle` network was trained on
    (-3, 3) and cannot be trusted outside it: widening the declared bounds of
    `angle` itself would let a sampler explore where the network never learned,
    which fails silently rather than loudly. A separate model earns a separate
    network.

    Naming follows `shrink_spot_extended`, the registry's other bounds-only
    variant. Note that one sets its own `"name"` field to `"shrink_spot"`, so
    its key and name disagree; do not copy that here.
    """
    return {
        "name": "angle_extended",
        "params": ["v", "a", "z", "t", "theta"],
        "param_bounds": [[-6.0, 0.3, 0.1, 1e-3, -0.1], [6.0, 3.0, 0.9, 2.0, 1.3]],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 5,
        "default_params": [0.0, 1.0, 0.5, 1e-3, 0.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [],
        },
    }


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
        "parameter_transforms": {
            "sampling": [],
            "simulation": [],
        },
    }
