"""Angle model configuration."""

import cssm
from ssms.basic_simulators import boundary_functions as bf


def get_angle_extended_config():
    """Get the Angle configuration under its second name, `angle_extended`.

    Up to 0.16.0 this was `angle` with `v` in (-6, 6) instead of (-3, 3).
    `angle` now has that box, so the two are one config. The name stays in the
    registry because HSSM 0.5.0 and the pipeline config that trained the
    network look the model up by it.
    """
    return {**get_angle_config(), "name": "angle_extended"}


def get_angle_config():
    """Get the configuration for the Angle model.

    `v` spans (-6, 6), the box the wider-range angle network
    (`angle_extended.onnx` on `franklab/HSSM`) was trained on. Up to 0.16.0 it
    was (-3, 3), the box of the 2023 network `angle.onnx`.
    """
    return {
        "name": "angle",
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
