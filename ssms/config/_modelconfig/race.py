"""Race model configurations."""

import numpy as np

import cssm
from ssms.basic_simulators import boundary_functions as bf
from ssms.transforms import (
    ColumnStackParameters,
    ExpandDimension,
    SetZeroArray,
    LambdaAdaptation,
)


def get_race_2_config():
    """Get configuration for race model with 2 choices."""
    return {
        "name": "race_2",
        "params": ["v0", "v1", "a", "z0", "z1", "t"],
        "param_bounds": [
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [2.5, 2.5, 3.0, 0.9, 0.9, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 6,
        "default_params": [0.0, 0.0, 2.0, 0.5, 0.5, 1e-3],
        "nchoices": 2,
        "choices": [0, 1],
        "n_particles": 2,
        "simulator": cssm.race_model,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                ColumnStackParameters(["z0", "z1"], "z", delete_sources=False),
                ColumnStackParameters(["v0", "v1"], "v", delete_sources=False),
                ExpandDimension(["t", "a"]),
            ],
        },
    }


def get_race_no_bias_2_config():
    """Get configuration for race model with 2 choices and no bias."""
    return {
        "name": "race_no_bias_2",
        "params": ["v0", "v1", "a", "z", "t"],
        "param_bounds": [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [2.5, 2.5, 3.0, 0.9, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.0, 0.0, 2.0, 0.5, 1e-3],
        "nchoices": 2,
        "choices": [0, 1],
        "n_particles": 2,
        "simulator": cssm.race_model,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update({"z": np.column_stack([theta["z"], theta["z"]])})
                        or theta
                    ),
                    name="duplicate_z_2",
                ),
                ColumnStackParameters(["v0", "v1"], "v", delete_sources=False),
                ExpandDimension(["t", "a"]),
            ],
        },
    }


def get_race_no_z_2_config():
    """Get configuration for race model with 2 choices and no z."""
    return {
        "name": "race_no_z_2",
        "params": ["v0", "v1", "a", "t"],
        "param_bounds": [
            [0.0, 0.0, 1.0, 0.0],
            [2.5, 2.5, 3.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 4,
        "default_params": [0.0, 0.0, 2.0, 1e-3],
        "nchoices": 2,
        "choices": [0, 1],
        "n_particles": 2,
        "simulator": cssm.race_model,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                SetZeroArray("z", shape=(None, 2)),
                ColumnStackParameters(["v0", "v1"], "v", delete_sources=False),
                ExpandDimension(["t", "a"]),
            ],
        },
    }


def get_race_no_bias_angle_2_config():
    """Get configuration for race model with 2 choices and no bias and angle boundary."""
    return {
        "name": "race_no_bias_angle_2",
        "params": ["v0", "v1", "a", "z", "t", "theta"],
        "param_bounds": [
            [0.0, 0.0, 1.0, 0.0, 0.0, -0.1],
            [2.5, 2.5, 3.0, 0.9, 2.0, 1.45],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 6,
        "default_params": [0.0, 0.0, 2.0, 0.5, 1e-3, 0.0],
        "nchoices": 2,
        "choices": [0, 1],
        "n_particles": 2,
        "simulator": cssm.race_model,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update({"z": np.column_stack([theta["z"], theta["z"]])})
                        or theta
                    ),
                    name="duplicate_z_2",
                ),
                ColumnStackParameters(["v0", "v1"], "v", delete_sources=False),
                ExpandDimension(["t", "a"]),
            ],
        },
    }


def get_race_no_z_angle_2_config():
    """Get configuration for race model with 2 choices and no z and angle boundary."""
    return {
        "name": "race_no_z_angle_2",
        "params": ["v0", "v1", "a", "t", "theta"],
        "param_bounds": [
            [0.0, 0.0, 1.0, 0.0, -0.1],
            [2.5, 2.5, 3.0, 2.0, 1.45],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 5,
        "default_params": [0.0, 0.0, 2.0, 1e-3, 0.0],
        "nchoices": 2,
        "choices": [0, 1],
        "n_particles": 2,
        "simulator": cssm.race_model,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                SetZeroArray("z", shape=(None, 2)),
                ColumnStackParameters(["v0", "v1"], "v", delete_sources=False),
                ExpandDimension(["t", "a"]),
            ],
        },
    }


def get_race_3_config():
    """Get configuration for race model with 3 choices."""
    return {
        "name": "race_3",
        "params": ["v0", "v1", "v2", "a", "z0", "z1", "z2", "t"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [2.5, 2.5, 2.5, 3.0, 0.9, 0.9, 0.9, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 8,
        "default_params": [0.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.5, 1e-3],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.race_model,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                ColumnStackParameters(["z0", "z1", "z2"], "z", delete_sources=False),
                ColumnStackParameters(["v0", "v1", "v2"], "v", delete_sources=False),
                ExpandDimension(["t", "a"]),
            ],
        },
    }


def get_race_no_bias_3_config():
    """Get configuration for race model with 3 choices and no bias."""
    return {
        "name": "race_no_bias_3",
        "params": ["v0", "v1", "v2", "a", "z", "t"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [2.5, 2.5, 2.5, 3.0, 0.9, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 6,
        "n_particles": 3,
        "default_params": [0.0, 0.0, 0.0, 2.0, 0.5, 1e-3],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "simulator": cssm.race_model,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update(
                            {"z": np.column_stack([theta["z"], theta["z"], theta["z"]])}
                        )
                        or theta
                    ),
                    name="duplicate_z_3",
                ),
                ColumnStackParameters(["v0", "v1", "v2"], "v", delete_sources=False),
                ExpandDimension(["t", "a"]),
            ],
        },
    }


def get_race_no_z_3_config():
    """Get configuration for race model with 3 choices and no z."""
    return {
        "name": "race_no_z_3",
        "params": ["v0", "v1", "v2", "a", "t"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [2.5, 2.5, 2.5, 3.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.0, 0.0, 0.0, 2.0, 1e-3],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.race_model,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                SetZeroArray("z", shape=(None, 3)),
                ColumnStackParameters(["v0", "v1", "v2"], "v", delete_sources=False),
                ExpandDimension(["t", "a"]),
            ],
        },
    }


def get_race_no_bias_angle_3_config():
    """Get configuration for race model with 3 choices and no bias and angle boundary."""
    return {
        "name": "race_no_bias_angle_3",
        "params": ["v0", "v1", "v2", "a", "z", "t", "theta"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -0.1],
            [2.5, 2.5, 2.5, 3.0, 0.9, 2.0, 1.45],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 7,
        "default_params": [0.0, 0.0, 0.0, 2.0, 0.5, 1e-3, 0.0],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.race_model,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update(
                            {"z": np.column_stack([theta["z"], theta["z"], theta["z"]])}
                        )
                        or theta
                    ),
                    name="duplicate_z_3",
                ),
                ColumnStackParameters(["v0", "v1", "v2"], "v", delete_sources=False),
                ExpandDimension(["t", "a"]),
            ],
        },
    }


def get_race_no_z_angle_3_config():
    """Get configuration for race model with 3 choices and no z and angle boundary."""
    return {
        "name": "race_no_z_angle_3",
        "params": ["v0", "v1", "v2", "a", "t", "theta"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, 0.0, -0.1],
            [2.5, 2.5, 2.5, 3.0, 2.0, 1.45],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 6,
        "default_params": [0.0, 0.0, 0.0, 2.0, 1e-3, 0.0],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.race_model,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                SetZeroArray("z", shape=(None, 3)),
                ColumnStackParameters(["v0", "v1", "v2"], "v", delete_sources=False),
                ExpandDimension(["t", "a"]),
            ],
        },
    }


def get_race_4_config():
    """Get configuration for race model with 4 choices."""
    return {
        "name": "race_4",
        "params": ["v0", "v1", "v2", "v3", "a", "z0", "z1", "z2", "z3", "t"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.5, 2.5, 2.5, 2.5, 3.0, 0.9, 0.9, 0.9, 0.9, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 10,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.5, 0.5, 1e-3],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 4,
        "simulator": cssm.race_model,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                ColumnStackParameters(
                    ["z0", "z1", "z2", "z3"], "z", delete_sources=False
                ),
                ColumnStackParameters(
                    ["v0", "v1", "v2", "v3"], "v", delete_sources=False
                ),
                ExpandDimension(["t", "a"]),
            ],
        },
    }


def get_race_no_bias_4_config():
    """Get configuration for race model with 4 choices and no bias."""
    return {
        "name": "race_no_bias_4",
        "params": ["v0", "v1", "v2", "v3", "a", "z", "t"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [2.5, 2.5, 2.5, 2.5, 3.0, 0.9, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 7,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 1e-3],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 4,
        "simulator": cssm.race_model,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update({"z": np.column_stack([theta["z"]] * 4)}) or theta
                    ),
                    name="duplicate_z_4",
                ),
                ColumnStackParameters(
                    ["v0", "v1", "v2", "v3"], "v", delete_sources=False
                ),
                ExpandDimension(["t", "a"]),
            ],
        },
    }


def get_race_no_z_4_config():
    """Get configuration for race model with 4 choices and no z."""
    return {
        "name": "race_no_z_4",
        "params": ["v0", "v1", "v2", "v3", "a", "t"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [2.5, 2.5, 2.5, 2.5, 3.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 6,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 1e-3],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 4,
        "simulator": cssm.race_model,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                SetZeroArray("z", shape=(None, 4)),
                ColumnStackParameters(
                    ["v0", "v1", "v2", "v3"], "v", delete_sources=False
                ),
                ExpandDimension(["t", "a"]),
            ],
        },
    }


def get_race_no_bias_angle_4_config():
    """Get configuration for race model with 4 choices and no bias and angle boundary."""
    return {
        "name": "race_no_bias_angle_4",
        "params": ["v0", "v1", "v2", "v3", "a", "z", "t", "theta"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -0.1],
            [2.5, 2.5, 2.5, 2.5, 3.0, 0.9, 2.0, 1.45],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 8,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 1e-3, 0.0],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 4,
        "simulator": cssm.race_model,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update({"z": np.column_stack([theta["z"]] * 4)}) or theta
                    ),
                    name="duplicate_z_4",
                ),
                ColumnStackParameters(
                    ["v0", "v1", "v2", "v3"], "v", delete_sources=False
                ),
                ExpandDimension(["t", "a"]),
            ],
        },
    }


def get_race_no_z_angle_4_config():
    """Get configuration for race model with 4 choices and no z and angle boundary."""
    return {
        "name": "race_no_z_angle_4",
        "params": ["v0", "v1", "v2", "v3", "a", "t", "theta"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1],
            [2.5, 2.5, 2.5, 2.5, 3.0, 2.0, 1.45],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 7,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 1e-3, 0.0],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 4,
        "simulator": cssm.race_model,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                SetZeroArray("z", shape=(None, 4)),
                ColumnStackParameters(
                    ["v0", "v1", "v2", "v3"], "v", delete_sources=False
                ),
                ExpandDimension(["t", "a"]),
            ],
        },
    }
