"""Configuration for conflict models with dynamical drift.

All conflict models use drift functions that include 'v' as a parameter
and return the final drift value. For conflict models, v=0 is the standard
as the drift is entirely driven by the dynamical system components.
"""

import cssm
from ssms.basic_simulators import boundary_functions as bf, drift_functions as df


def get_conflict_ds_config():
    return {
        "name": "conflict_ds",
        "params": [
            "v",
            "a",
            "z",
            "t",
            "tinit",
            "dinit",
            "tslope",
            "dslope",
            "tfixedp",
            "tcoh",
            "dcoh",
        ],
        "param_bounds": [
            [-3.0, 0.3, 0.1, 1e-3, 0.0, 0.0, 0.01, 0.01, 0.0, -1.0, -1.0],
            [3.0, 3.0, 0.9, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "drift_name": "conflict_ds_drift",
        "drift_fun": df.conflict_ds_drift,
        "n_params": 11,
        "default_params": [0.0, 2.0, 0.5, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 0.5, -0.5],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [],
        },
    }


def get_conflict_ds_angle_config():
    return {
        "name": "conflict_ds_angle",
        "params": [
            "v",
            "a",
            "z",
            "t",
            "tinit",
            "dinit",
            "tslope",
            "dslope",
            "tfixedp",
            "tcoh",
            "dcoh",
            "theta",
        ],
        "param_bounds": [
            [-3.0, 0.3, 0.1, 1e-3, 0.0, 0.0, 0.01, 0.01, 0.0, -1.0, -1.0, 0.0],
            [3.0, 3.0, 0.9, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.3],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "drift_name": "conflict_ds_drift",
        "drift_fun": df.conflict_ds_drift,
        "n_params": 12,
        "default_params": [0.0, 2.0, 0.5, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 0.5, -0.5, 0.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [],
        },
    }


def get_conflict_dsstimflex_config():
    return {
        "name": "conflict_dsstimflex",
        "params": [
            "v",
            "a",
            "z",
            "t",
            "tinit",
            "dinit",
            "tslope",
            "dslope",
            "tfixedp",
            "tcoh",
            "dcoh",
            "tonset",
            "donset",
        ],
        "param_bounds": [
            [-3.0, 0.3, 0.1, 1e-3, 0.0, 0.0, 0.01, 0.01, 0.0, -1.0, -1.0, 0.0, 0.0],
            [3.0, 3.0, 0.9, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "drift_name": "conflict_dsstimflex_drift",
        "drift_fun": df.conflict_dsstimflex_drift,
        "n_params": 13,
        "default_params": [
            0.0,
            2.0,
            0.5,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            3.0,
            0.5,
            -0.5,
            0.0,
            0.0,
        ],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [],
        },
    }


def get_conflict_dsstimflex_angle_config():
    return {
        "name": "conflict_dsstimflex_angle",
        "params": [
            "v",
            "a",
            "z",
            "t",
            "tinit",
            "dinit",
            "tslope",
            "dslope",
            "tfixedp",
            "tcoh",
            "dcoh",
            "tonset",
            "donset",
            "theta",
        ],
        "param_bounds": [
            [
                -3.0,
                0.3,
                0.1,
                1e-3,
                0.0,
                0.0,
                0.01,
                0.01,
                0.0,
                -1.0,
                -1.0,
                0.0,
                0.0,
                0.0,
            ],
            [3.0, 3.0, 0.9, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.3],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "drift_name": "conflict_dsstimflex_drift",
        "drift_fun": df.conflict_dsstimflex_drift,
        "n_params": 14,
        "default_params": [
            0.0,
            2.0,
            0.5,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            3.0,
            0.5,
            -0.5,
            0.0,
            0.0,
            0.0,
        ],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [],
        },
    }


def get_conflict_stimflex_config():
    return {
        "name": "conflict_stimflex",
        "params": ["v", "a", "z", "t", "vt", "vd", "tcoh", "dcoh", "tonset", "donset"],
        "param_bounds": [
            [-3.0, 0.3, 0.1, 1e-3, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0],
            [3.0, 3.0, 0.9, 2.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "drift_name": "conflict_stimflex_drift",
        "drift_fun": df.conflict_stimflex_drift,
        "n_params": 10,
        "default_params": [0.0, 2.0, 0.5, 1.0, 2.0, 2.0, 0.5, -0.5, 0.0, 0.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [],
        },
    }


def get_conflict_stimflex_angle_config():
    return {
        "name": "conflict_stimflex_angle",
        "params": [
            "v",
            "a",
            "z",
            "t",
            "vt",
            "vd",
            "tcoh",
            "dcoh",
            "tonset",
            "donset",
            "theta",
        ],
        "param_bounds": [
            [-3.0, 0.3, 0.1, 1e-3, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0],
            [3.0, 3.0, 0.9, 2.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.3],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "drift_name": "conflict_stimflex_drift",
        "drift_fun": df.conflict_stimflex_drift,
        "n_params": 11,
        "default_params": [0.0, 2.0, 0.5, 1.0, 2.0, 2.0, 0.5, -0.5, 0.0, 0.0, 0.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [],
        },
    }


def get_conflict_stimflexrel1_config():
    return {
        "name": "conflict_stimflexrel1",
        "params": ["v", "a", "z", "t", "vt", "vd", "tcoh", "dcoh", "tonset", "donset"],
        "param_bounds": [
            [-3.0, 0.3, 0.1, 1e-3, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0],
            [3.0, 3.0, 0.9, 2.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "drift_name": "conflict_stimflexrel1_drift",
        "drift_fun": df.conflict_stimflexrel1_drift,
        "n_params": 10,
        "default_params": [0.0, 2.0, 0.5, 1.0, 2.0, 2.0, 0.5, -0.5, 0.0, 0.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [],
        },
    }


def get_conflict_stimflexrel1_angle_config():
    return {
        "name": "conflict_stimflexrel1_angle",
        "params": [
            "v",
            "a",
            "z",
            "t",
            "vt",
            "vd",
            "tcoh",
            "dcoh",
            "tonset",
            "donset",
            "theta",
        ],
        "param_bounds": [
            [-3.0, 0.3, 0.1, 1e-3, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0],
            [3.0, 3.0, 0.9, 2.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.3],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "drift_name": "conflict_stimflexrel1_drift",
        "drift_fun": df.conflict_stimflexrel1_drift,
        "n_params": 11,
        "default_params": [0.0, 2.0, 0.5, 1.0, 2.0, 2.0, 0.5, -0.5, 0.0, 0.0, 0.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [],
        },
    }


def get_conflict_stimflexrel1_leak_config():
    return {
        "name": "conflict_stimflexrel1_leak",
        "params": [
            "v",
            "a",
            "z",
            "t",
            "vt",
            "vd",
            "tcoh",
            "dcoh",
            "tonset",
            "donset",
            "toffset",
            "doffset",
            "g",
        ],
        "param_bounds": [
            [-3.0, 0.3, 0.1, 1e-3, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 3.0, 0.9, 2.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "drift_name": "conflict_stimflexrel1_drift",
        "drift_fun": df.conflict_stimflexrel1_drift,
        "n_params": 13,
        "default_params": [
            0.0,
            2.0,
            0.5,
            1.0,
            2.0,
            2.0,
            0.5,
            -0.5,
            0.0,
            0.0,
            0.2,
            0.2,
            0.0,
        ],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex_leak,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [],
        },
    }


def get_conflict_stimflexrel1_leak2_config():
    """Configuration for dual-drift conflict model with separate leak parameters.

    Note: This model uses conflict_stimflexrel1_dual_drift which returns a 2D array
    and is handled specially by ddm_flex_leak2. The v parameter is not used by the
    drift function but is required by the simulator.
    """
    return {
        "name": "conflict_stimflexrel1_leak2",
        "params": [
            "v",
            "a",
            "z",
            "t",
            "vt",
            "vd",
            "tcoh",
            "dcoh",
            "tonset",
            "donset",
            "toffset",
            "doffset",
            "gt",
            "gd",
        ],
        "param_bounds": [
            [-3.0, 0.3, 0.1, 1e-3, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 3.0, 0.9, 2.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "drift_name": "conflict_stimflexrel1_dual_drift",
        "drift_fun": df.conflict_stimflexrel1_dual_drift,
        "n_params": 14,
        "default_params": [
            0.0,
            2.0,
            0.5,
            1.0,
            2.0,
            2.0,
            0.5,
            -0.5,
            0.0,
            0.0,
            0.2,
            0.2,
            0.0,
            0.0,
        ],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex_leak2,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [],
        },
    }
