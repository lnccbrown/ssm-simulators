"""Configuration for conflict models with dynamical drift."""

import cssm
from ssms.basic_simulators import boundary_functions as bf, drift_functions as df


def get_ds_conflict_drift_config():
    return {
        "name": "ds_conflict_drift",
        "params": [
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
            [0.3, 0.1, 1e-3, 0.0, 0.0, 0.01, 0.01, 0.0, -1.0, -1.0],
            [3.0, 0.9, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "drift_name": "ds_conflict_drift",
        "drift_fun": df.ds_conflict_drift,
        "n_params": 10,
        "default_params": [2.0, 0.5, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 0.5, -0.5],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
    }


def get_ds_conflict_drift_angle_config():
    return {
        "name": "ds_conflict_drift_angle",
        "params": [
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
            [0.3, 0.1, 1e-3, 0.0, 0.0, 0.01, 0.01, 0.0, -1.0, -1.0, 0.0],
            [3.0, 0.9, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.3],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "drift_name": "ds_conflict_drift",
        "drift_fun": df.ds_conflict_drift,
        "n_params": 11,
        "default_params": [2.0, 0.5, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 0.5, -0.5, 0.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
    }


def get_ds_conflict_stimflexons_drift_config():
    return {
        "name": "ds_conflict_stimflexons_drift",
        "params": [
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
            [0.3, 0.1, 1e-3, 0.0, 0.0, 0.01, 0.01, 0.0, -1.0, -1.0, 0.0, 0.0],
            [3.0, 0.9, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "drift_name": "ds_conflict_stimflexons_drift",
        "drift_fun": df.ds_conflict_stimflexons_drift,
        "n_params": 12,
        "default_params": [2.0, 0.5, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 0.5, -0.5, 0.0, 0.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
    }


def get_ds_conflict_stimflexons_drift_angle_config():
    return {
        "name": "ds_conflict_stimflexons_drift_angle",
        "params": [
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
            [0.3, 0.1, 1e-3, 0.0, 0.0, 0.01, 0.01, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0],
            [3.0, 0.9, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.3],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "drift_name": "ds_conflict_stimflexons_drift",
        "drift_fun": df.ds_conflict_stimflexons_drift,
        "n_params": 13,
        "default_params": [
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
    }



def get_ds_conflict_stimflexons_leak_drift_config():
    return {
        "name": "ds_conflict_stimflexons_leak_drift",
        "params": [
            "a",
            "z",
            "g",
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
            [0.3, 0.1, 0.0, 1e-3, 0.0, 0.0, 0.01, 0.01, 0.0, -1.0, -1.0, 0.0, 0.0],
            [3.0, 0.9, 0.9, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "drift_name": "ds_conflict_stimflexons_drift",
        "drift_fun": df.ds_conflict_stimflexons_drift,
        "n_params": 13,
        "default_params": [2.0, 0.5, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 0.5, -0.5, 0.0, 0.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex_leak,
    }


def get_ds_conflict_stimflexons_leak_drift_angle_config():
    return {
        "name": "ds_conflict_stimflexons_leak_drift_angle",
        "params": [
            "a",
            "z",
            "g",
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
            [0.3, 0.1, 0.0, 1e-3, 0.0, 0.0, 0.01, 0.01, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0],
            [3.0, 0.9, 0.9, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.3],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "drift_name": "ds_conflict_stimflexons_drift",
        "drift_fun": df.ds_conflict_stimflexons_drift,
        "n_params": 14,
        "default_params": [
            2.0,
            0.5,
            0.0,
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
        "simulator": cssm.ddm_flex_leak,
    }
