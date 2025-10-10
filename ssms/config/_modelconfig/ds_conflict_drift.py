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



def get_conflict_stimflexons_drift_config():
    return {
        "name": "conflict_stimflexons_drift",
        "params": [
            "a",
            "z",
            "t",
            "v_t",
            "v_d",
            "tcoh",
            "dcoh",
            "tonset",
            "donset",
        ],
        "param_bounds": [
            [0.3, 0.1, 1e-3, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0],
            [3.0, 0.9, 2.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "drift_name": "conflict_stimflexons_drift",
        "drift_fun": df.conflict_stimflexons_drift,
        "n_params": 9,
        "default_params": [2.0, 0.5, 1.0, 2.0, 2.0, 0.5, -0.5, 0.0, 0.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
    }

def get_conflict_stimflexons_drift_angle_config():
    return {
        "name": "conflict_stimflexons_drift_angle",
        "params": [
            "a",
            "z",
            "t",
            "v_t",
            "v_d",
            "tcoh",
            "dcoh",
            "tonset",
            "donset",
            "theta"
        ],
        "param_bounds": [
            [0.3, 0.1, 1e-3, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0],
            [3.0, 0.9, 2.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.3],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "drift_name": "conflict_stimflexons_drift",
        "drift_fun": df.conflict_stimflexons_drift,
        "n_params": 10,
        "default_params": [2.0, 0.5, 1.0, 2.0, 2.0, 0.5, -0.5, 0.0, 0.0, 0.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
    }


def get_conflict_stimflex_drift_config():
    return {
        "name": "conflict_stimflex_drift",
        "params": [
            "a",
            "z",
            "t",
            "v_t",
            "v_d",
            "tcoh",
            "dcoh",
            "tonset",
            "donset",
            "toffset",
            "doffset",
        ],
        "param_bounds": [
            [0.3, 0.1, 1e-3, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [3.0, 0.9, 2.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "drift_name": "conflict_stimflex_drift",
        "drift_fun": df.conflict_stimflex_drift,
        "n_params": 11,
        "default_params": [2.0, 0.5, 1.0, 2.0, 2.0, 0.5, -0.5, 0.0, 0.0, 0.2, 0.2],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
    }


def get_conflict_stimflex_drift_angle_config():
    return {
        "name": "conflict_stimflex_drift_angle",
        "params": [
            "a",
            "z",
            "t",
            "v_t",
            "v_d",
            "tcoh",
            "dcoh",
            "tonset",
            "donset",
            "toffset",
            "doffset",
            "theta"
        ],
        "param_bounds": [
            [0.3, 0.1, 1e-3, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0],
            [3.0, 0.9, 2.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.3],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "drift_name": "conflict_stimflex_drift",
        "drift_fun": df.conflict_stimflex_drift,
        "n_params": 12,
        "default_params": [2.0, 0.5, 1.0, 2.0, 2.0, 0.5, -0.5, 0.0, 0.0, 0.2, 0.2, 0.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,
    }


def get_conflict_stimflex_leak2_drift_config():
    return {
        "name": "conflict_stimflex_leak2_drift",
        "params": [
            "a",
            "z",
            "t",
            "v_t",
            "v_d",
            "tcoh",
            "dcoh",
            "tonset",
            "donset",
            "toffset",
            "doffset",
            "g_t",
            "g_d",
        ],
        "param_bounds": [
            [0.3, 0.1, 1e-3, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 0.9, 2.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "drift_name": "conflict_stimflex_drift2",
        "drift_fun": df.conflict_stimflex_drift2,
        "n_params": 13,
        "default_params": [2.0, 0.5, 1.0, 2.0, 2.0, 0.5, -0.5, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex_leak2,
    }

