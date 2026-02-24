"""Configuration for RLWM LBA models."""

import cssm
from ssms.basic_simulators import boundary_functions as bf
from ssms.transforms import (
    SwapIfLessConstraint,
    NormalizeToSumConstraint,
    ColumnStackParameters,
    ExpandDimension,
    LambdaAdaptation,
)

import numpy as np


def get_dev_rlwm_lba_pw_v1_config():
    """Get configuration for RLWM LBA pairwise v1 model."""
    return {
        "name": "dev_rlwm_lba_pw_v1",
        "params": [
            "vRL0",
            "vRL1",
            "vRL2",
            "vWM0",
            "vWM1",
            "vWM2",
            "a",
            "z",
            "tWM",
        ],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.01],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 0.5, 0.5],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 9,
        "default_params": [0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.5, 0.2, 0.1],
        "nchoices": 3,
        "n_particles": 3,
        "simulator": cssm.rlwm_lba_pw_v1,
        "parameter_transforms": {
            "sampling": [
                SwapIfLessConstraint("a", "z"),
            ],
            "simulation": [
                ColumnStackParameters(
                    ["vRL0", "vRL1", "vRL2"], "vRL", delete_sources=False
                ),
                ColumnStackParameters(
                    ["vWM0", "vWM1", "vWM2"], "vWM", delete_sources=False
                ),
                ExpandDimension(["a", "z", "tWM"]),
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update({"t": np.zeros(n).astype(np.float32)}) or theta
                    ),
                    name="set_zero_t",
                ),
            ],
        },
    }


def get_dev_rlwm_lba_race_v1_config():
    """Get configuration for RLWM LBA race v1 model."""
    return {
        # RLWM_Race_LBA_3 without ndt; sum of all vRL = 1 and sum of all vWM = 1
        "name": "dev_rlwm_lba_race_v1",
        "params": [
            "vRL0",
            "vRL1",
            "vRL2",
            "vWM0",
            "vWM1",
            "vWM2",
            "a",
            "z",
        ],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 0.5],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 8,
        "default_params": [0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.5, 0.2],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.rlwm_lba_race,
        "parameter_transforms": {
            "sampling": [
                NormalizeToSumConstraint(["vRL0", "vRL1", "vRL2"]),
                NormalizeToSumConstraint(["vWM0", "vWM1", "vWM2"]),
                SwapIfLessConstraint("a", "z"),
            ],
            "simulation": [
                ColumnStackParameters(
                    ["vRL0", "vRL1", "vRL2"], "vRL", delete_sources=False
                ),
                ColumnStackParameters(
                    ["vWM0", "vWM1", "vWM2"], "vWM", delete_sources=False
                ),
                ExpandDimension(["a", "z"]),
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update({"t": np.zeros(n).astype(np.float32)}) or theta
                    ),
                    name="set_zero_t",
                ),
            ],
        },
    }


def get_dev_rlwm_lba_race_v2_config():
    """Get configuration for RLWM LBA race v2 model."""
    return {
        # RLWM_Race_LBA_3 without ndt; no constraints on the sum of vRL and vWM.
        "name": "dev_rlwm_lba_race_v2",
        "params": [
            "vRL0",
            "vRL1",
            "vRL2",
            "vWM0",
            "vWM1",
            "vWM2",
            "a",
            "z",
        ],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 8,
        "default_params": [0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.5, 0.2],
        "nchoices": 3,
        "n_particles": 3,
        "simulator": cssm.rlwm_lba_race,
        "parameter_transforms": {
            "sampling": [
                SwapIfLessConstraint("a", "z"),
            ],
            "simulation": [
                ColumnStackParameters(
                    ["vRL0", "vRL1", "vRL2"], "vRL", delete_sources=False
                ),
                ColumnStackParameters(
                    ["vWM0", "vWM1", "vWM2"], "vWM", delete_sources=False
                ),
                ExpandDimension(["a", "z"]),
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update({"t": np.zeros(n).astype(np.float32)}) or theta
                    ),
                    name="set_zero_t",
                ),
            ],
        },
    }
