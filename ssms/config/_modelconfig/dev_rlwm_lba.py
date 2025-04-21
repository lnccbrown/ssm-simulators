"""Configuration for RLWM LBA models."""

import cssm
import ssms.basic_simulators.boundary_functions as bf


def get_dev_rlwm_lba_pw_v1_config():
    """Get configuration for RLWM LBA pairwise v1 model."""
    return {
        "name": "dev_rlwm_lba_pw_v1",
        "params": [
            "v_RL_0",
            "v_RL_1",
            "v_RL_2",
            "v_WM_0",
            "v_WM_1",
            "v_WM_2",
            "a",
            "z",
            "t_WM",
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
    }


def get_dev_rlwm_lba_race_v1_config():
    """Get configuration for RLWM LBA race v1 model."""
    return {
        # RLWM_Race_LBA_3 without ndt; sum of all v_RL = 1 and sum of all v_WM = 1
        "name": "dev_rlwm_lba_race_v1",
        "params": [
            "v_RL_0",
            "v_RL_1",
            "v_RL_2",
            "v_WM_0",
            "v_WM_1",
            "v_WM_2",
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
    }


def get_dev_rlwm_lba_race_v2_config():
    """Get configuration for RLWM LBA race v2 model."""
    return {
        # RLWM_Race_LBA_3 without ndt; no constraints on the sum of v_RL and v_WM.
        "name": "dev_rlwm_lba_race_v2",
        "params": [
            "v_RL_0",
            "v_RL_1",
            "v_RL_2",
            "v_WM_0",
            "v_WM_1",
            "v_WM_2",
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
    }
