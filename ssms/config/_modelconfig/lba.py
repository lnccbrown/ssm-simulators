"""LBA (Linear Ballistic Accumulator) model configurations."""

from ssms.basic_simulators import boundary_functions as bf
import cssm


def get_lba2_config():
    """Get configuration for LBA2 model."""
    return {
        "name": "lba2",
        "params": ["A", "b", "v0", "v1"],
        "param_bounds": [[0.0, 0.0, 0.0, 0.1], [1.0, 1.0, 1.0, 1.1]],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 4,
        "default_params": [0.3, 0.5, 0.5, 0.5],
        "nchoices": 2,
        "choices": [0, 1],
        "n_particles": 2,
        "simulator": cssm.lba_vanilla,
    }


def get_lba3_config():
    """Get configuration for LBA3 model."""
    return {
        "name": "lba3",
        "params": ["A", "b", "v0", "v1", "v2"],
        "param_bounds": [[0.0, 0.0, 0.0, 0.1, 0.1], [1.0, 1.0, 1.0, 1.1, 0.50]],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.3, 0.5, 0.25, 0.5, 0.25],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.lba_vanilla,
    }


def get_lba_3_vs_constraint_config():
    """Get configuration for LBA3 with vs constraint model."""
    return {
        # conventional analytical LBA with constraints on vs (sum of all v = 1)
        "name": "lba_3_vs_constraint",
        "params": ["v0", "v1", "v2", "a", "z"],
        "param_bounds": [[0.0, 0.0, 0.0, 0.1, 0.1], [1.0, 1.0, 1.0, 1.1, 0.50]],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.5, 0.3, 0.2, 0.5, 0.2],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.lba_vanilla,
    }


def get_lba_angle_3_vs_constraint_config():
    """Get configuration for LBA angle 3 vs constraint model."""
    return {
        # conventional analytical LBA with angle with constraints on vs (sum of all v=1)
        "name": "lba_angle_3_vs_constraint",
        "params": ["v0", "v1", "v2", "a", "z", "theta"],
        "param_bounds": [[0.0, 0.0, 0.0, 0.1, 0.0, 0], [1.0, 1.0, 1.0, 1.1, 0.5, 1.3]],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 6,
        "default_params": [0.5, 0.3, 0.2, 0.5, 0.2, 0.0],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.lba_angle,
    }


def get_lba_angle_3_config():
    """Get configuration for LBA angle 3 model without vs constraints."""
    return {
        # conventional analytical LBA with angle without any constraints on vs
        "name": "lba_angle_3",
        "params": ["v0", "v1", "v2", "a", "z", "theta"],
        "param_bounds": [[0.0, 0.0, 0.0, 0.1, 0.0, 0], [6.0, 6.0, 6.0, 1.1, 0.5, 1.3]],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 6,
        "default_params": [0.5, 0.3, 0.2, 0.5, 0.2, 0.0],
        "nchoices": 3,
        "n_particles": 3,
        "simulator": cssm.lba_angle,
    }
