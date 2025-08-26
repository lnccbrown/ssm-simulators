"""LCA (Leaky Competing Accumulator) model configurations."""

import cssm
from ssms.basic_simulators import boundary_functions as bf


def get_lca_3_config():
    """Get configuration for LCA3 model."""
    return {
        "name": "lca_3",
        "params": ["v0", "v1", "v2", "a", "z0", "z1", "z2", "g", "b", "t"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0],
            [2.5, 2.5, 2.5, 3.0, 0.9, 0.9, 0.9, 1.0, 1.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 10,
        "default_params": [0.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.5, 0.0, 0.0, 1e-3],
        "nchoices": 3,
        "n_particles": 3,
        "simulator": cssm.lca,
    }


def get_lca_no_z_3_config():
    """Get configuration for LCA3 model without bias."""
    return {
        "name": "lca_no_z_3",
        "params": ["v0", "v1", "v2", "a", "g", "b", "t"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0],
            [2.5, 2.5, 2.5, 3.0, 1.0, 1.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 7,
        "default_params": [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1e-3],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.lca,
    }


def get_lca_4_config():
    """Get configuration for LCA4 model."""
    return {
        "name": "lca_4",
        "params": [
            "v0",
            "v1",
            "v2",
            "v3",
            "a",
            "z0",
            "z1",
            "z2",
            "z3",
            "g",
            "b",
            "t",
        ],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0],
            [2.5, 2.5, 2.5, 2.5, 3.0, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 12,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 1e-3],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 4,
        "simulator": cssm.lca,
    }


def get_lca_no_z_4_config():
    """Get configuration for LCA4 model without bias."""
    return {
        "name": "lca_no_z_4",
        "params": ["v0", "v1", "v2", "v3", "a", "g", "b", "t"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0],
            [2.5, 2.5, 2.5, 2.5, 3.0, 1.0, 1.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 8,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1e-3],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 4,
        "simulator": cssm.lca,
    }


def get_lca_no_z_angle_4_config():
    """Get configuration for LCA4 model with angle boundary and no bias."""
    return {
        "name": "lca_no_z_angle_4",
        "params": ["v0", "v1", "v2", "v3", "a", "g", "b", "t", "theta"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0, -0.1],
            [2.5, 2.5, 2.5, 2.5, 3.0, 1.0, 1.0, 2.0, 1.45],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 9,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1e-3, 0.0],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 4,
        "simulator": cssm.lca,
    }


def get_lca_no_bias_3_config():
    return {
        "name": "lca_no_bias_3",
        "params": ["v0", "v1", "v2", "a", "z", "g", "b", "t"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, 0.0],
            [2.5, 2.5, 2.5, 3.0, 0.9, 1.0, 1.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 8,
        "default_params": [0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1e-3],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.lca,
    }


def get_lca_no_bias_angle_3_config():
    return {
        "name": "lca_no_bias_angle_3",
        "params": ["v0", "v1", "v2", "a", "z", "g", "b", "t", "theta"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, 0.0, -1.0],
            [2.5, 2.5, 2.5, 3.0, 0.9, 1.0, 1.0, 2.0, 1.45],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 9,
        "default_params": [0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1e-3, 0.0],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.lca,
    }


def get_lca_no_z_angle_3_config():
    return {
        "name": "lca_no_z_angle_3",
        "params": ["v0", "v1", "v2", "a", "g", "b", "t", "theta"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0, -1.0],
            [2.5, 2.5, 2.5, 3.0, 1.0, 1.0, 2.0, 1.45],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 8,
        "default_params": [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1e-3, 0.0],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.lca,
    }


def get_lca_no_bias_4_config():
    return {
        "name": "lca_no_bias_4",
        "params": ["v0", "v1", "v2", "v3", "a", "z", "g", "b", "t"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, 0.0],
            [2.5, 2.5, 2.5, 2.5, 3.0, 0.9, 1.0, 1.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 9,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1e-3],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 4,
        "simulator": cssm.lca,
    }


def get_lca_no_bias_angle_4_config():
    return {
        "name": "lca_no_bias_angle_4",
        "params": ["v0", "v1", "v2", "v3", "a", "z", "g", "b", "t", "theta"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, 0.0, -0.1],
            [2.5, 2.5, 2.5, 2.5, 3.0, 0.9, 1.0, 1.0, 2.0, 1.45],
        ],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "n_params": 10,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1e-3, 0.0],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 4,
        "simulator": cssm.lca,
    }
