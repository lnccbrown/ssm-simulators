"""LBA (Linear Ballistic Accumulator) model configurations."""

import numpy as np

from ssms.basic_simulators import boundary_functions as bf
import cssm

from ssms.transforms import (
    # Sampling transforms
    SwapIfLessConstraint,
    # Simulation transforms
    ColumnStackParameters,
    ExpandDimension,
    RenameParameter,
    DeleteParameters,
    LambdaAdaptation,
)


# ============================================================================
# Shared simulation transforms for LBA models
# ============================================================================

# Helper for setting t to zeros (LBA models don't use t parameter)
_SET_ZERO_T = LambdaAdaptation(
    lambda theta, cfg, n: theta.update({"t": np.zeros(n).astype(np.float32)}) or theta,
    name="set_zero_t",
)

# LBA2 simulation transforms
_LBA2_SIMULATION_TRANSFORMS = [
    LambdaAdaptation(
        lambda theta, cfg, n: theta.update({"nact": 2}) or theta,
        name="set_nact_2",
    ),
    ColumnStackParameters(["v0", "v1"], "v", delete_sources=False),
    RenameParameter("A", "z", lambda x: np.expand_dims(x, axis=1)),
    RenameParameter("b", "a", lambda x: np.expand_dims(x, axis=1)),
    DeleteParameters(["A", "b"]),
    _SET_ZERO_T,
]

# LBA3 simulation transforms
_LBA3_SIMULATION_TRANSFORMS = [
    LambdaAdaptation(
        lambda theta, cfg, n: theta.update({"nact": 3}) or theta,
        name="set_nact_3",
    ),
    ColumnStackParameters(["v0", "v1", "v2"], "v", delete_sources=False),
    RenameParameter("A", "z", lambda x: np.expand_dims(x, axis=1)),
    RenameParameter("b", "a", lambda x: np.expand_dims(x, axis=1)),
    DeleteParameters(["A", "b"]),
    _SET_ZERO_T,
]

# LBA 3-choice with vs constraint simulation transforms (non-angle)
_LBA_3_VS_CONSTRAINT_SIMULATION_TRANSFORMS = [
    ColumnStackParameters(["v0", "v1", "v2"], "v", delete_sources=False),
    ExpandDimension(["a", "z"]),
    _SET_ZERO_T,
]

# LBA 3-choice with angle simulation transforms
_LBA_ANGLE_3_SIMULATION_TRANSFORMS = [
    ColumnStackParameters(["v0", "v1", "v2"], "v", delete_sources=False),
    ExpandDimension(["a", "z", "theta"]),
    _SET_ZERO_T,
]


# ============================================================================
# Model configuration functions
# ============================================================================


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
        "parameter_transforms": {
            "sampling": [],
            "simulation": _LBA2_SIMULATION_TRANSFORMS,
        },
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
        "parameter_transforms": {
            "sampling": [],
            "simulation": _LBA3_SIMULATION_TRANSFORMS,
        },
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
        "parameter_transforms": {
            "sampling": [],
            "simulation": _LBA_3_VS_CONSTRAINT_SIMULATION_TRANSFORMS,
        },
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
        "parameter_transforms": {
            "sampling": [],
            "simulation": _LBA_ANGLE_3_SIMULATION_TRANSFORMS,
        },
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
        # Unified parameter_transforms - both sampling and simulation in one place
        "parameter_transforms": {
            "sampling": [
                SwapIfLessConstraint("a", "z"),
            ],
            "simulation": _LBA_ANGLE_3_SIMULATION_TRANSFORMS,
        },
    }
