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

_LBA_START_THRESHOLD_SAMPLING_TRANSFORMS = [SwapIfLessConstraint("b", "A")]

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

# LBA4 simulation transforms
_LBA4_SIMULATION_TRANSFORMS = [
    LambdaAdaptation(
        lambda theta, cfg, n: theta.update({"nact": 4}) or theta,
        name="set_nact_4",
    ),
    ColumnStackParameters(["v0", "v1", "v2", "v3"], "v", delete_sources=False),
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

# LBA 3-choice with angle and *relative* start-point range simulation transforms.
# Shape-identical to the list above: the relative-z reparameterization lives in the
# simulator (cssm.dev_lba_angle_v2), not in the transform pipeline, so that the
# reported parameters stay in the units the model was fit in.
_DEV_LBA_ANGLE_3_V2_SIMULATION_TRANSFORMS = [
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
            "sampling": _LBA_START_THRESHOLD_SAMPLING_TRANSFORMS,
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
            "sampling": _LBA_START_THRESHOLD_SAMPLING_TRANSFORMS,
            "simulation": _LBA3_SIMULATION_TRANSFORMS,
        },
    }


def get_lba4_config():
    """Get configuration for LBA4 model."""
    return {
        "name": "lba4",
        "params": ["A", "b", "v0", "v1", "v2", "v3"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.1, 0.1, 0.1],
            [1.0, 1.0, 1.0, 1.1, 0.50, 0.50],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 6,
        "default_params": [0.3, 0.5, 0.25, 0.25, 0.25, 0.25],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 4,
        "simulator": cssm.lba_vanilla,
        "parameter_transforms": {
            "sampling": _LBA_START_THRESHOLD_SAMPLING_TRANSFORMS,
            "simulation": _LBA4_SIMULATION_TRANSFORMS,
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


def get_dev_lba_angle_3_v2_config():
    """Get configuration for the 3-choice angle-LBA with a relative start point.

    Description
    -----------
    Angle-LBA over three accumulators, identical to ``lba_angle_3`` except that
    the start-point parameter ``z`` is **relative**: it gives the width of the
    uniform start-point distribution as a fraction of the threshold ``a``, so
    start points are drawn from ``U(0, z * a)`` rather than ``U(0, z)``.

    Making ``z`` relative decouples it from ``a``. Any ``z`` in ``[0, 1]`` keeps
    start points below threshold for every ``a``, which is why this model can use
    a much wider threshold range than ``lba_angle_3`` (``a`` up to 3.0 rather than
    1.1, ``z`` up to 0.9 rather than 0.5) and why it needs no ``a``/``z`` swap
    constraint during parameter sampling.

    Parameters
    ----------
    v0, v1, v2 : float
        Mean drift rate of each accumulator. Valid range: [0.0, 6.0].
        Unconstrained -- unlike ``lba_angle_3_vs_constraint``, the drift rates are
        not required to sum to 1.
    a : float
        Decision threshold. Valid range: [0.1, 3.0].
    z : float
        Start-point range as a fraction of ``a``. Valid range: [0.0, 0.9].
        ``z = 0`` means every accumulator starts at 0; ``z = 0.9`` means start
        points are uniform on 90% of the distance to threshold.
    theta : float
        Angle of the collapsing boundary, in radians. Valid range: [0.0, 1.3].
        Enters the finishing time as ``tan(theta)`` added to the drift rate.

    Model Characteristics
    ---------------------
    - Number of choices: 3
    - Boundary type: constant (the collapse is applied inside the simulator via
      the ``tan(theta)`` term, as for all angle-LBA models)
    - Drift type: constant, drawn per accumulator as ``|N(v, sd)|``
    - Key assumptions: no within-trial noise; the start point is a *fraction* of
      the threshold; no non-decision time (``t`` is forced to 0)

    Notes
    -----
    Ported from the ``origin/dev_lba_angle_3_v2`` development branch of
    ``ssm-simulators`` (commits 97e1e45/10b288c), which predates the
    modularization of ``config.py``. ``param_bounds`` and ``default_params`` are
    reproduced from that branch unchanged: trained likelihood-approximation
    networks for this model exist and were trained inside this box, so widening
    or shifting the bounds would invalidate them.

    Examples
    --------
    >>> from ssms import Simulator
    >>> sim = Simulator("dev_lba_angle_3_v2")
    >>> out = sim.simulate(
    ...     theta={"v0": 0.5, "v1": 0.3, "v2": 0.2, "a": 0.5, "z": 0.2, "theta": 0.0},
    ...     n_samples=1000,
    ... )
    >>> out["rts"].shape
    (1000, 1)

    See Also
    --------
    get_lba_angle_3_config : same model with an absolute start-point range.
    get_lba_angle_3_vs_constraint_config : absolute ``z``, drift rates summing to 1.
    """
    return {
        # angle LBA without constraints on vs, start point relative to threshold
        "name": "dev_lba_angle_3_v2",
        "params": ["v0", "v1", "v2", "a", "z", "theta"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
            [6.0, 6.0, 6.0, 3.0, 0.9, 1.3],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 6,
        "default_params": [0.5, 0.3, 0.2, 0.5, 0.2, 0.0],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.dev_lba_angle_v2,
        "parameter_transforms": {
            # No SwapIfLessConstraint("a", "z") here, unlike lba_angle_3: z is a
            # fraction of a, so it is consistent with any a by construction.
            "sampling": [],
            "simulation": _DEV_LBA_ANGLE_3_V2_SIMULATION_TRANSFORMS,
        },
    }
