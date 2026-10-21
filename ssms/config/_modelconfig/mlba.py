"""Model configurations for the multiattribute linear ballistic accumulator.

Mirrors ``ssms/config/_modelconfig/lba.py``: a ``get_*_config()`` factory per
registered model, a shared list of simulation transforms, and the same
``SwapIfLessConstraint("b", "A")`` sampling constraint the LBA models use to
keep the threshold above the start-point range.

Both factories are listed in the ``configs`` dict of
``ssms/config/_modelconfig/__init__.py``, which registers them statically.
"""

from __future__ import annotations

from ssms.basic_simulators import boundary_functions as bf
from ssms.basic_simulators.lba_seeded import lba_vanilla_seeded
from ssms.transforms import (
    ColumnStackParameters,
    ExpandDimension,
    MLBAFrontEnd,
    RenameParameter,
    SwapIfLessConstraint,
)

# Back-end parameters shared by every MLBA config, in registration order.
# A: start-point range (the paper's A), b: threshold (the paper's chi),
# s: drift standard deviation across trials, t: non-decision time.
_BACK_END = ["A", "b", "s", "t"]
_BACK_END_BOUNDS = ([0.0, 0.1, 0.01, 0.0], [2.0, 4.0, 2.0, 2.0])
_BACK_END_DEFAULTS = [1.0, 2.0, 1.0, 0.0]

# Front-end parameters (Table 2 of the paper, plus beta from Equation 5).
_FRONT_END = ["m", "lambda1", "lambda2", "beta", "I0"]
_FRONT_END_BOUNDS = ([0.1, 0.0, 0.0, 0.1, 0.0], [40.0, 1.0, 1.0, 5.0, 20.0])
_FRONT_END_DEFAULTS = [5.0, 0.2, 0.4, 1.0, 5.0]

# Attribute values are experimental covariates, registered as parameters the
# way the conflict models register tcoh/dcoh.  Bounds cover the paper's
# artificial stimuli (Table 3, attribute space 1-9) and its empirical stimuli
# (eyewitness strengths 0-100 divided by 10, as in the paper's fits).
_ATTRIBUTE_BOUNDS = (0.1, 10.0)


def _attribute_names(n_options: int) -> list[str]:
    return [f"{a}{i}" for i in range(n_options) for a in ("P", "Q")]


def _simulation_transforms(n_options: int) -> list:
    """Front end, then the same plumbing ``lba2``/``lba3`` use."""
    return [
        MLBAFrontEnd(n_options=n_options),
        ColumnStackParameters(
            [f"v{i}" for i in range(n_options)], "v", delete_sources=True
        ),
        RenameParameter("A", "z"),
        RenameParameter("b", "a"),
        ExpandDimension(["z", "a"]),
    ]


def _mlba_config(name: str, n_options: int, attribute_defaults: list[float]) -> dict:
    attributes = _attribute_names(n_options)
    params = _BACK_END + _FRONT_END + attributes
    lower = (
        list(_BACK_END_BOUNDS[0])
        + list(_FRONT_END_BOUNDS[0])
        + [_ATTRIBUTE_BOUNDS[0]] * len(attributes)
    )
    upper = (
        list(_BACK_END_BOUNDS[1])
        + list(_FRONT_END_BOUNDS[1])
        + [_ATTRIBUTE_BOUNDS[1]] * len(attributes)
    )
    return {
        "name": name,
        "params": params,
        "param_bounds": [lower, upper],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": len(params),
        "default_params": _BACK_END_DEFAULTS + _FRONT_END_DEFAULTS + attribute_defaults,
        "nchoices": n_options,
        "choices": list(range(n_options)),
        "n_particles": n_options,
        "simulator": lba_vanilla_seeded,
        "parameter_transforms": {
            "sampling": [SwapIfLessConstraint("b", "A")],
            "simulation": _simulation_transforms(n_options),
        },
    }


def get_mlba2_config() -> dict:
    """MLBA for binary choice sets (the paper's binary inference experiment).

    Default attribute values are choice set 1 of Table 8, an extreme option
    (75, 35) against a mediocre one (65, 45), on the divided-by-ten scale the
    paper uses when fitting.
    """
    return _mlba_config("mlba2", 2, [7.5, 3.5, 6.5, 4.5])


def get_mlba3_config() -> dict:
    """MLBA for ternary choice sets: the paper's main model.

    Option 0 is the focal/target option X, option 1 the non-focal option Y,
    option 2 the decoy.  Default attribute values are the attraction (range
    decoy) set 1 of Table 3: X = (4, 6), Y = (6, 4), decoy = (3.4, 6).
    """
    return _mlba_config("mlba3", 3, [4.0, 6.0, 6.0, 4.0, 3.4, 6.0])
