"""Model configurations ``mlba2`` and ``mlba3`` for the ``ssms`` registry.

Description
-----------
The multiattribute linear ballistic accumulator (MLBA; Trueblood, Brown &
Heathcote, 2014) is an LBA race whose mean drift rates are computed from the
attribute values of the options by a "front end" (Eqs. 1-5, Appendix C):
subjective values on a curve of exponent ``m``, pairwise comparisons weighted
by similarity-dependent attention weights with decay ``lambda1`` (positive
differences) and ``lambda2`` (negative differences), an optional attribute
bias ``beta`` (Eq. 5), and a baseline input ``I0``. The "back end" is the
standard LBA (start ``U(0, A)``, threshold ``b``, rates ``N(d_i, sd)``), in which
a negative rate never reaches threshold.

Two configurations are registered: ``mlba2`` (binary choice sets, as in the
paper's regularity and binary-inference analyses) and ``mlba3`` (ternary choice
sets, the paper's context-effect paradigm). Choice ``i`` is option ``i``.

Units
-----
Like ``lba3``, the drift-rate SD is not a parameter but comes from
``sigma_noise`` (the ``ssms`` default for LBA-type models is 0.1). The model is
scale covariant: (P, Q, I0, A, b, sd) * c with (lambda1, lambda2) / c gives the
same RTs and choices. The paper's choice-only setting ``A = 1, chi = 2, s = 1``
with attributes on a 0-10 scale is therefore the library-unit setting
``A = 0.1, b = 0.2, sd = 0.1`` with attributes on a 0-1 scale and lambdas x 10;
``ssms.basic_simulators.mlba_units.paper_to_library`` does the conversion. The
defaults below are the paper's Fig. 7 setting (m = 5, lambda1 = .2, lambda2 = .4, I0 = 5,
X = (4, 6), Y = (6, 4), range decoy (3.4, 6)) in library units.

Parameters
----------
m : curvature of the subjective-value map, (0.1, 40]; 1 = identity.
lambda1, lambda2 : attention-weight decay for positive / negative differences,
    [0, 10]; 0 = weights of 1 (attention mechanism off).
I0 : baseline input, [0, 2].
beta : attribute bias on the Q weights (Eq. 5), [0.1, 10]; 1 = no bias.
A : start-point range, [0, 1].   b : threshold, [0, 1], b >= A.
t : non-decision time (s), [0, 2] (not part of the paper, which fits choices).
P0, Q0, P1, Q1[, P2, Q2] : attribute values of the options, [0.01, 1].
"""

from __future__ import annotations

from ssms.basic_simulators import boundary_functions as bf
from ssms.basic_simulators.mlba import mlba_race
from ssms.transforms import SwapIfLessConstraint

_CORE = [
    # name, lower, upper, default
    ("m", 0.1, 40.0, 5.0),
    ("lambda1", 0.0, 10.0, 2.0),
    ("lambda2", 0.0, 10.0, 4.0),
    ("I0", 0.0, 2.0, 0.5),
    ("beta", 0.1, 10.0, 1.0),
    ("A", 0.0, 1.0, 0.1),
    ("b", 0.0, 1.0, 0.2),
    ("t", 0.0, 2.0, 0.0),
]
_ATTR_BOUNDS = (0.01, 1.0)
# X = (4, 6), Y = (6, 4), range decoy R_X = (3.4, 6) (Table 3), divided by 10.
_ATTR_DEFAULTS = [(0.4, 0.6), (0.6, 0.4), (0.34, 0.6)]


def _get_mlba_config(n_options: int) -> dict:
    params, lower, upper, defaults = [], [], [], []
    for name, lo, hi, de in _CORE:
        params.append(name)
        lower.append(lo)
        upper.append(hi)
        defaults.append(de)
    for i in range(n_options):
        for k, attr in enumerate(("P", "Q")):
            params.append(f"{attr}{i}")
            lower.append(_ATTR_BOUNDS[0])
            upper.append(_ATTR_BOUNDS[1])
            defaults.append(_ATTR_DEFAULTS[i][k])
    config = {
        "name": f"mlba{n_options}",
        "params": params,
        "param_bounds": [lower, upper],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": len(params),
        "default_params": defaults,
        "nchoices": n_options,
        "choices": list(range(n_options)),
        "n_particles": n_options,
        "tags": ["python_simulator"],
        "simulator": mlba_race,
        "parameter_transforms": {
            "sampling": [SwapIfLessConstraint("b", "A")],
            "simulation": [],
        },
    }
    config["param_bounds_dict"] = {
        p: (lo, hi) for p, lo, hi in zip(params, lower, upper)
    }
    return config


def get_mlba2_config() -> dict:
    """Binary-choice MLBA (two options)."""
    return _get_mlba_config(2)


def get_mlba3_config() -> dict:
    """Ternary-choice MLBA (three options; the paper's context-effect sets)."""
    return _get_mlba_config(3)
