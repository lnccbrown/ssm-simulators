"""Test-support reference for the MLBA (Trueblood, Brown & Heathcote, 2014).

Two things live here, both used only by ``tests/mlba``:

* closed-form LBA first-passage quantities, used only to check the simulator;
* the paper's qualitative-analysis parameters and choice sets, plus the
  declared model specification the registered ``mlba2`` / ``mlba3`` configs
  are checked against (``MLBA_SPEC``).

The closed form is a *reference*, not part of the registered model: it gives an
analytic value for the choice probabilities that the Monte-Carlo simulator in
``cssm`` estimates, so that simulation error can be separated from
implementation error.

Back end, per accumulator (Brown & Heathcote, 2008): start point
``z ~ U(0, A)``, drift ``v`` drawn once per trial, threshold ``b``, hitting
time ``(b - z) / v``.

Two drift conventions are supported, because they differ:

``folded=False``
    ``v ~ N(d, s)``; accumulators with ``v <= 0`` never finish.  This is the
    textbook LBA and what the MLBA paper describes ("drift rates are drawn
    from normal distributions", Table 2 lists ``s``).  Choice probabilities
    then sum to ``1 - P(all drifts <= 0)``.
``folded=True``
    ``v ~ |N(d, s)|``.  This is what ``cssm.lba_vanilla`` does
    (``np.abs(np.random.normal(...))``), so it is the convention the registered
    ``mlba2`` / ``mlba3`` simulators actually follow.  Choice probabilities sum
    to one.  The two agree to ``O(P(N(d, s) < 0))``, which is ~3e-7 at the
    paper's operating point (``d ~ 5``, ``s = 1``).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ssms.basic_simulators.simulator import OMISSION_SENTINEL, simulator

__all__ = [
    "lba_cdf",
    "lba_pdf",
    "choice_probabilities",
    "PAPER_PARAMS",
    "CHOICE_SETS",
    "CONDITIONS",
    "MLBA_SPEC",
    "paper_theta",
    "simulate_paper",
]


def _gh(t: np.ndarray, d: float, A: float, b: float, s: float):
    ts = t * s
    g = (b - A - t * d) / ts
    h = (b - t * d) / ts
    return g, h


def _cdf_one(t: np.ndarray, d: float, A: float, b: float, s: float) -> np.ndarray:
    """P(hitting time <= t) for a single accumulator with v ~ N(d, s)."""
    g, h = _gh(t, d, A, b, s)
    ts = t * s
    return (
        1.0
        + (
            (b - A - t * d) * norm.cdf(g)
            - (b - t * d) * norm.cdf(h)
            + ts * norm.pdf(g)
            - ts * norm.pdf(h)
        )
        / A
    )


def _pdf_one(t: np.ndarray, d: float, A: float, b: float, s: float) -> np.ndarray:
    """Density of the hitting time for a single accumulator with v ~ N(d, s)."""
    g, h = _gh(t, d, A, b, s)
    return (-d * norm.cdf(g) + s * norm.pdf(g) + d * norm.cdf(h) - s * norm.pdf(h)) / A


def lba_cdf(
    t: np.ndarray, d: float, A: float, b: float, s: float, folded: bool = True
) -> np.ndarray:
    """Hitting-time CDF of one accumulator (see module docstring)."""
    out = _cdf_one(np.asarray(t, dtype=np.float64), d, A, b, s)
    if folded:
        out = out + _cdf_one(np.asarray(t, dtype=np.float64), -d, A, b, s)
    return out


def lba_pdf(
    t: np.ndarray, d: float, A: float, b: float, s: float, folded: bool = True
) -> np.ndarray:
    """Hitting-time density of one accumulator (see module docstring)."""
    out = _pdf_one(np.asarray(t, dtype=np.float64), d, A, b, s)
    if folded:
        out = out + _pdf_one(np.asarray(t, dtype=np.float64), -d, A, b, s)
    return out


def choice_probabilities(
    drifts: np.ndarray,
    A: float,
    b: float,
    s: float,
    folded: bool = True,
    t_max: float = 400.0,
    n_grid: int = 400_001,
) -> np.ndarray:
    """Marginal choice probabilities of the LBA race, by quadrature.

    ``P(i) = int_0^inf f_i(t) prod_{j != i} (1 - F_j(t)) dt``, evaluated with
    Simpson's rule on a uniform grid.  ``n_grid`` must be odd.

    Parameters
    ----------
    drifts
        Mean drift rates, shape ``(n_options,)``.
    A, b, s
        Start-point range, threshold, drift standard deviation.
    folded
        Drift convention, see module docstring.
    t_max, n_grid
        Quadrature range and resolution.  The defaults integrate to within
        ~1e-9 of the refined value at the paper's operating point; see
        ``tests/test_numerical_accuracy.py``.

    Returns
    -------
    np.ndarray
        Probability of each accumulator winning, shape ``(n_options,)``.
    """
    drifts = np.asarray(drifts, dtype=np.float64).ravel()
    if n_grid % 2 == 0:  # Simpson needs an even number of intervals
        n_grid += 1
    t = np.linspace(0.0, t_max, n_grid)
    # t = 0 is a removable singularity of F and f (both vanish there); evaluate
    # just off zero and patch the endpoint, leaving the grid itself uniform --
    # perturbing t[0] would change Simpson's step and bias the whole integral.
    probe = t.copy()
    probe[0] = t[1] * 1e-6

    F = np.stack([lba_cdf(probe, d, A, b, s, folded) for d in drifts])
    f = np.stack([lba_pdf(probe, d, A, b, s, folded) for d in drifts])
    F[:, 0] = 0.0
    f[:, 0] = 0.0
    surv = np.clip(1.0 - F, 0.0, 1.0)

    out = np.empty(drifts.size)
    for i in range(drifts.size):
        integrand = f[i] * np.prod(np.delete(surv, i, axis=0), axis=0)
        out[i] = _simpson(integrand, t)
    return out


def _simpson(y: np.ndarray, x: np.ndarray) -> float:
    h = x[1] - x[0]
    return float(
        h / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-2:2].sum())
    )


# ---------------------------------------------------------------------------
# The paper's qualitative-analysis parameterisation
# ---------------------------------------------------------------------------

# Parameter values the paper fixes for its qualitative analyses (p. 188 for
# Figure 4, p. 191 for Figure 7).  ``chi`` is the paper's threshold, registered
# as ``b``.  ``beta = 1`` (no attribute bias) is the default for the inference
# stimuli; it is only freed when fitting the perceptual experiments.
PAPER_PARAMS: dict = {
    "m": 5.0,
    "lambda1": 0.2,
    "lambda2": 0.4,
    "I0": 5.0,
    "beta": 1.0,
    "A": 1.0,
    "chi": 2.0,
    "s": 1.0,
}

# X is the focal option, Y the non-focal option, D the decoy.
_X = (4.0, 6.0)
_Y = (6.0, 4.0)

CHOICE_SETS: dict[str, tuple[tuple[float, float], ...]] = {
    # Attraction decoys, Table 3 set 1 (decoy near X).
    "attraction_range_set1": (_X, _Y, (3.4, 6.0)),
    "attraction_range_frequency_set1": (_X, _Y, (3.7, 5.7)),
    "attraction_frequency_set1": (_X, _Y, (4.0, 5.4)),
    # Similarity decoy, Table 3 set 1 (decoy near X).
    "similarity_set1": (_X, _Y, (3.8, 6.2)),
    # Compromise, Table 3 set 2: the decoy is extreme beyond X, which makes X
    # the compromise option between the decoy and Y.
    "compromise_set2": (_X, _Y, (2.0, 8.0)),
    # The binary set itself.
    "binary_xy": (_X, _Y),
}

CONDITIONS: list = list(CHOICE_SETS)


def paper_theta(paper_params: dict, options: tuple[tuple[float, float], ...]) -> dict:
    """Map the paper's parameter names onto the registered model's names."""
    theta = {
        "A": float(paper_params["A"]),
        "b": float(paper_params["chi"]),  # the paper's threshold chi
        "s": float(paper_params["s"]),
        "t": float(paper_params.get("t", 0.0)),  # no non-decision time in the paper
        "m": float(paper_params["m"]),
        "lambda1": float(paper_params["lambda1"]),
        "lambda2": float(paper_params["lambda2"]),
        "beta": float(paper_params.get("beta", 1.0)),
        "I0": float(paper_params["I0"]),
    }
    for i, (p, q) in enumerate(options):
        theta[f"P{i}"] = float(p)
        theta[f"Q{i}"] = float(q)
    return theta


def simulate_paper(
    paper_params: dict, condition: str | None, n: int, seed: int
) -> dict:
    """Simulate ``n`` trials of one paper choice set through ``mlba2``/``mlba3``.

    Returns ``{"rt": float array, "choice": int array}`` with choice ``1`` =
    option X, ``2`` = option Y, ``3`` = the decoy; censored RTs become nan.
    ``None`` selects the first condition.
    """
    if condition is None:
        condition = CONDITIONS[0]
    if condition not in CHOICE_SETS:
        raise ValueError(
            f"unknown condition {condition!r}; expected one of {CONDITIONS}"
        )

    options = CHOICE_SETS[condition]
    out = simulator(
        theta=paper_theta(paper_params, options),
        model=f"mlba{len(options)}",
        n_samples=int(n),
        random_state=int(seed),
    )
    rt = np.asarray(out["rts"], dtype=np.float64).ravel()
    choice = np.asarray(out["choices"]).ravel().astype(int) + 1
    rt = np.where(rt == OMISSION_SENTINEL, np.nan, rt)
    return {"rt": rt, "choice": choice}


# ---------------------------------------------------------------------------
# Declared specification of the registered models
# ---------------------------------------------------------------------------

# Parameters, bounds, defaults, transform pipelines and reductions declared for
# ``mlba2`` / ``mlba3``, the conditions above, and the paper's reported
# relative choice shares (p. 192) with their tolerances.
MLBA_SPEC: dict = {
    "models": [
        {
            "name": "mlba3",
            "base_model": "lba3",
            "parameters": [
                {"name": "A", "bounds": [0.0, 2.0], "default": 1.0},
                {"name": "b", "bounds": [0.1, 4.0], "default": 2.0},
                {"name": "s", "bounds": [0.01, 2.0], "default": 1.0},
                {"name": "t", "bounds": [0.0, 2.0], "default": 0.0},
                {"name": "m", "bounds": [0.1, 40.0], "default": 5.0},
                {"name": "lambda1", "bounds": [0.0, 1.0], "default": 0.2},
                {"name": "lambda2", "bounds": [0.0, 1.0], "default": 0.4},
                {"name": "beta", "bounds": [0.1, 5.0], "default": 1.0},
                {"name": "I0", "bounds": [0.0, 20.0], "default": 5.0},
                {"name": "P0", "bounds": [0.1, 10.0], "default": 4.0},
                {"name": "Q0", "bounds": [0.1, 10.0], "default": 6.0},
                {"name": "P1", "bounds": [0.1, 10.0], "default": 6.0},
                {"name": "Q1", "bounds": [0.1, 10.0], "default": 4.0},
                {"name": "P2", "bounds": [0.1, 10.0], "default": 3.4},
                {"name": "Q2", "bounds": [0.1, 10.0], "default": 6.0},
            ],
            "parameter_transforms": {
                "sampling": [
                    {
                        "type": "ssms.transforms.sampling.swap.SwapIfLessConstraint",
                        "param_a": "b",
                        "param_b": "A",
                    }
                ],
                "simulation": [
                    {
                        "type": "ssms.transforms.simulation.mlba.MLBAFrontEnd",
                        "n_options": 3,
                    },
                    {
                        "type": "ssms.transforms.simulation.common.ColumnStackParameters",
                        "source_params": ["v0", "v1", "v2"],
                        "target_param": "v",
                        "delete_sources": True,
                    },
                    {
                        "type": "ssms.transforms.simulation.common.RenameParameter",
                        "old_name": "A",
                        "new_name": "z",
                        "transform_fn": None,
                    },
                    {
                        "type": "ssms.transforms.simulation.common.RenameParameter",
                        "old_name": "b",
                        "new_name": "a",
                        "transform_fn": None,
                    },
                    {
                        "type": "ssms.transforms.simulation.common.ExpandDimension",
                        "param_names": ["z", "a"],
                    },
                ],
            },
            "reductions": [
                {
                    "to": "lba3",
                    "settings": {
                        "A": 0.3,
                        "b": 0.5,
                        "s": 0.1,
                        "t": 0.0,
                        "m": 1.0,
                        "lambda1": 0.0,
                        "lambda2": 0.0,
                        "beta": 1.0,
                        "I0": 0.3333333333333333,
                        "P0": 0.5,
                        "Q0": 0.5,
                        "P1": 0.5,
                        "Q1": 0.5833333333333333,
                        "P2": 0.5,
                        "Q2": 0.5,
                    },
                    "map": {"A": "A", "b": "b"},
                    "expect": "distribution",
                }
            ],
        },
        {
            "name": "mlba2",
            "base_model": "lba2",
            "parameters": [
                {"name": "A", "bounds": [0.0, 2.0], "default": 1.0},
                {"name": "b", "bounds": [0.1, 4.0], "default": 2.0},
                {"name": "s", "bounds": [0.01, 2.0], "default": 1.0},
                {"name": "t", "bounds": [0.0, 2.0], "default": 0.0},
                {"name": "m", "bounds": [0.1, 40.0], "default": 5.0},
                {"name": "lambda1", "bounds": [0.0, 1.0], "default": 0.2},
                {"name": "lambda2", "bounds": [0.0, 1.0], "default": 0.4},
                {"name": "beta", "bounds": [0.1, 5.0], "default": 1.0},
                {"name": "I0", "bounds": [0.0, 20.0], "default": 5.0},
                {"name": "P0", "bounds": [0.1, 10.0], "default": 7.5},
                {"name": "Q0", "bounds": [0.1, 10.0], "default": 3.5},
                {"name": "P1", "bounds": [0.1, 10.0], "default": 6.5},
                {"name": "Q1", "bounds": [0.1, 10.0], "default": 4.5},
            ],
            "parameter_transforms": {
                "sampling": [
                    {
                        "type": "ssms.transforms.sampling.swap.SwapIfLessConstraint",
                        "param_a": "b",
                        "param_b": "A",
                    }
                ],
                "simulation": [
                    {
                        "type": "ssms.transforms.simulation.mlba.MLBAFrontEnd",
                        "n_options": 2,
                    },
                    {
                        "type": "ssms.transforms.simulation.common.ColumnStackParameters",
                        "source_params": ["v0", "v1"],
                        "target_param": "v",
                        "delete_sources": True,
                    },
                    {
                        "type": "ssms.transforms.simulation.common.RenameParameter",
                        "old_name": "A",
                        "new_name": "z",
                        "transform_fn": None,
                    },
                    {
                        "type": "ssms.transforms.simulation.common.RenameParameter",
                        "old_name": "b",
                        "new_name": "a",
                        "transform_fn": None,
                    },
                    {
                        "type": "ssms.transforms.simulation.common.ExpandDimension",
                        "param_names": ["z", "a"],
                    },
                ],
            },
            "reductions": [
                {
                    "to": "lba2",
                    "settings": {
                        "A": 0.3,
                        "b": 0.5,
                        "s": 0.1,
                        "t": 0.0,
                        "m": 1.0,
                        "lambda1": 0.0,
                        "lambda2": 0.0,
                        "beta": 1.0,
                        "I0": 0.5,
                        "P0": 1.0,
                        "Q0": 1.0,
                        "P1": 1.0,
                        "Q1": 1.0,
                    },
                    "map": {"A": "A", "b": "b"},
                    "expect": "identical",
                }
            ],
        },
    ],
    "conditions": [
        {
            "id": "attraction_range_set1",
            "levels": {
                "X_P": 4.0,
                "X_Q": 6.0,
                "Y_P": 6.0,
                "Y_Q": 4.0,
                "decoy_P": 3.4,
                "decoy_Q": 6.0,
            },
        },
        {
            "id": "attraction_range_frequency_set1",
            "levels": {
                "X_P": 4.0,
                "X_Q": 6.0,
                "Y_P": 6.0,
                "Y_Q": 4.0,
                "decoy_P": 3.7,
                "decoy_Q": 5.7,
            },
        },
        {
            "id": "attraction_frequency_set1",
            "levels": {
                "X_P": 4.0,
                "X_Q": 6.0,
                "Y_P": 6.0,
                "Y_Q": 4.0,
                "decoy_P": 4.0,
                "decoy_Q": 5.4,
            },
        },
        {
            "id": "similarity_set1",
            "levels": {
                "X_P": 4.0,
                "X_Q": 6.0,
                "Y_P": 6.0,
                "Y_Q": 4.0,
                "decoy_P": 3.8,
                "decoy_Q": 6.2,
            },
        },
        {
            "id": "compromise_set2",
            "levels": {
                "X_P": 4.0,
                "X_Q": 6.0,
                "Y_P": 6.0,
                "Y_Q": 4.0,
                "decoy_P": 2.0,
                "decoy_Q": 8.0,
            },
        },
        {"id": "binary_xy", "levels": {"X_P": 4.0, "X_Q": 6.0, "Y_P": 6.0, "Y_Q": 4.0}},
    ],
    "paper_parameters": {
        "m": 5.0,
        "lambda1": 0.2,
        "lambda2": 0.4,
        "I0": 5.0,
        "beta": 1.0,
        "A": 1.0,
        "chi": 2.0,
        "s": 1.0,
    },
    "paper_quantities": [
        {
            "id": "rel_share_attraction_range",
            "paper_value": 0.58,
            "tolerance": 0.01,
            "computed": True,
        },
        {
            "id": "rel_share_attraction_range_frequency",
            "paper_value": 0.52,
            "tolerance": 0.01,
            "computed": True,
        },
        {
            "id": "rel_share_attraction_frequency",
            "paper_value": 0.45,
            "tolerance": 0.01,
            "computed": True,
        },
        {
            "id": "rel_share_similarity",
            "paper_value": 0.41,
            "tolerance": 0.01,
            "computed": True,
        },
        {
            "id": "rel_share_compromise",
            "paper_value": 0.62,
            "tolerance": 0.01,
            "computed": True,
        },
        {
            "id": "binary_symmetric_share",
            "paper_value": 0.5,
            "tolerance": 0.01,
            "computed": True,
        },
    ],
}
