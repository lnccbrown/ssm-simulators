"""The MLBA front end: attribute values -> mean drift rates.

Trueblood, Brown & Heathcote (2014), *Psychological Review*, 121(2), 179-205.

The front end is a deterministic map from the objective attribute values of the
options in a choice set onto the mean drift rates of a linear ballistic
accumulator race (the "back end", which is the LBA of Brown & Heathcote, 2008
and is already in ``ssms`` as ``lba2`` / ``lba3``).

Equations, in the paper's numbering:

* (C1)-(C3) objective -> subjective values, a power curve with exponent ``m``
* (4)/(5)   attention weights, exponentially decaying in subjective distance
            with decay ``lambda1`` for positive and ``lambda2`` for negative
            differences, and an attribute bias ``beta`` on attribute Q
* (3)       pairwise valuation ``V_ij``
* (1a)-(1c) ``d_i = sum_{j != i} V_ij + I0``

All functions are pure NumPy and broadcast over a leading trial axis.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "subjective_values",
    "attention_weights",
    "pairwise_valuations",
    "drift_rates",
]


def subjective_values(
    P: np.ndarray, Q: np.ndarray, m: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    """Map objective attribute values onto the subjective curve of Appendix C.

    Options that are objectively indifferent (``P + Q`` equal) lie on the line
    ``x / a + y / b = 1`` with ``a = b = P + Q``.  The subjective curve through
    the same intercepts is ``(x / a) ** m + (y / b) ** m = 1``; the subjective
    value of an option is where the ray from the origin through ``(P, Q)``
    crosses that curve.

    With ``theta = arctan(Q / P)`` the paper writes the solution as Equations
    C2/C3; because ``tan(arctan(Q / P)) = Q / P`` we use the ratio directly,
    which avoids a round trip through the trigonometric functions.

    Parameters
    ----------
    P, Q
        Objective attribute values, shape ``(..., n_options)``.  Must be
        strictly positive (the mapping is defined in the positive quadrant).
    m
        Curvature exponent, ``m > 0``.  Broadcast against ``(..., 1)``.
        ``m == 1`` is the identity map, ``m > 1`` is concave (extremeness
        aversion), ``0 < m < 1`` is convex.

    Returns
    -------
    (uP, uQ)
        Subjective attribute values, same shape as ``P``.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    if np.any(P <= 0) or np.any(Q <= 0):
        raise ValueError("MLBA attribute values P, Q must be strictly positive")
    if np.any(m <= 0):
        raise ValueError("MLBA curvature parameter m must be strictly positive")

    ratio = Q / P
    scale = P + Q  # = a = b, the common intercept of the indifference curve
    denom = (1.0 + ratio**m) ** (1.0 / m)
    uP = scale / denom
    uQ = scale * ratio / denom
    return uP, uQ


def attention_weights(
    duP: np.ndarray,
    duQ: np.ndarray,
    lambda1: np.ndarray | float,
    lambda2: np.ndarray | float,
    beta: np.ndarray | float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Attention weights of Equations 4/5 for given subjective differences.

    ``lambda1`` governs positive differences (evidence for the target option),
    ``lambda2`` negative ones (evidence against it).  ``beta`` rescales the
    decay on attribute Q only: ``beta > 1`` biases attention toward attribute
    Q, ``beta < 1`` toward attribute P, ``beta == 1`` is unbiased.
    """
    lam_P = np.where(duP >= 0, lambda1, lambda2)
    lam_Q = np.where(duQ >= 0, lambda1, lambda2)
    wP = np.exp(-lam_P * np.abs(duP))
    wQ = np.exp(-lam_Q * np.asarray(beta, dtype=np.float64) * np.abs(duQ))
    return wP, wQ


def pairwise_valuations(
    P: np.ndarray,
    Q: np.ndarray,
    m: np.ndarray | float,
    lambda1: np.ndarray | float,
    lambda2: np.ndarray | float,
    beta: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Matrix of pairwise valuations ``V_ij`` (Equation 3).

    Returns an array of shape ``(..., n_options, n_options)`` whose ``[i, j]``
    entry is the value of option ``i`` evaluated with option ``j`` as the
    reference.  The diagonal is exactly zero.
    """
    uP, uQ = subjective_values(
        P, Q, np.expand_dims(np.asarray(m, dtype=np.float64), -1)
    )
    duP = uP[..., :, None] - uP[..., None, :]
    duQ = uQ[..., :, None] - uQ[..., None, :]
    lam1 = np.expand_dims(np.asarray(lambda1, dtype=np.float64), (-1, -2))
    lam2 = np.expand_dims(np.asarray(lambda2, dtype=np.float64), (-1, -2))
    bet = np.expand_dims(np.asarray(beta, dtype=np.float64), (-1, -2))
    wP, wQ = attention_weights(duP, duQ, lam1, lam2, bet)
    return wP * duP + wQ * duQ


def drift_rates(
    P: np.ndarray,
    Q: np.ndarray,
    m: np.ndarray | float,
    lambda1: np.ndarray | float,
    lambda2: np.ndarray | float,
    I0: np.ndarray | float,
    beta: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Mean drift rates ``d_i`` of Equations 1a-1c.

    Parameters
    ----------
    P, Q
        Objective attribute values, shape ``(n_trials, n_options)``.
    m, lambda1, lambda2, I0, beta
        Front-end parameters, scalars or shape ``(n_trials,)``.

    Returns
    -------
    np.ndarray
        Mean drift rates, shape ``(n_trials, n_options)``.
    """
    V = pairwise_valuations(P, Q, m, lambda1, lambda2, beta)
    I0 = np.expand_dims(np.asarray(I0, dtype=np.float64), -1)
    return V.sum(axis=-1) + I0
