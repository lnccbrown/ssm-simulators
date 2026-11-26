"""MLBA front end: stimulus attributes -> mean drift rates.

Implements Trueblood, Brown & Heathcote (2014), Psychological Review 121,
179-205:

* Appendix C (Eqs. C1-C3) - mapping objective attribute values ``(P, Q)`` to
  subjective values ``(uP, uQ)``. The option is projected along the ray from
  the origin onto the curve ``(x/a)^m + (y/b)^m = 1`` whose intercepts are those
  of the additive indifference line through the option, ``a = b = P + Q``.
  With ``tan(theta) = Q/P`` this gives the closed form

      uP = (P + Q) * P / (P^m + Q^m)^(1/m),
      uQ = (P + Q) * Q / (P^m + Q^m)^(1/m).

  ``m = 1`` is the identity (subjective = objective values); ``m > 1`` favours
  options with less attribute dispersion (extremeness aversion).

* Eq. 3 - pairwise valuation ``V_ij = wP_ij (uP_i - uP_j) + wQ_ij (uQ_i - uQ_j)``.

* Eq. 4 / Eq. 5 - attention weights ``wP_ij = exp(-lambda |uP_i - uP_j|)`` and
  ``wQ_ij = exp(-lambda * beta * |uQ_i - uQ_j|)`` with ``lambda = lambda1`` for a
  non-negative difference and ``lambda = lambda2`` for a negative one. ``beta``
  is the attribute-bias extension the paper uses for the perceptual fits
  (Eq. 5); ``beta = 1`` is Eq. 4.

* Eq. 1 - mean drift rate ``d_i = sum_{j != i} V_ij + I0``.

Everything is vectorised over trials: attribute arrays have shape
``(n_trials, n_options)`` and scalar parameters broadcast as ``(n_trials,)``.

Scale covariance. ``u`` is homogeneous of degree one in ``(P, Q)``, so
multiplying all attributes and ``I0`` by ``c`` and dividing ``lambda1`` and
``lambda2`` by ``c`` multiplies every drift by ``c`` exactly. Combined with the
LBA's invariance under joint scaling of ``(A, b, v, s)`` this lets the paper's
parameterisation (``s = 1``) be expressed in the library's LBA units
(``sigma_noise = 0.1``); see ``ssms.basic_simulators.mlba_units``.
"""

from __future__ import annotations

import numpy as np


def subjective_values(P, Q, m):
    """Map objective attribute values to subjective values (Appendix C).

    Parameters
    ----------
    P, Q : array_like
        Objective attribute values, strictly positive (broadcastable).
    m : array_like
        Curvature exponent, ``m > 0`` (broadcastable against ``P``).

    Returns
    -------
    (uP, uQ) : tuple of ndarray
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    if np.any(P < 0) or np.any(Q < 0) or np.any((P + Q) <= 0):
        raise ValueError("MLBA attribute values must be non-negative and not both 0.")
    if np.any(m <= 0):
        raise ValueError("MLBA curvature m must be strictly positive.")
    # (P^m + Q^m)^(1/m) computed relative to max(P, Q) to avoid overflow /
    # underflow at the large m the paper reports (m ~ 30).
    big = np.maximum(P, Q)
    norm = big * ((P / big) ** m + (Q / big) ** m) ** (1.0 / m)
    scale = (P + Q) / norm
    return scale * P, scale * Q


def mlba_drifts(P, Q, m, lambda1, lambda2, I0, beta=1.0):
    """Mean drift rates of the MLBA (Eqs. 1, 3, 4/5 and Appendix C).

    Parameters
    ----------
    P, Q : array_like, shape (n_trials, n_options) or (n_options,)
        Attribute values of each option.
    m, lambda1, lambda2, I0, beta : scalar or array of shape (n_trials,)

    Returns
    -------
    ndarray, shape (n_trials, n_options)
        Mean drift rate ``d_i`` of each option's accumulator.
    """
    P = np.atleast_2d(np.asarray(P, dtype=np.float64))
    Q = np.atleast_2d(np.asarray(Q, dtype=np.float64))
    if P.shape != Q.shape:
        raise ValueError(f"P and Q shapes differ: {P.shape} vs {Q.shape}.")
    n_trials, _n_opt = P.shape

    def col(x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 1:
            return np.full((n_trials, 1), x.item())
        if x.size != n_trials:
            raise ValueError(f"parameter of size {x.size} for {n_trials} trials")
        return x[:, None]

    m_, l1, l2, i0, be = (col(v) for v in (m, lambda1, lambda2, I0, beta))
    if np.any(l1 < 0) or np.any(l2 < 0):
        raise ValueError("lambda1 and lambda2 must be non-negative.")
    if np.any(be <= 0):
        raise ValueError("beta must be strictly positive.")

    uP, uQ = subjective_values(P, Q, m_)
    # Pairwise differences, target i (axis 1) minus comparison j (axis 2).
    dP = uP[:, :, None] - uP[:, None, :]
    dQ = uQ[:, :, None] - uQ[:, None, :]
    l1b, l2b, beb = l1[:, :, None], l2[:, :, None], be[:, :, None]
    lamP = np.where(dP >= 0, l1b, l2b)
    lamQ = np.where(dQ >= 0, l1b, l2b)
    wP = np.exp(-lamP * np.abs(dP))
    wQ = np.exp(-lamQ * beb * np.abs(dQ))
    V = wP * dP + wQ * dQ  # V_ii = 0 since dP_ii = dQ_ii = 0
    return V.sum(axis=2) + i0
