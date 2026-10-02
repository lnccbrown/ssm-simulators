"""Analytical kernels for a one-boundary linear first-passage problem.

The functions in this module implement equation (14) of the standard one-boundary first-passage model.
They are the single-stage building blocks for :mod:`notebooks.volterra_accuracy.race`: a race model
has one such process for every response alternative.
"""

from __future__ import annotations

from math import erfc, log, pi, sqrt

import numpy as np


_SQRT_2 = sqrt(2.0)
_SQRT_2PI = sqrt(2.0 * pi)


def _normal_cdf(x):
    x = np.asarray(x, dtype=float)
    # erfc retains useful precision in the negative tail, unlike 1 + erf(x).
    return 0.5 * np.vectorize(erfc, otypes=[float])(-x / _SQRT_2)


def _log_normal_cdf(x):
    """Stable log Phi(x), including the far negative tail."""
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x)
    tail = x < -8.0
    # First correction to Mills' ratio; sufficient for the quadrature tail.
    result[tail] = -0.5 * x[tail] ** 2 - np.log(-x[tail]) - 0.5 * log(2.0 * pi) - 1.0 / x[tail] ** 2
    result[~tail] = np.log(_normal_cdf(x[~tail]))
    return result


def _validate_parameters(sigma: float, a: float, x0, T: float) -> None:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if np.any(np.asarray(x0) >= a):
        raise ValueError("x0 must be strictly below the initial upper boundary a")
    if T <= 0:
        raise ValueError("T must be positive")


def small_f(t, mu: float, sigma: float, a: float, b: float, T: float, x0):
    """Truncated FPT density to ``a + b t`` conditional on ``x0``.

    Values outside ``(0, T]`` are zero.  ``t`` and ``x0`` may be arrays and
    are broadcast using NumPy's standard rules.
    """
    _validate_parameters(sigma, a, x0, T)
    t = np.asarray(t, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    t, x0 = np.broadcast_arrays(t, x0)
    out = np.zeros_like(t, dtype=float)
    valid = (t > 0.0) & (t <= T)
    distance = a - x0[valid]
    relative_drift = mu - b
    tv = t[valid]
    out[valid] = distance / (_SQRT_2PI * sigma * tv**1.5) * np.exp(
        -((distance - relative_drift * tv) ** 2) / (2.0 * sigma**2 * tv)
    )
    return out


def big_F(t, mu: float, sigma: float, a: float, b: float, T: float, x0):
    """FPT CDF, held constant after the stage horizon ``T``."""
    _validate_parameters(sigma, a, x0, T)
    t = np.asarray(t, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    t, x0 = np.broadcast_arrays(t, x0)
    out = np.zeros_like(t, dtype=float)
    valid = t > 0.0
    elapsed = np.minimum(t[valid], T)
    distance = a - x0[valid]
    relative_drift = mu - b
    root_elapsed = np.sqrt(elapsed)
    first_term = _normal_cdf(
        (relative_drift * elapsed - distance) / (sigma * root_elapsed)
    )
    log_second_term = (
        2.0 * relative_drift * distance / sigma**2
        + _log_normal_cdf(
            (-distance - relative_drift * elapsed) / (sigma * root_elapsed)
        )
    )
    value = first_term + np.exp(np.minimum(log_second_term, 0.0))
    # Floating-point round-off can otherwise produce values just outside [0, 1].
    out[valid] = np.clip(value, 0.0, 1.0)
    return out


def q(x, mu: float, sigma: float, a: float, b: float, T: float, x0):
    """Killed transition subdensity at the end of one stage.

    This is the non-passage density at time ``T`` and is zero on or above the
    stage-end boundary ``a + b*T``.
    """
    _validate_parameters(sigma, a, x0, T)
    x = np.asarray(x, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    x, x0 = np.broadcast_arrays(x, x0)
    boundary = a + b * T
    out = np.zeros_like(x, dtype=float)
    inside = x < boundary
    xi, x0i = x[inside], x0[inside]
    gaussian = np.exp(-((xi - x0i - mu * T) ** 2) / (2.0 * sigma**2 * T))
    killed_factor = -np.expm1(
        2.0 * (a - x0i) * (xi - boundary) / (sigma**2 * T)
    )
    out[inside] = gaussian * killed_factor / (_SQRT_2PI * sigma * sqrt(T))
    return out


__all__ = ["small_f", "big_F", "q"]
