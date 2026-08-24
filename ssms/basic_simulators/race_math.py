"""Analytical one-sided race-model quantities from equation (14).

These functions describe one independent accumulator with a piecewise-linear
upper boundary. They are the mathematical reference for the CSSM forward
simulator and the later multi-stage race likelihood implementation.
"""

from __future__ import annotations

from math import erf, sqrt

import numpy as np


_SQRT_2 = sqrt(2.0)
_SQRT_2PI = sqrt(2.0 * np.pi)


def _normal_cdf(x: np.ndarray | float) -> np.ndarray:
    """Standard-normal CDF without requiring SciPy."""
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf, otypes=[float])(x / _SQRT_2))


def small_f(
    t: np.ndarray | float,
    mu: float,
    sigma: float,
    a: float,
    b: float,
    T: float,
    x0: float,
) -> np.ndarray:
    """One-sided FPT density ``f_tau(t)`` in race-model equation (14)."""
    if sigma <= 0.0 or T <= 0.0:
        raise ValueError("sigma and T must be positive")
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    valid = (t > 0.0) & (t <= T)
    distance = a - x0
    relative_drift = mu - b
    t_valid = t[valid]
    out[valid] = (
        distance
        / (_SQRT_2PI * sigma * t_valid**1.5)
        * np.exp(
            -((distance - relative_drift * t_valid) ** 2) / (2.0 * sigma**2 * t_valid)
        )
    )
    return out


def big_F(
    t: np.ndarray | float,
    mu: float,
    sigma: float,
    a: float,
    b: float,
    T: float,
    x0: float,
) -> np.ndarray:
    """One-sided FPT CDF ``F_tau(t)`` in race-model equation (14)."""
    if sigma <= 0.0 or T <= 0.0:
        raise ValueError("sigma and T must be positive")
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    valid = t > 0.0
    elapsed = np.minimum(t[valid], T)
    distance = a - x0
    relative_drift = mu - b
    root_elapsed = np.sqrt(elapsed)
    out[valid] = _normal_cdf(
        (relative_drift * elapsed - distance) / (sigma * root_elapsed)
    ) + np.exp(2.0 * relative_drift * distance / sigma**2) * _normal_cdf(
        (-distance - relative_drift * elapsed) / (sigma * root_elapsed)
    )
    return out


def q(
    x: np.ndarray | float,
    mu: float,
    sigma: float,
    a: float,
    b: float,
    T: float,
    x0: float,
) -> np.ndarray:
    """Killed/non-passage density ``q(x; ..., T, x0)`` in equation (14)."""
    if sigma <= 0.0 or T <= 0.0:
        raise ValueError("sigma and T must be positive")
    x = np.asarray(x, dtype=float)
    boundary = a + b * T
    out = np.zeros_like(x)
    inside = x < boundary
    x_inside = x[inside]
    gaussian = np.exp(-((x_inside - x0 - mu * T) ** 2) / (2.0 * sigma**2 * T))
    killed_factor = 1.0 - np.exp(
        2.0 * (a - x0) * (x_inside - boundary) / (sigma**2 * T)
    )
    out[inside] = gaussian * killed_factor / (_SQRT_2PI * sigma * sqrt(T))
    return out


__all__ = ["small_f", "big_F", "q"]
