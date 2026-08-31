"""Analytical one-sided race-model quantities from equation (14).

These functions describe one independent accumulator with a piecewise-linear
upper boundary. They are the mathematical reference for the CSSM forward
simulator and the later multi-stage race likelihood implementation.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
from scipy.special import ndtr


_SQRT_2PI = sqrt(2.0 * np.pi)


def _normal_cdf(x: np.ndarray | float) -> np.ndarray:
    """Standard-normal CDF."""
    return ndtr(np.asarray(x, dtype=float))


def _validate_race_parameters(sigma: float, T: float, a: float, x0: float) -> None:
    """Validate the shared scalar parameters of the one-sided race model."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if T <= 0.0:
        raise ValueError("T must be positive")
    if x0 >= a:
        raise ValueError("x0 must be less than a")


def _nonpassage_density(
    x: np.ndarray,
    mu: float,
    sigma: float,
    boundary: float,
    T: float,
    a: float,
    x0: float,
) -> np.ndarray:
    """Return the Gaussian density corrected for absorption at the boundary."""
    distance_to_boundary = a - x0
    terminal_mean = x0 + mu * T
    variance = sigma**2 * T
    gaussian = np.exp(-((x - terminal_mean) ** 2) / (2.0 * variance))
    killed_factor = 1.0 - np.exp(
        2.0 * distance_to_boundary * (x - boundary) / variance
    )
    return gaussian * killed_factor


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
    _validate_race_parameters(sigma, T, a, x0)
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
    _validate_race_parameters(sigma, T, a, x0)
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    valid = t > 0.0
    elapsed = np.minimum(t[valid], T)
    distance = a - x0
    relative_drift = mu - b
    root_elapsed = np.sqrt(elapsed)
    standard_error = sigma * root_elapsed
    passage_z_score = (relative_drift * elapsed - distance) / standard_error
    survival_z_score = (-distance - relative_drift * elapsed) / standard_error
    reflection_factor = np.exp(2.0 * relative_drift * distance / sigma**2)
    out[valid] = (
        _normal_cdf(passage_z_score)
        + reflection_factor * _normal_cdf(survival_z_score)
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
    _validate_race_parameters(sigma, T, a, x0)
    x = np.asarray(x, dtype=float)
    boundary = a + b * T
    out = np.zeros_like(x)
    inside = x < boundary
    x_inside = x[inside]
    out[inside] = _nonpassage_density(
        x_inside, mu, sigma, boundary, T, a, x0
    )
    out[inside] /= _SQRT_2PI * sigma * sqrt(T)
    return out


__all__ = ["small_f", "big_F", "q"]
