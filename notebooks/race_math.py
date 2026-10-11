"""Analytical one-sided race-model quantities (notebook reference code).

One independent accumulator starts at ``x0``, drifts at rate ``mu`` with
diffusion coefficient ``sigma``, and is absorbed at an upper boundary that
starts at ``a`` and moves linearly: ``boundary(t) = a + b * t``.  In the frame
of the moving boundary the process has drift ``mu - b`` and a fixed boundary
at distance ``a - x0``, so its first-passage time is Wald (inverse-Gaussian)
distributed.  The three functions below give that first-passage density, its
CDF (reflection formula), and the density of the position of a path that has
*not* been absorbed by time ``T`` (method of images), each restricted to the
observation window ``(0, T]``.

They are the exact single-accumulator references the notebooks in this
directory use to check ``cssm.race_multistage``.  They are not part of the
package API; import them with the notebook's working directory set to
``notebooks/``.

Arguments shared by all functions:

mu
    Drift rate of the accumulator.
sigma
    Diffusion coefficient (standard deviation of the increment per unit
    time).  Must be positive.
a
    Boundary position at ``t = 0``.  Must exceed ``x0``.
b
    Boundary slope, so the boundary at time ``t`` is ``a + b * t``.
T
    Length of the observation window.  Must be positive.
x0
    Starting position of the accumulator.
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
    """Return the unnormalised Gaussian density killed at ``boundary``.

    ``boundary`` is the boundary position at time ``T``.  The second factor
    is the method-of-images correction that removes paths which touched the
    boundary before ``T``.
    """
    distance_to_boundary = a - x0
    terminal_mean = x0 + mu * T
    variance = sigma**2 * T
    gaussian = np.exp(-((x - terminal_mean) ** 2) / (2.0 * variance))
    killed_factor = 1.0 - np.exp(2.0 * distance_to_boundary * (x - boundary) / variance)
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
    """First-passage-time density ``f(t)`` of the accumulator.

    ``t`` may be a scalar or an array.  The density is zero outside the
    observation window ``(0, T]``.  See the module docstring for the
    remaining arguments.
    """
    _validate_race_parameters(sigma, T, a, x0)
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    valid = (t > 0.0) & (t <= T)
    distance = a - x0
    relative_drift = mu - b
    t_valid = t[valid]
    normaliser = distance / (_SQRT_2PI * sigma * t_valid**1.5)
    exponent = -((distance - relative_drift * t_valid) ** 2) / (
        2.0 * sigma**2 * t_valid
    )
    out[valid] = normaliser * np.exp(exponent)
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
    """First-passage-time CDF ``F(t)``: probability of absorption by ``t``.

    ``t`` may be a scalar or an array.  The CDF is zero for ``t <= 0`` and
    constant for ``t >= T`` (absorption after the window is not observed).
    See the module docstring for the remaining arguments.
    """
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
    out[valid] = _normal_cdf(passage_z_score) + reflection_factor * _normal_cdf(
        survival_z_score
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
    """Density ``q(x)`` of the position at time ``T`` of a surviving path.

    ``x`` may be a scalar or an array of positions.  The density is zero at
    or above the boundary ``a + b * T`` and integrates to ``1 - F(T)``, the
    probability that the accumulator has not been absorbed by ``T``.  See the
    module docstring for the remaining arguments.
    """
    _validate_race_parameters(sigma, T, a, x0)
    x = np.asarray(x, dtype=float)
    boundary = a + b * T
    out = np.zeros_like(x)
    inside = x < boundary
    x_inside = x[inside]
    out[inside] = _nonpassage_density(x_inside, mu, sigma, boundary, T, a, x0)
    out[inside] /= _SQRT_2PI * sigma * sqrt(T)
    return out


__all__ = ["small_f", "big_F", "q"]
