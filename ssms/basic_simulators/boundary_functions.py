"""Define a collection of boundary functions for the simulators in the package."""

# External
from collections.abc import Callable

import numpy as np
from scipy.stats import gamma  # type: ignore

# Collection of boundary functions


# Constant boundary
def constant(t: float | np.ndarray = 0, a: float = 1.0) -> float | np.ndarray:  # noqa: ARG001
    """Constant boundary function.

    Arguments
    ---------
        t (float or np.ndarray, optional): Time point(s). Defaults to 0.
        a (float, optional): Threshold parameter. Defaults to 1.0.

    Returns
    -------
        float or np.ndarray: Constant boundary value = a (scalar if t is scalar, array if t is array)
    """
    # If t is an array, return array of constant values; if scalar, return scalar
    if isinstance(t, np.ndarray):
        return np.full_like(t, a)
    return a


# Angle boundary with linear collapse
def angle(
    t: float | np.ndarray = 1, a: float = 1.0, theta: float = 1.0
) -> np.ndarray | float:
    """Linear collapsing boundary at angle theta.

    Arguments
    ---------
        t (float or np.ndarray, optional): Time point(s). Defaults to 1.
        a (float, optional): Threshold parameter (starting height). Defaults to 1.0.
        theta (float, optional): Collapse angle in radians. Defaults to 1.0.

    Returns
    -------
        np.ndarray or float: Boundary value = a + t * tan(theta)
    """
    return a + np.multiply(t, (-np.sin(theta) / np.cos(theta)))


# Generalized logistic boundary
def generalized_logistic(
    t: float | np.ndarray = 1,
    a: float = 1.0,
    B: float = 2.0,  # noqa: N803
    M: float = 3.0,  # noqa: N803
    v: float = 0.5,  # noqa: N803
) -> np.ndarray | float:
    """Generalized logistic boundary function.

    Arguments
    ---------
        t (float or np.ndarray, optional): Time point(s). Defaults to 1.
        a (float, optional): Threshold parameter. Defaults to 1.0.
        B (float, optional): Growth rate. Defaults to 2.0.
        M (float, optional): Time of maximum growth. Defaults to 3.0.
        v (float, optional): Affects near which asymptote maximum growth occurs.
        Defaults to 0.5.

    Returns
    -------
        np.ndarray or float: Boundary value = a + logistic_decay(t)
    """
    offset = 1 - (1 / np.power(1 + np.exp(-B * (t - M)), 1 / v))
    return a + offset


# Weibull decay boundary
def weibull_cdf(
    t: float | np.ndarray = 1,
    a: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> np.ndarray | float:
    """Weibull decay boundary function.

    Arguments
    ---------
        t (float or np.ndarray, optional): Time point(s). Defaults to 1.
        a (float, optional): Threshold parameter (starting height). Defaults to 1.0.
        alpha (float, optional): Shape parameter. Defaults to 1.0.
        beta (float, optional): Scale parameter. Defaults to 1.0.

    Returns
    -------
        np.ndarray or float: Boundary value = a * exp(-(t/β)^α)
    """
    return a * np.exp(-np.power(np.divide(t, beta), alpha))


def conflict_gamma(
    t: float | np.ndarray = np.arange(  # noqa: B008
        0, 20, 0.1
    ),  # TODO: #81 B008 Do not perform function call `np.arange` in argument defaults; instead, perform the call within the function, or read the default from a module-level singleton variable  # noqa: B008, FIX002
    a: float = 1.0,
    theta: float = 0.5,
    scale: float = 1.0,
    alphaGamma: float = 1.01,
    scaleGamma: float = 0.3,
) -> np.ndarray:
    """Conflict boundary with gamma bump and linear collapse.

    Arguments
    ---------
        t: (float, np.ndarray)
            Time points (with arbitrary measure, but in HDDM it is used as seconds),
            at which to evaluate the bound. Defaults to np.arange(0, 20, 0.1).
        a: float
            Threshold parameter (starting height). Defaults to 1.0.
        theta: float
            Collapse angle. Defaults to 0.5.
        scale: float
            Scaling the gamma distribution of the boundary
            (since bound does not have to integrate to one). Defaults to 1.0.
        alphaGamma: float
            alpha parameter for a gamma in scale shape parameterization. Defaults to 1.01.
        scaleGamma: float
            scale parameter for a gamma in scale shape parameterization. Defaults to 0.3.

    Returns
    -------
        np.ndarray: Boundary value = a + gamma_bump(t) + t*tan(theta)
    """
    gamma_bump = scale * gamma.pdf(t, a=alphaGamma, loc=0, scale=scaleGamma)
    angle_offset = np.multiply(t, (-np.sin(theta) / np.cos(theta)))
    return a + gamma_bump + angle_offset


# Define Type alias for boundary functions
BoundaryFunction = Callable[..., float | np.ndarray]

constant: BoundaryFunction = constant  # noqa: PLW0127
angle: BoundaryFunction = angle  # noqa: PLW0127
generalized_logistic: BoundaryFunction = generalized_logistic  # noqa: PLW0127
weibull_cdf: BoundaryFunction = weibull_cdf  # noqa: PLW0127
conflict_gamma: BoundaryFunction = conflict_gamma  # noqa: PLW0127
