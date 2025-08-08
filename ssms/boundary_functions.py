"""Define a collection of boundary functions for the simulators in the package."""

# External
from collections.abc import Callable

import numpy as np
from scipy.stats import gamma  # type: ignore

# Collection of boundary functions


# TODO: #82 Revise how the constant function is implemented. It is not clear what it's supposed to do.  # noqa: FIX002
# Constant: (multiplicative)
def constant(t: float | np.ndarray = 0) -> float | np.ndarray:  # noqa: ARG001
    """Constant boundary function.

    Arguments
    ---------
        t (float or np.ndarray, optional): Time point(s). Defaults to 0.

    Returns
    -------
        float or np.ndarray: Constant boundary value(s), same shape as t
    """
    return 1


# Angle (additive)
def angle(t: float | np.ndarray = 1, theta: float = 1) -> np.ndarray:
    """Angle boundary function.

    Arguments
    ---------
        t (float or np.ndarray, optional): Time point(s). Defaults to 1.
        theta (float, optional): Angle in radians. Defaults to 1.

    Returns
    -------
        np.ndarray: Array of boundary values, same shape as t
    """
    return np.multiply(t, (-np.sin(theta) / np.cos(theta)))


# Generalized logistic bound (additive)
def generalized_logistic(
    t: float | np.ndarray = 1,
    B: float = 2.0,  # noqa: N803
    M: float = 3.0,  # noqa: N803
    v: float = 0.5,  # noqa: N803
) -> np.ndarray:
    """Generalized logistic bound.

    Arguments
    ---------
        t (float or np.ndarray, optional): Time point(s). Defaults to 1.
        B (float, optional): Growth rate. Defaults to 2.0.
        M (float, optional): Time of maximum growth. Defaults to 3.0.
        v (float, optional): Affects near which asymptote maximum growth occurs.
        Defaults to 0.5.

    Returns
    -------
        np.ndarray: Array of boundary values, same shape as t
    """
    return 1 - (1 / np.power(1 + np.exp(-B * (t - M)), 1 / v))


# Weibull survival fun (multiplicative)
def weibull_cdf(
    t: float | np.ndarray = 1, alpha: float = 1, beta: float = 1
) -> np.ndarray:
    """Boundary based on weibull survival function.

    Arguments
    ---------
        t (float or np.ndarray, optional): Time point(s). Defaults to 1.
        alpha (float, optional): Shape parameter. Defaults to 1.
        beta (float, optional): Scale parameter. Defaults to 1.

    Returns
    -------
        np.ndarray: Array of boundary values, same shape as t
    """
    return np.exp(-np.power(np.divide(t, beta), alpha))


def conflict_gamma(
    t: float | np.ndarray = np.arange(  # noqa: B008
        0, 20, 0.1
    ),  # TODO: #81 B008 Do not perform function call `np.arange` in argument defaults; instead, perform the call within the function, or read the default from a module-level singleton variable  # noqa: B008, FIX002
    theta: float = 0.5,
    scale: float = 1,
    alpha_gamma: float = 1.01,
    scale_gamma: float = 0.3,
) -> np.ndarray:
    """Conflict bound that allows initial divergence then collapse.

    Arguments
    ---------
        t: (float, np.ndarray)
            Time points (with arbitrary measure, but in HDDM it is used as seconds),
            at which to evaluate the bound. Defaults to np.arange(0, 20, 0.1).
        theta: float
            Collapse angle. Defaults to 0.5.
        scale: float
            Scaling the gamma distribution of the boundary
            (since bound does not have to integrate to one). Defaults to 1.0.
        alpha_gamma: float
            alpha parameter for a gamma in scale shape parameterization. Defaults to
    """
    return (
        scale * gamma.pdf(t, a=alpha_gamma, loc=0, scale=scale_gamma)
        + np.multiply(t, (-np.sin(theta) / np.cos(theta))),
    )


# Type alias for boundary functions
BoundaryFunction = Callable[..., np.ndarray]

# Export all boundary functions
__all__ = [
    "constant",
    "angle",
    "generalized_logistic",
    "weibull_cdf",
    "conflict_gamma",
    "BoundaryFunction",
]
