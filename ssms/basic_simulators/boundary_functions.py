# External
from scipy.stats import gamma
import numpy as np
from typing import Callable

# Collection of boundary functions

"""
Module defines a collection of boundary functions for the simulators in the package.
"""


# Constant: (multiplicative)
def constant(t: float | np.ndarray = 0) -> float | np.ndarray:
    """constant boundary function

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
    """angle boundary function

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
    t: float | np.ndarray = 1, B: float = 2.0, M: float = 3.0, v: float = 0.5
) -> np.ndarray:
    """generalized logistic bound

    Arguments
    ---------
        t (float or np.ndarray, optional): Time point(s). Defaults to 1.
        B (float, optional): Growth rate. Defaults to 2.0.
        M (float, optional): Time of maximum growth. Defaults to 3.0.
        v (float, optional): Affects near which asymptote maximum growth occurs. Defaults to 0.5.

    Returns
    -------
        np.ndarray: Array of boundary values, same shape as t
    """
    return 1 - (1 / np.power(1 + np.exp(-B * (t - M)), 1 / v))


# Weibull survival fun (multiplicative)
def weibull_cdf(
    t: float | np.ndarray = 1, alpha: float = 1, beta: float = 1
) -> np.ndarray:
    """boundary based on weibull survival function.

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
    t: float | np.ndarray = np.arange(0, 20, 0.1),
    theta: float = 0.5,
    scale: float = 1,
    alpha_gamma: float = 1.01,
    scale_gamma: float = 0.3,
) -> np.ndarray:
    """conflict bound that allows initial divergence then collapse

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


# Define Type alias for boundary functions
BoundaryFunction = Callable[..., np.ndarray]

constant: BoundaryFunction = constant
angle: BoundaryFunction = angle
generalized_logistic: BoundaryFunction = generalized_logistic
weibull_cdf: BoundaryFunction = weibull_cdf
conflict_gamma: BoundaryFunction = conflict_gamma
