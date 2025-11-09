# Globaly settings for cython
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: initializedcheck=False

"""
Shared utility functions for CSSM simulators.

This module contains common helper functions used across all simulator modules,
including random number generation and basic mathematical operations.
"""

import cython
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.math cimport log, sqrt, pow, fmax, atan, sin, cos, tan, M_PI, M_PI_2
from libc.time cimport time

import numpy as np
cimport numpy as np
import numbers

DTYPE = np.float32

cpdef set_seed(random_state):
    """
    Set the random seed for the simulation.

    Args:
        random_state: An integer seed or None. If None, the current time is used as seed.

    This function sets a random state globally for the simulation.
    """
    if random_state is None:
        return srand(time(NULL))
    if isinstance(random_state, numbers.Integral):
        return srand(random_state)

# Method to draw random samples from a gaussian
cpdef float random_uniform():
    """
    Generate a random float from a uniform distribution between 0 and 1.

    Returns:
        float: A random float between 0 and 1.
    """
    cdef float r = rand()
    return r / RAND_MAX

cpdef float random_exponential():
    """
    Generate a random float from an exponential distribution with rate 1.

    Returns:
        float: A random float from an exponential distribution.
    """
    return - log(random_uniform())

cpdef float random_stable(float alpha):
    """
    Generate a random float from a stable distribution.

    Args:
        alpha (float): The stability parameter of the distribution.

    Returns:
        float: A random float from a stable distribution.
    """
    cdef float eta, u, w, x

    u = M_PI * (random_uniform() - 0.5)
    w = random_exponential()

    if alpha == 1.0:
        eta = M_PI_2 # useless but kept to remain faithful to wikipedia entry
        x = (1.0 / eta) * ((M_PI_2) * tan(u))
    else:
        x = (sin(alpha * u) / (pow(cos(u), 1 / alpha))) * pow(cos(u - (alpha * u)) / w, (1.0 - alpha) / alpha)
    return x

cpdef float[:] draw_random_stable(int n, float alpha):
    """
    Generate an array of random floats from a stable distribution.

    Args:
        n (int): The number of random floats to generate.
        alpha (float): The stability parameter of the distribution.

    Returns:
        float[:]: An array of random floats from a stable distribution.
    """
    cdef int i
    cdef float[:] result = np.zeros(n, dtype = DTYPE)

    for i in range(n):
        result[i] = random_stable(alpha)
    return result

cpdef float random_gaussian():
    """
    Generate a random float from a standard normal distribution.

    Returns:
        float: A random float from a standard normal distribution.
    """
    cdef float x1, x2, w
    w = 2.0

    while(w >= 1.0):
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
        w = x1 * x1 + x2 * x2

    w = ((-2.0 * log(w)) / w) ** 0.5
    return x1 * w

cpdef int sign(float x):
    """
    Determine the sign of a float.

    Args:
        x (float): The input float.

    Returns:
        int: 1 if x is positive, -1 if x is negative, 0 if x is zero.
    """
    return (x > 0) - (x < 0)

cpdef float csum(float[:] x):
    """
    Calculate the sum of elements in an array.

    Args:
        x (float[:]): The input array.

    Returns:
        float: The sum of all elements in the array.
    """
    cdef int i
    cdef int n = x.shape[0]
    cdef float total = 0
    
    for i in range(n):
        total += x[i]
    
    return total

## @cythonboundscheck(False)
cdef void assign_random_gaussian_pair(float[:] out, int assign_ix):
    """
    Generate a pair of random floats from a standard normal distribution and assign them to an array.

    Args:
        out (float[:]): The output array to store the generated values.
        assign_ix (int): The starting index in the output array to assign the values.
    """
    cdef float x1, x2, w
    w = 2.0

    while(w >= 1.0):
        x1 = (2.0 * random_uniform()) - 1.0
        x2 = (2.0 * random_uniform()) - 1.0
        w = (x1 * x1) + (x2 * x2)

    w = ((-2.0 * log(w)) / w) ** 0.5
    out[assign_ix] = x1 * w
    out[assign_ix + 1] = x2 * w # this was x2 * 2 ..... :0 

# @cythonboundscheck(False)
cpdef float[:] draw_gaussian(int n):
    """
    Generate an array of random floats from a standard normal distribution.

    Args:
        n (int): The number of random floats to generate.

    Returns:
        float[:]: An array of random floats from a standard normal distribution.
    """
    # Draws standard normal variables - need to have the variance rescaled
    cdef int i
    cdef float[:] result = np.zeros(n, dtype=DTYPE)
    for i in range(n // 2):

        assign_random_gaussian_pair(result, i * 2)
    if n % 2 == 1:
        result[n - 1] = random_gaussian()
    return result

