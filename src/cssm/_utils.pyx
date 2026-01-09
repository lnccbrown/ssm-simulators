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

from libc.math cimport (
    log,
    sqrt,
    pow,
    fmax,
    atan,
    sin,
    cos,
    tan,
    M_PI,
    M_PI_2
)

from libc.time cimport time

import numpy as np
cimport numpy as np
import numbers

DTYPE = np.float32

cdef object _global_rng = None

cpdef void set_seed(random_state):
    """
    Set the random seed for the simulation.

    Args:
        random_state: An integer seed or None. If None, the current time is used as seed.

    This function sets a random state globally for the simulation.
    """

    cdef long seed_value
    global _global_rng

    if random_state is None:
        srand(time(NULL))
        _global_rng = np.random.default_rng()
        return

    if isinstance(random_state, numbers.Integral):
        seed_value = <long>random_state
    else:
        try:
            seed_value = <long>int(random_state)
        except (TypeError, ValueError):
            raise ValueError("random_state must be an integer or None")

    srand(seed_value)
    _global_rng = np.random.default_rng(seed_value)

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

# OLD IMPLEMENTATION (C-based random_stable) - Kept for reference
# cpdef float random_stable(float alpha):
#     """
#     Generate a random float from a stable distribution.
#     DEPRECATED: This C-based implementation is ~1.8x slower than the NumPy version.
#
#     Args:
#         alpha (float): The stability parameter of the distribution.
#
#     Returns:
#         float: A random float from a stable distribution.
#     """
#     cdef float eta, u, w, x
#
#     u = M_PI * (random_uniform() - 0.5)
#     w = random_exponential()
#
#     if alpha == 1.0:
#         eta = M_PI_2 # useless but kept to remain faithful to wikipedia entry
#         x = (1.0 / eta) * ((M_PI_2) * tan(u))
#     else:
#         x = (sin(alpha * u) / (pow(cos(u), 1 / alpha))) * pow(cos(u - (alpha * u)) / w, (1.0 - alpha) / alpha)
#     return x
#
# cpdef float[:] draw_random_stable(int n, float alpha):
#     """
#     Generate an array of random floats from a stable distribution.
#     DEPRECATED: This C-based implementation is ~1.8x slower than the NumPy version.
#
#     Args:
#         n (int): The number of random floats to generate.
#         alpha (float): The stability parameter of the distribution.
#
#     Returns:
#         float[:]: An array of random floats from a stable distribution.
#     """
#     cdef int i
#     cdef float[:] result = np.zeros(n, dtype = DTYPE)
#
#     for i in range(n):
#         result[i] = random_stable(alpha)
#     return result

# OLD IMPLEMENTATION (Marsaglia polar method) - Kept for reference
# cpdef float random_gaussian():
#     """
#     Generate a random float from a standard normal distribution.
#     DEPRECATED: This Marsaglia polar method is ~2.4x slower than NumPy's Ziggurat.
#
#     Returns:
#         float: A random float from a standard normal distribution.
#     """
#     cdef float x1, x2, w
#     w = 2.0
#
#     while(w >= 1.0):
#         x1 = 2.0 * random_uniform() - 1.0
#         x2 = 2.0 * random_uniform() - 1.0
#         w = x1 * x1 + x2 * x2
#
#     w = ((-2.0 * log(w)) / w) ** 0.5
#     return x1 * w

cpdef float[:] draw_gaussian(int n):
    """
    Generate Gaussian samples using NumPy's Ziggurat algorithm.
    ~2.4x faster than the old Marsaglia polar method.

    Args:
        n (int): The number of random floats to generate.

    Returns:
        float[:]: An array of random floats from a standard normal distribution.
    """
    global _global_rng
    if _global_rng is None:
        _global_rng = np.random.default_rng()

    # Generate with NumPy's fast Ziggurat, return as memoryview
    cdef np.ndarray[float, ndim=1] samples = _global_rng.standard_normal(
        n, dtype=np.float32
    )
    return samples

cpdef float[:] draw_random_stable_scipy(int n, float alpha):
    """
    Generate alpha-stable variates using SciPy's optimized implementation.
    This is the fastest option but adds scipy dependency.
    """
    from scipy.stats import levy_stable

    # SciPy uses different parameterization:
    # beta=0 for symmetric, scale and loc for standardization
    cdef np.ndarray[float, ndim=1] samples = levy_stable.rvs(
        alpha=alpha,
        beta=0, # symmetric
        loc=0, # location
        scale=1, # scale
        size=n,
        random_state=_global_rng
    ).astype(np.float32)

    return samples

# cpdef float[:] draw_random_stable_numpy(int n, float alpha):
#     """
#     Generate alpha-stable random variates using NumPy's fast RNG.
#     ~2-3x faster than the Marsaglia-based version.
#
#     Args:
#         n (int): Number of samples to generate
#         alpha (float): Stability parameter (0 < alpha <= 2)
#
#     Returns:
#         float[:]: Array of stable random variates
#     """
#     global _global_rng
#     if _global_rng is None:
#         _global_rng = np.random.default_rng()
#
#     # Generate all random numbers at once (FAST!)
#     cdef np.ndarray[float, ndim=1] u_vals = _global_rng.uniform(
#         -np.pi/2, np.pi/2, n
#     ).astype(np.float32)
#
#     cdef np.ndarray[float, ndim=1] w_vals = _global_rng.exponential(
#         1.0, n
#     ).astype(np.float32)
#
#     # Prepare output
#     cdef np.ndarray[float, ndim=1] result = np.empty(n, dtype=np.float32)
#     cdef float[:] result_view = result
#     cdef float[:] u_view = u_vals
#     cdef float[:] w_view = w_vals
#
#     cdef int i
#     cdef float u, w, x
#
#     # Special case: alpha = 1 (Cauchy)
#     if alpha == 1.0:
#         for i in range(n):
#             result_view[i] = tan(u_view[i])
#         return result
#
#     # General case
#     cdef float alpha_inv = 1.0 / alpha
#     cdef float one_minus_alpha = 1.0 - alpha
#     cdef float scale = one_minus_alpha / alpha
#
#     for i in range(n):
#         u = u_view[i]
#         w = w_view[i]
#         x = (sin(alpha * u) / pow(cos(u), alpha_inv)) * \
#             pow(cos(u - alpha * u) / w, scale)
#         result_view[i] = x
#
#     return result

cpdef float[:] draw_random_stable(int n, float alpha):
    """
    Generate alpha-stable variates using NumPy RNG with vectorized operations.
    ~1.8x faster than the old C-based implementation.

    Args:
        n (int): The number of random floats to generate.
        alpha (float): The stability parameter of the distribution.

    Returns:
        float[:]: An array of random floats from a stable distribution.
    """
    global _global_rng
    if _global_rng is None:
        _global_rng = np.random.default_rng()

    # Generate as float64, then cast to float32
    cdef np.ndarray[float, ndim=1] u_vals = _global_rng.uniform(
        -np.pi/2, np.pi/2, n
    ).astype(np.float32)

    cdef np.ndarray[float, ndim=1] w_vals = _global_rng.exponential(
        1.0, n
    ).astype(np.float32)

    # Declare result array and intermediate values
    cdef np.ndarray[float, ndim=1] result
    cdef float alpha_inv
    cdef float scale

    # Vectorized computation
    if alpha == 1.0:
        result = np.tan(u_vals).astype(np.float32)
        return result
    else:
        alpha_inv = 1.0 / alpha
        scale = (1.0 - alpha) / alpha

        # All operations with explicit float32 cast at end
        result = ((np.sin(alpha * u_vals) / (np.cos(u_vals) ** alpha_inv)) * \
                  ((np.cos(u_vals - alpha * u_vals) / w_vals) ** scale)).astype(np.float32)

        return result

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

# OLD IMPLEMENTATION (Marsaglia polar method helpers) - Kept for reference
# ## @cythonboundscheck(False)
# cdef void assign_random_gaussian_pair(float[:] out, int assign_ix):
#     """
#     Generate a pair of random floats from a standard normal distribution and assign them to an array.
#     DEPRECATED: Part of old Marsaglia-based implementation.
#
#     Args:
#         out (float[:]): The output array to store the generated values.
#         assign_ix (int): The starting index in the output array to assign the values.
#     """
#     cdef float x1, x2, w
#     w = 2.0
#
#     while(w >= 1.0):
#         x1 = (2.0 * random_uniform()) - 1.0
#         x2 = (2.0 * random_uniform()) - 1.0
#         w = (x1 * x1) + (x2 * x2)
#
#     w = ((-2.0 * log(w)) / w) ** 0.5
#     out[assign_ix] = x1 * w
#     out[assign_ix + 1] = x2 * w # this was x2 * 2 ..... :0
#
# # @cythonboundscheck(False)
# cpdef float[:] draw_gaussian(int n):
#     """
#     Generate an array of random floats from a standard normal distribution.
#     DEPRECATED: This Marsaglia-based implementation is ~2.4x slower than NumPy's Ziggurat.
#
#     Args:
#         n (int): The number of random floats to generate.
#
#     Returns:
#         float[:]: An array of random floats from a standard normal distribution.
#     """
#     # Draws standard normal variables - need to have the variance rescaled
#     cdef int i
#     cdef float[:] result = np.zeros(n, dtype=DTYPE)
#     for i in range(n // 2):
#
#         assign_random_gaussian_pair(result, i * 2)
#     if n % 2 == 1:
#         result[n - 1] = random_gaussian()
#     return result


# OLD IMPLEMENTATION (C-based setup_simulation) - Kept for reference
# cpdef dict setup_simulation(
#       int n_samples,
#       int n_trials,
#       float max_t,
#       float delta_t,
#       random_state
# ):
#     """
#     Set up common simulation data structures.
#     DEPRECATED: Uses old C-based draw_gaussian which is ~2.4x slower.
#     """
#     set_seed(random_state)
#
#     cdef int num_draws = int((max_t / delta_t) + 1)
#
#     # Trajectory storage
#     traj = np.zeros((num_draws, 1), dtype=DTYPE)
#     traj[:, :] = -999
#
#     # Output arrays
#     rts = np.zeros((n_samples,
#                     n_trials,
#                     1),
#                     dtype=DTYPE)
#     choices = np.zeros((n_samples,
#                         n_trials,
#                         1),
#                         dtype=np.intc)
#
#     # Time array
#     t_s = np.arange(0,
#                     max_t + delta_t,
#                     delta_t)\
#                     .astype(DTYPE)
#
#     # Gaussian values
#     gaussian_values = draw_gaussian(num_draws)  # OLD VERSION
#     return {
#         'traj': traj,
#         'rts': rts,
#         'choices': choices,
#         't_s': t_s,
#         'gaussian_values': gaussian_values,
#         'num_draws': num_draws,
#         'delta_t_sqrt': sqrt(delta_t)
#     }

cpdef dict setup_simulation(
      int n_samples,
      int n_trials,
      float max_t,
      float delta_t,
      random_state
):
    """
    Set up common simulation data structures using NumPy's fast RNG.
    ~2.4x faster than the old C-based implementation.
    """
    set_seed(random_state)

    cdef int num_draws = int((max_t / delta_t) + 1)

    # Initialize NumPy RNG with same seed
    global _global_rng
    if random_state is not None:
        _global_rng = np.random.default_rng(random_state)
    else:
        _global_rng = np.random.default_rng()

    # Trajectory storage
    traj = np.zeros((num_draws, 1), dtype=DTYPE)
    traj[:, :] = -999

    # Output arrays
    rts = np.zeros((n_samples,
                    n_trials,
                    1),
                    dtype=DTYPE)
    choices = np.zeros((n_samples,
                        n_trials,
                        1),
                        dtype=np.intc)

    # Time array
    t_s = np.arange(0,
                    max_t + delta_t,
                    delta_t)\
                    .astype(DTYPE)

    # Gaussian values (using new fast NumPy-based draw_gaussian)
    gaussian_values = draw_gaussian(num_draws)
    return {
        'traj': traj,
        'rts': rts,
        'choices': choices,
        't_s': t_s,
        'gaussian_values': gaussian_values,
        'num_draws': num_draws,
        'delta_t_sqrt': sqrt(delta_t)
    }

cpdef void compute_boundary(
    boundary,
    t_s,
    boundary_fun,
    dict boundary_params
):
    """Compute boundary values for given time points.

    Boundary functions now receive all parameters via boundary_params,
    including 'a'. No special treatment needed.
    """
    boundary[:] = boundary_fun(t=t_s, **boundary_params).astype(DTYPE)

cpdef float compute_smooth_unif(
    bint smooth_unif,
    float t_particle,
    float deadline_tmp,
    float delta_t
):
    """Compute uniform smoothing adjustment for reaction times."""
    if not smooth_unif:
        return 0.0

    if t_particle == 0.0:
        return random_uniform() * 0.5 * delta_t
    elif t_particle < deadline_tmp:
        return (0.5 - random_uniform()) * delta_t
    else:
        return 0.0

cpdef void enforce_deadline(
    float[:, :, :] rts_view,
    float[:] deadline_view,
    int n,
    int k,
    int idx=0
):
    """Set RT to OMISSION_SENTINEL (-999) if it exceeds deadline or deadline is invalid.

    The value -999 is the sentinel indicating an omission (no response within deadline).
    This corresponds to ssms.OMISSION_SENTINEL in the Python API.
    """
    if (rts_view[n, k, idx] >= deadline_view[k]) or (deadline_view[k] <= 0):
        rts_view[n, k, idx] = -999

cpdef inline float compute_deadline_tmp(
    float max_t,
    float deadline_k,
    float t_k
):
    """Compute effective deadline for simulation loop."""
    return min(max_t, deadline_k - t_k)

cpdef dict build_param_dict_from_2d_array(
    np.ndarray arr,
    str prefix,
    int n_cols
):
    """
    Build dictionary from 2D array columns for race/LBA models.
    Example: v[:, 0], v[:, 1] -> {'v0': v[:, 0], 'v1': v[:, 1]}
    """
    result = {}
    for i in range(n_cols):
        result[f'{prefix}{i}'] = arr[:, i]
    return result

cpdef dict build_minimal_metadata(
    str simulator_name,
    list possible_choices,
    int n_samples,
    int n_trials,
    str boundary_fun_name=None
):
    """Build minimal metadata dictionary - the BASE for all metadata."""
    metadata = {
        'simulator': simulator_name,
        'possible_choices': possible_choices,
        'n_samples': n_samples,
        'n_trials': n_trials,
    }
    if boundary_fun_name is not None:
        metadata['boundary_fun_type'] = boundary_fun_name
    return metadata

cpdef dict build_full_metadata(
    dict minimal_metadata,
    dict params,
    dict sim_config,
    boundary_fun=None,
    np.ndarray boundary=None,
    np.ndarray traj=None,
    dict boundary_params=None,
    dict drift_params=None,
    dict extra_params=None
):
    """
    Build full metadata by AUGMENTING minimal_metadata.
    This ensures consistency - full always includes everything from minimal.
    """
    # Start with a copy of minimal metadata
    metadata = dict(minimal_metadata)

    # Add all the full metadata fields
    metadata.update(params)
    metadata.update(sim_config)

    # Add optional fields
    if boundary_fun is not None:
        metadata['boundary_fun_type'] = boundary_fun.__name__
    if boundary is not None:
        metadata['boundary'] = boundary
    if traj is not None:
        metadata['trajectory'] = traj
    if boundary_params:
        metadata.update(boundary_params)
    if drift_params:
        metadata.update(drift_params)
    if extra_params:
        metadata.update(extra_params)

    return metadata

cpdef dict build_return_dict(
    np.ndarray rts,
    np.ndarray choices,
    dict metadata,
    dict extra_arrays=None
):
    """Build return dictionary with rts, choices, and metadata."""
    result = {'rts': rts, 'choices': choices, 'metadata': metadata}
    if extra_arrays:
        result.update(extra_arrays)
    return result
