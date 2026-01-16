# cython: language_level=3
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: initializedcheck=False

"""
Cython nogil DDM Simulators with True Multi-threading

This module implements DDM simulators that release the GIL during computation,
enabling true parallel execution across multiple CPU cores using prange.

Key optimizations:
1. Pure C random number generation (no Python calls in hot loops)
2. Thread-local random state using xoroshiro128+
3. prange with nogil for parallel sample generation
4. Minimal memory allocations in tight loops
"""

import cython
from cython.parallel cimport prange, parallel
from libc.math cimport sqrt, log, cos, sin, floor
from libc.stdlib cimport malloc, free
from libc.stdint cimport uint64_t

import numpy as np
cimport numpy as np

# Compile-time constant
DEF M_PI = 3.14159265358979323846

# ============================================================================
# Fast Random Number Generation (Thread-Safe, No GIL)
# Using xoroshiro128+ algorithm - fast and high quality
# ============================================================================

cdef struct RngState:
    uint64_t s0
    uint64_t s1

cdef inline uint64_t rotl(uint64_t x, int k) noexcept nogil:
    """Rotate left helper for xoroshiro128+"""
    return (x << k) | (x >> (64 - k))

cdef inline uint64_t xoroshiro128plus_next(RngState* state) noexcept nogil:
    """Generate next random uint64 using xoroshiro128+"""
    cdef uint64_t s0 = state.s0
    cdef uint64_t s1 = state.s1
    cdef uint64_t result = s0 + s1

    s1 ^= s0
    state.s0 = rotl(s0, 24) ^ s1 ^ (s1 << 16)
    state.s1 = rotl(s1, 37)

    return result

cdef inline void init_rng_state(RngState* state, uint64_t seed, int thread_id) noexcept nogil:
    """Initialize RNG state with seed and thread_id for reproducibility"""
    # Mix seed and thread_id using splitmix64 to get initial state
    cdef uint64_t z = seed + <uint64_t>thread_id * 0x9E3779B97F4A7C15ULL
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL
    state.s0 = z ^ (z >> 31)

    z = state.s0 + 0x9E3779B97F4A7C15ULL
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL
    state.s1 = z ^ (z >> 31)

    # Ensure state is not all zeros
    if state.s0 == 0 and state.s1 == 0:
        state.s0 = 1

cdef inline double random_uniform_nogil(RngState* state) noexcept nogil:
    """Generate uniform random number in [0, 1)"""
    cdef uint64_t x = xoroshiro128plus_next(state)
    # Use the top 53 bits for a double in [0, 1)
    return <double>(x >> 11) * (1.0 / 9007199254740992.0)

cdef inline double random_gaussian_nogil(RngState* state) noexcept nogil:
    """Generate standard normal random number using Box-Muller transform"""
    cdef double u1 = random_uniform_nogil(state)
    cdef double u2 = random_uniform_nogil(state)

    # Avoid log(0)
    while u1 <= 1e-10:
        u1 = random_uniform_nogil(state)

    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2)

cdef inline int sign_nogil(double x) noexcept nogil:
    """Return sign of x: 1 if positive, -1 if negative, 0 if zero"""
    return (x > 0) - (x < 0)

# ============================================================================
# DDM Parallel Simulator
# ============================================================================

def ddm_parallel(
    np.ndarray[float, ndim=1] v,
    np.ndarray[float, ndim=1] a,
    np.ndarray[float, ndim=1] z,
    np.ndarray[float, ndim=1] t,
    np.ndarray[float, ndim=1] deadline,
    np.ndarray[float, ndim=1] s,
    float delta_t = 0.001,
    float max_t = 20.0,
    int n_samples = 20000,
    int n_trials = 1,
    int n_threads = 0,
    random_state = None,
    return_option = 'full',
    smooth_unif = False,
):
    """
    Parallel DDM simulator using Cython prange with nogil.

    This version parallelizes across samples for each trial, which provides
    good speedup for large n_samples values.

    Parameters
    ----------
    v : np.ndarray[float32]
        Drift rate for each trial
    a : np.ndarray[float32]
        Boundary separation for each trial
    z : np.ndarray[float32]
        Starting point (as proportion of a) for each trial
    t : np.ndarray[float32]
        Non-decision time for each trial
    deadline : np.ndarray[float32]
        Maximum allowed RT for each trial
    s : np.ndarray[float32]
        Noise standard deviation for each trial
    delta_t : float
        Time step size (default: 0.001)
    max_t : float
        Maximum simulation time (default: 20.0)
    n_samples : int
        Number of samples per trial (default: 20000)
    n_trials : int
        Number of trials (default: 1)
    n_threads : int
        Number of threads to use (0 = auto, default: 0)
    random_state : int or None
        Random seed for reproducibility
    return_option : str
        'full' or 'minimal' (default: 'full')
    smooth_unif : bool
        Apply uniform smoothing (default: False)

    Returns
    -------
    dict
        Dictionary with 'rts', 'choices', and 'metadata'
    """

    # Determine thread count
    if n_threads <= 0:
        import os
        n_threads = os.cpu_count() or 4

    # Handle random state
    cdef uint64_t seed
    if random_state is None:
        seed = <uint64_t>np.random.default_rng().integers(0, 2**63)
    else:
        seed = <uint64_t>random_state

    # Allocate output arrays
    cdef np.ndarray[float, ndim=3] rts = np.zeros((n_samples, n_trials, 1), dtype=np.float32)
    cdef np.ndarray[int, ndim=3] choices = np.zeros((n_samples, n_trials, 1), dtype=np.intc)

    # Create memory views for nogil access
    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices
    cdef float[:] v_view = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s

    # Precompute constants
    cdef float delta_t_sqrt = sqrt(delta_t)
    cdef int num_steps = <int>((max_t / delta_t) + 1)

    # Loop variables
    cdef int n, k, thread_id
    cdef float y, t_particle, sqrt_st, deadline_tmp, smooth_u
    cdef RngState rng
    cdef bint do_smooth = smooth_unif

    # Run parallel simulation
    for k in range(n_trials):
        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
        sqrt_st = delta_t_sqrt * s_view[k]

        with nogil, parallel(num_threads=n_threads):
            for n in prange(n_samples, schedule='static'):
                # Initialize thread-local RNG
                thread_id = cython.parallel.threadid()
                init_rng_state(&rng, seed + <uint64_t>k * 1000000 + <uint64_t>n, thread_id)

                # Initialize particle
                y = z_view[k] * a_view[k]
                t_particle = 0.0

                # Random walk until boundary or deadline
                while y <= a_view[k] and y >= 0 and t_particle <= deadline_tmp:
                    y = y + v_view[k] * delta_t + sqrt_st * random_gaussian_nogil(&rng)
                    t_particle = t_particle + delta_t

                # Apply smoothing if requested
                if do_smooth:
                    if t_particle == 0.0:
                        smooth_u = random_uniform_nogil(&rng) * 0.5 * delta_t
                    elif t_particle < deadline_tmp:
                        smooth_u = (0.5 - random_uniform_nogil(&rng)) * delta_t
                    else:
                        smooth_u = 0.0
                else:
                    smooth_u = 0.0

                # Store results
                rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u
                choices_view[n, k, 0] = sign_nogil(y)

                # Enforce deadline
                if rts_view[n, k, 0] >= deadline_view[k] or deadline_view[k] <= 0:
                    rts_view[n, k, 0] = -999.0

    # Build metadata
    metadata = {
        'simulator': 'ddm_parallel_cython',
        'possible_choices': [-1, 1],
        'n_samples': n_samples,
        'n_trials': n_trials,
        'n_threads': n_threads,
        'boundary_fun_type': 'constant',
    }

    if return_option == 'full':
        metadata.update({
            'delta_t': delta_t,
            'max_t': max_t,
            'v': v, 'a': a, 'z': z, 't': t,
            'deadline': deadline, 's': s,
        })

    return {'rts': rts, 'choices': choices, 'metadata': metadata}


def ddm_flexbound_parallel(
    np.ndarray[float, ndim=1] v,
    np.ndarray[float, ndim=1] a,
    np.ndarray[float, ndim=1] z,
    np.ndarray[float, ndim=1] t,
    np.ndarray[float, ndim=1] deadline,
    np.ndarray[float, ndim=1] s,
    float delta_t = 0.001,
    float max_t = 20.0,
    int n_samples = 20000,
    int n_trials = 1,
    int n_threads = 0,
    boundary_fun = None,
    boundary_params = None,
    random_state = None,
    return_option = 'full',
    smooth_unif = False,
):
    """
    Parallel DDM simulator with flexible boundaries.

    Note: The boundary is precomputed before the parallel section
    since boundary_fun is a Python callable.
    """
    if boundary_params is None:
        boundary_params = {}

    # Determine thread count
    if n_threads <= 0:
        import os
        n_threads = os.cpu_count() or 4

    # Handle random state
    cdef uint64_t seed
    if random_state is None:
        seed = <uint64_t>np.random.default_rng().integers(0, 2**63)
    else:
        seed = <uint64_t>random_state

    # Precompute time array and boundary
    cdef int num_steps = <int>((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t, dtype=np.float32)

    # Allocate output arrays
    cdef np.ndarray[float, ndim=3] rts = np.zeros((n_samples, n_trials, 1), dtype=np.float32)
    cdef np.ndarray[int, ndim=3] choices = np.zeros((n_samples, n_trials, 1), dtype=np.intc)
    cdef np.ndarray[float, ndim=1] boundary = np.zeros(num_steps, dtype=np.float32)

    # Create memory views
    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices
    cdef float[:] v_view = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s
    cdef float[:] boundary_view = boundary

    # Precompute constants
    cdef float delta_t_sqrt = sqrt(delta_t)

    # Loop variables
    cdef int n, k, ix, thread_id
    cdef float y, t_particle, sqrt_st, deadline_tmp, smooth_u
    cdef RngState rng
    cdef bint do_smooth = smooth_unif

    for k in range(n_trials):
        # Precompute boundary for this trial (must be done with GIL)
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
        boundary[:] = boundary_fun(t=t_s, **boundary_params_tmp).astype(np.float32)

        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
        sqrt_st = delta_t_sqrt * s_view[k]

        with nogil, parallel(num_threads=n_threads):
            for n in prange(n_samples, schedule='static'):
                # Initialize thread-local RNG
                thread_id = cython.parallel.threadid()
                init_rng_state(&rng, seed + <uint64_t>k * 1000000 + <uint64_t>n, thread_id)

                # Initialize particle with flexible boundary
                y = (-1) * boundary_view[0] + (z_view[k] * 2 * boundary_view[0])
                t_particle = 0.0
                ix = 0

                # Random walk
                while (y >= (-1) * boundary_view[ix]) and (y <= boundary_view[ix]) and (t_particle <= deadline_tmp):
                    y = y + v_view[k] * delta_t + sqrt_st * random_gaussian_nogil(&rng)
                    t_particle = t_particle + delta_t
                    ix = ix + 1
                    if ix >= num_steps:
                        ix = num_steps - 1

                # Apply smoothing
                if do_smooth:
                    if t_particle == 0.0:
                        smooth_u = random_uniform_nogil(&rng) * 0.5 * delta_t
                    elif t_particle < deadline_tmp:
                        smooth_u = (0.5 - random_uniform_nogil(&rng)) * delta_t
                    else:
                        smooth_u = 0.0
                else:
                    smooth_u = 0.0

                # Store results
                rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u
                choices_view[n, k, 0] = sign_nogil(y)

                # Enforce deadline
                if rts_view[n, k, 0] >= deadline_view[k] or deadline_view[k] <= 0:
                    rts_view[n, k, 0] = -999.0

    # Build metadata
    metadata = {
        'simulator': 'ddm_flexbound_parallel_cython',
        'possible_choices': [-1, 1],
        'n_samples': n_samples,
        'n_trials': n_trials,
        'n_threads': n_threads,
        'boundary_fun_type': boundary_fun.__name__ if boundary_fun else 'unknown',
    }

    if return_option == 'full':
        metadata.update({
            'delta_t': delta_t,
            'max_t': max_t,
            'v': v, 'a': a, 'z': z, 't': t,
            'deadline': deadline, 's': s,
            'boundary': boundary,
        })

    return {'rts': rts, 'choices': choices, 'metadata': metadata}


def full_ddm_parallel(
    np.ndarray[float, ndim=1] v,
    np.ndarray[float, ndim=1] a,
    np.ndarray[float, ndim=1] z,
    np.ndarray[float, ndim=1] t,
    np.ndarray[float, ndim=1] sz,
    np.ndarray[float, ndim=1] sv,
    np.ndarray[float, ndim=1] st,
    np.ndarray[float, ndim=1] deadline,
    np.ndarray[float, ndim=1] s,
    float delta_t = 0.001,
    float max_t = 20.0,
    int n_samples = 20000,
    int n_trials = 1,
    int n_threads = 0,
    boundary_fun = None,
    boundary_params = None,
    random_state = None,
    return_option = 'full',
    smooth_unif = False,
):
    """
    Parallel full DDM simulator with inter-trial variability.

    Parameters
    ----------
    sz : np.ndarray[float32]
        Starting point variability
    sv : np.ndarray[float32]
        Drift rate variability
    st : np.ndarray[float32]
        Non-decision time variability
    """
    if boundary_params is None:
        boundary_params = {}

    # Determine thread count
    if n_threads <= 0:
        import os
        n_threads = os.cpu_count() or 4

    # Handle random state
    cdef uint64_t seed
    if random_state is None:
        seed = <uint64_t>np.random.default_rng().integers(0, 2**63)
    else:
        seed = <uint64_t>random_state

    # Precompute time array and boundary
    cdef int num_steps = <int>((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t, dtype=np.float32)

    # Allocate output arrays
    cdef np.ndarray[float, ndim=3] rts = np.zeros((n_samples, n_trials, 1), dtype=np.float32)
    cdef np.ndarray[int, ndim=3] choices = np.zeros((n_samples, n_trials, 1), dtype=np.intc)
    cdef np.ndarray[float, ndim=1] boundary = np.zeros(num_steps, dtype=np.float32)

    # Create memory views
    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices
    cdef float[:] v_view = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] sz_view = sz
    cdef float[:] sv_view = sv
    cdef float[:] st_view = st
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s
    cdef float[:] boundary_view = boundary

    # Precompute constants
    cdef float delta_t_sqrt = sqrt(delta_t)

    # Loop variables
    cdef int n, k, ix, thread_id
    cdef float y, t_particle, sqrt_st, deadline_tmp, smooth_u
    cdef float drift_increment, t_tmp
    cdef RngState rng
    cdef bint do_smooth = smooth_unif

    for k in range(n_trials):
        # Precompute boundary for this trial
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
        boundary[:] = boundary_fun(t=t_s, **boundary_params_tmp).astype(np.float32)

        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
        sqrt_st = delta_t_sqrt * s_view[k]

        with nogil, parallel(num_threads=n_threads):
            for n in prange(n_samples, schedule='static'):
                # Initialize thread-local RNG
                thread_id = cython.parallel.threadid()
                init_rng_state(&rng, seed + <uint64_t>k * 1000000 + <uint64_t>n, thread_id)

                # Initialize with inter-trial variability
                y = (-1) * boundary_view[0] + (z_view[k] * 2 * boundary_view[0])
                y = y + 2 * (random_uniform_nogil(&rng) - 0.5) * sz_view[k]

                drift_increment = (v_view[k] + sv_view[k] * random_gaussian_nogil(&rng)) * delta_t
                t_tmp = t_view[k] + (2 * (random_uniform_nogil(&rng) - 0.5) * st_view[k])

                t_particle = 0.0
                ix = 0

                # Random walk
                while (y >= (-1) * boundary_view[ix]) and (y <= boundary_view[ix]) and (t_particle <= deadline_tmp):
                    y = y + drift_increment + sqrt_st * random_gaussian_nogil(&rng)
                    t_particle = t_particle + delta_t
                    ix = ix + 1
                    if ix >= num_steps:
                        ix = num_steps - 1

                # Apply smoothing
                if do_smooth:
                    if t_particle == 0.0:
                        smooth_u = random_uniform_nogil(&rng) * 0.5 * delta_t
                    elif t_particle < deadline_tmp:
                        smooth_u = (0.5 - random_uniform_nogil(&rng)) * delta_t
                    else:
                        smooth_u = 0.0
                else:
                    smooth_u = 0.0

                # Store results
                rts_view[n, k, 0] = t_particle + t_tmp + smooth_u
                choices_view[n, k, 0] = sign_nogil(y)

                # Enforce deadline
                if rts_view[n, k, 0] >= deadline_view[k] or deadline_view[k] <= 0:
                    rts_view[n, k, 0] = -999.0

    # Build metadata
    metadata = {
        'simulator': 'full_ddm_parallel_cython',
        'possible_choices': [-1, 1],
        'n_samples': n_samples,
        'n_trials': n_trials,
        'n_threads': n_threads,
        'boundary_fun_type': boundary_fun.__name__ if boundary_fun else 'unknown',
    }

    if return_option == 'full':
        metadata.update({
            'delta_t': delta_t,
            'max_t': max_t,
            'v': v, 'a': a, 'z': z, 't': t,
            'sz': sz, 'sv': sv, 'st': st,
            'deadline': deadline, 's': s,
            'boundary': boundary,
        })

    return {'rts': rts, 'choices': choices, 'metadata': metadata}
