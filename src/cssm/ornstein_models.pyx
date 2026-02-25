# Globaly settings for cython
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: initializedcheck=False

"""
Ornstein-Uhlenbeck Models

This module contains simulator functions for Ornstein-Uhlenbeck diffusion processes,
which include mean-reversion (drift toward a central value) in addition to standard
drift and diffusion.
"""

import cython
from libc.math cimport sqrt, fmin
from cython.parallel import prange, parallel
from libc.stdint cimport uint64_t

import numpy as np
cimport numpy as np

# Import utility functions from the _utils module
from cssm._utils import (
    set_seed,
    draw_uniform,
    draw_gaussian,
    sign,
    setup_simulation,
    compute_boundary,
    compute_smooth_unif,
    enforce_deadline,
    compute_deadline_tmp,
    build_full_metadata,
    build_minimal_metadata,
    build_return_dict,
)

# =============================================================================
# C-LEVEL GSL RNG (for parallel execution)
# =============================================================================
# Uses GSL's validated Ziggurat implementation for correct variance.
# Per-thread RNG states are allocated before parallel block and freed after.

cdef extern from "gsl_rng.h" nogil:
    # Struct with known size (pointer) so Cython can allocate arrays
    ctypedef struct ssms_rng_state:
        void* rng  # gsl_rng pointer (void* for Cython compatibility)

    void ssms_rng_alloc(ssms_rng_state* state)
    void ssms_rng_free(ssms_rng_state* state)
    void ssms_rng_seed(ssms_rng_state* state, uint64_t seed)
    float ssms_gaussian_f32(ssms_rng_state* state)
    double ssms_uniform(ssms_rng_state* state)
    uint64_t ssms_mix_seed(uint64_t base, uint64_t t1, uint64_t t2)

# Use Cython's parallel module for thread ID
from cython.parallel cimport threadid

# Type alias for consistency
ctypedef ssms_rng_state RngState

# Include shared constants (MAX_THREADS, etc.)
include "_constants.pxi"

# Wrapper functions for GSL RNG
cdef inline void rng_alloc(RngState* state) noexcept nogil:
    ssms_rng_alloc(state)

cdef inline void rng_free(RngState* state) noexcept nogil:
    ssms_rng_free(state)

cdef inline void rng_seed(RngState* state, uint64_t seed) noexcept nogil:
    ssms_rng_seed(state, seed)

cdef inline uint64_t rng_mix_seed(uint64_t base_seed, uint64_t thread_id, uint64_t trial_id) noexcept nogil:
    return ssms_mix_seed(base_seed, thread_id, trial_id)

cdef inline float rng_gaussian_f32(RngState* state) noexcept nogil:
    return ssms_gaussian_f32(state)

# Import OpenMP status check
from cssm._openmp_status import check_parallel_request

DTYPE = np.float32

def ornstein_uhlenbeck(np.ndarray[float, ndim = 1] v, # drift parameter
                       np.ndarray[float, ndim = 1] z, # starting point bias
                       np.ndarray[float, ndim = 1] g, # decay parameter
                       np.ndarray[float, ndim = 1] t,
                       np.ndarray[float, ndim = 1] deadline,
                       np.ndarray[float, ndim = 1] s, # noise sigma
                       float delta_t = 0.001, # size of timestep
                       float max_t = 20, # maximal time in trial
                       int n_samples = 20000, # number of samples from process
                       int n_trials = 1,
                       boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                       boundary_params = {},
                       random_state = None,
                       return_option = 'full',
                       smooth_unif = False,
                       int n_threads = 1,
                       **kwargs):
    """
    Simulate reaction times and choices from an Ornstein-Uhlenbeck process with flexible boundaries.

    Args:
        v (np.ndarray): Drift parameter for each trial.
        a (np.ndarray): Initial boundary separation for each trial.
        z (np.ndarray): Starting point bias for each trial.
        g (np.ndarray): Decay parameter for each trial.
        t (np.ndarray): Non-decision time for each trial.
        deadline (np.ndarray): Maximum reaction time allowed for each trial.
        s (np.ndarray): Noise sigma for each trial.
        delta_t (float): Size of timestep for simulation (default: 0.001).
        max_t (float): Maximum time for simulation (default: 20).
        n_samples (int): Number of samples to simulate per trial (default: 20000).
        n_trials (int): Number of trials to simulate (default: 1).
        boundary_fun (callable): Function defining the shape of the boundary over time.
        boundary_params (dict): Parameters for the boundary function.
        random_state (int or None): Seed for random number generator (default: None).
        return_option (str): 'full' for complete output, 'minimal' for basic output (default: 'full').
        smooth_unif (bool): Whether to apply uniform smoothing to reaction times (default: False).
        **kwargs: Additional keyword arguments.

    Returns:
        dict: A dictionary containing simulated reaction times, choices, and metadata.
              The exact contents depend on the return_option.

    Raises:
        ValueError: If return_option is not 'full' or 'minimal'.
    """

    # Check if parallel execution is requested and available
    n_threads = check_parallel_request(n_threads)

    # Get seed for reproducibility
    cdef uint64_t seed = random_state if random_state is not None else np.random.randint(0, 2**31)

    setup = setup_simulation(n_samples, n_trials, max_t, delta_t, random_state)

    # Extract arrays and create memory views for C-level performance
    traj = setup['traj']
    rts = setup['rts']
    choices = setup['choices']
    cdef float[:, :] traj_view = traj
    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices
    cdef float[:] gaussian_values = setup['gaussian_values']
    cdef float[:] uniform_values = setup['uniform_values']
    t_s = setup['t_s']
    cdef int num_draws = setup['num_draws']
    cdef float delta_t_sqrt = setup['delta_t_sqrt']
    cdef int num_steps = int((max_t / delta_t) + 1)
    cdef float c_max_t = max_t

    # Param views
    cdef float[:] v_view  = v
    cdef float[:] z_view = z
    cdef float[:] g_view = g
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s

    # Boundary storage for the upper bound
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef float y, t_particle, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0
    cdef Py_ssize_t mu = 0

    # Variables for parallel execution
    cdef float[:, :] boundaries_all
    cdef float[:] deadlines_tmp_all
    cdef float[:] sqrt_st_all
    cdef RngState[MAX_THREADS] rng_states  # Per-thread RNG states
    cdef uint64_t combined_seed
    cdef float z_k, v_k, g_k, t_k, s_k, sqrt_st_k, deadline_k
    cdef float bound_val, neg_bound_val, noise
    cdef int choice
    cdef int tid  # Thread ID
    cdef int i_thread
    cdef int c_n_threads

    # Flattened parallelization variables
    cdef Py_ssize_t flat_idx
    cdef Py_ssize_t total_iterations
    cdef int c_n_samples = n_samples

    # =========================================================================
    # SEQUENTIAL PATH (n_threads == 1): Original algorithm
    # =========================================================================
    if n_threads == 1:
        for k in range(n_trials):
            # Precompute boundary evaluations
            boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
            compute_boundary(boundary, t_s, boundary_fun,
                            boundary_params_tmp)

            deadline_tmp = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])
            sqrt_st = delta_t_sqrt * s_view[k]
            # Loop over samples
            for n in range(n_samples):
                y = (-1) * boundary_view[0] + (z_view[k] * 2 * boundary_view[0])
                t_particle = 0.0
                ix = 0

                if n == 0:
                    if k == 0:
                        traj_view[0, 0] = y

                # Random walker
                while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t_particle <= deadline_tmp:
                    y += ((v_view[k] - (g_view[k] * y)) * delta_t) + sqrt_st * gaussian_values[m]
                    t_particle += delta_t
                    ix += 1
                    m += 1

                    if n == 0:
                        if k == 0:
                            traj_view[ix, 0] = y

                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                smooth_u = compute_smooth_unif(smooth_unif, t_particle, deadline_tmp, delta_t, uniform_values[mu])
                mu += 1
                if mu == num_draws:
                    uniform_values = draw_uniform(num_draws)
                    mu = 0

                rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u
                choices_view[n, k, 0] = sign(y)

                enforce_deadline(rts_view, deadline_view, n, k, 0)

    # =========================================================================
    # PARALLEL PATH (n_threads > 1): FLATTENED OpenMP parallelization with C RNG
    # Parallelizes over (n_trials × n_samples) for optimal efficiency
    # =========================================================================
    else:
        # Precompute ALL trial data outside nogil
        boundaries_all_np = np.zeros((n_trials, num_steps), dtype=DTYPE)
        deadlines_tmp_all_np = np.zeros(n_trials, dtype=DTYPE)
        sqrt_st_all_np = np.zeros(n_trials, dtype=DTYPE)

        for k in range(n_trials):
            boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
            compute_boundary(boundary, t_s, boundary_fun, boundary_params_tmp)
            boundaries_all_np[k, :] = boundary
            deadlines_tmp_all_np[k] = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])
            sqrt_st_all_np[k] = delta_t_sqrt * s_view[k]

        boundaries_all = boundaries_all_np
        deadlines_tmp_all = deadlines_tmp_all_np
        sqrt_st_all = sqrt_st_all_np

        # Total iterations = n_trials × n_samples
        total_iterations = <Py_ssize_t>n_trials * <Py_ssize_t>n_samples
        c_n_threads = n_threads

        # Allocate per-thread GSL RNGs BEFORE parallel block
        for i_thread in range(c_n_threads):
            rng_alloc(&rng_states[i_thread])

        with nogil, parallel(num_threads=n_threads):
            for flat_idx in prange(total_iterations, schedule='dynamic'):
                # Get thread ID for per-thread RNG
                tid = threadid()

                # Compute (k, n) from flat index
                k = flat_idx // c_n_samples
                n = flat_idx % c_n_samples

                z_k = z_view[k]
                v_k = v_view[k]
                g_k = g_view[k]
                t_k = t_view[k]
                deadline_k = deadline_view[k]
                sqrt_st_k = sqrt_st_all[k]
                deadline_tmp = deadlines_tmp_all[k]

                # Re-seed per-thread RNG with unique seed for this (trial, sample)
                combined_seed = rng_mix_seed(seed, <uint64_t>k, <uint64_t>n)
                rng_seed(&rng_states[tid], combined_seed)

                bound_val = boundaries_all[k, 0]
                y = (-1.0) * bound_val + (z_k * 2.0 * bound_val)
                t_particle = 0.0
                ix = 0

                while True:
                    bound_val = boundaries_all[k, ix]
                    neg_bound_val = -bound_val

                    if y < neg_bound_val or y > bound_val or t_particle > deadline_tmp:
                        break

                    noise = rng_gaussian_f32(&rng_states[tid])
                    # Ornstein-Uhlenbeck: drift with decay toward 0
                    y = y + ((v_k - g_k * y) * delta_t) + sqrt_st_k * noise
                    t_particle = t_particle + delta_t
                    ix = ix + 1

                    if ix >= num_steps:
                        break

                rts_view[n, k, 0] = t_particle + t_k

                # Choice based on sign of y (same as sequential path)
                if y > 0.0:
                    choice = 1
                elif y < 0.0:
                    choice = -1
                else:
                    choice = 0
                choices_view[n, k, 0] = choice

                if rts_view[n, k, 0] >= deadline_k or deadline_k <= 0:
                    rts_view[n, k, 0] = -999.0

        # Free per-thread GSL RNGs AFTER parallel block
        for i_thread in range(c_n_threads):
            rng_free(&rng_states[i_thread])

    # Build minimal metadata first
    minimal_meta = build_minimal_metadata(
        simulator_name='ornstein_uhlenbeck',
        possible_choices=[-1, 1],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )

    if return_option == 'full':
        sim_config = {'delta_t': delta_t, 'max_t': max_t, 'n_threads': n_threads}
        params = {
            'v': v, 'z': z, 'g': g, 't': t,
            'deadline': deadline, 's': s
        }
        full_meta = build_full_metadata(
            minimal_metadata=minimal_meta,
            params=params,
            sim_config=sim_config,
            boundary_fun=boundary_fun,
            boundary=boundary,
            traj=traj,
            boundary_params=boundary_params
        )
        return build_return_dict(rts, choices, full_meta)

    elif return_option == 'minimal':
        return build_return_dict(rts, choices, minimal_meta)

    else:
        raise ValueError('return_option must be either "full" or "minimal"')
