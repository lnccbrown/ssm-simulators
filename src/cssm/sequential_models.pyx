# Globaly settings for cython
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: initializedcheck=False

"""
Sequential Sampling Models

This module contains simulator functions for sequential (two-stage) decision models.
These models involve hierarchical decisions where an initial high-dimensional choice
influences subsequent low-dimensional choices.
"""

import cython
import warnings
from libc.math cimport sqrt, log, fmax
from libc.stdint cimport uint64_t

import numpy as np
cimport numpy as np

# OpenMP imports
from cython.parallel cimport prange, parallel, threadid

# Import utility functions from the _utils module
from cssm._utils import (
    set_seed,
    draw_gaussian,
    draw_uniform,
    sign,
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

# Type alias for consistency
ctypedef ssms_rng_state RngState

# Wrapper functions for GSL RNG
cdef inline void rng_alloc(RngState* state) noexcept nogil:
    ssms_rng_alloc(state)

cdef inline void rng_free(RngState* state) noexcept nogil:
    ssms_rng_free(state)

cdef inline void rng_seed(RngState* state, uint64_t seed) noexcept nogil:
    ssms_rng_seed(state, seed)

cdef inline uint64_t rng_mix_seed(uint64_t base, uint64_t t, uint64_t n) noexcept nogil:
    return ssms_mix_seed(base, t, n)

cdef inline float rng_gaussian_f32(RngState* state) noexcept nogil:
    return ssms_gaussian_f32(state)

cdef inline double rng_uniform(RngState* state) noexcept nogil:
    return ssms_uniform(state)

DTYPE = np.float32

# Maximum number of time steps for stack-allocated arrays
# For max_t=20, delta_t=0.001 -> 20001 steps. Use 25000 for safety margin.
DEF MAX_STEPS = 25000

# Include shared constants (MAX_THREADS, etc.)
include "_constants.pxi"

# Sequential Models ------------------------------------

def ddm_flexbound_seq2(np.ndarray[float, ndim = 1] vh,
                       np.ndarray[float, ndim = 1] vl1,
                       np.ndarray[float, ndim = 1] vl2,
                       np.ndarray[float, ndim = 1] zh,
                       np.ndarray[float, ndim = 1] zl1,
                       np.ndarray[float, ndim = 1] zl2,
                       np.ndarray[float, ndim = 1] t,
                       np.ndarray[float, ndim = 1] deadline,
                       np.ndarray[float, ndim = 1] s, # noise sigma
                       float delta_t = 0.001,
                       float max_t = 20,
                       int n_samples = 20000,
                       int n_trials = 1,
                       print_info = True,
                       boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                       boundary_params = {},
                       random_state = None,
                       return_option = 'full',
                       smooth_unif = False,
                       int n_threads = 1,
                       **kwargs):
    """
    Simulate reaction times and choices from a sequential two-stage drift diffusion model with flexible boundaries.

    Parameters:
    -----------
    vh : np.ndarray, shape (n_trials,)
        Drift rate for the high-level decision.
    vl1, vl2 : np.ndarray, shape (n_trials,)
        Drift rates for the two low-level decisions.
    a : np.ndarray, shape (n_trials,)
        Initial boundary separation.
    zh : np.ndarray, shape (n_trials,)
        Starting point bias for the high-level decision.
    zl1, zl2 : np.ndarray, shape (n_trials,)
        Starting point biases for the two low-level decisions.
    t : np.ndarray, shape (n_trials,)
        Non-decision time.
    deadline : np.ndarray, shape (n_trials,)
        Deadline for each trial.
    s : np.ndarray, shape (n_trials,)
        Diffusion coefficient (standard deviation of the diffusion process).
    delta_t : float, optional
        Size of the time step in the simulation (default: 0.001).
    max_t : float, optional
        Maximum time for the simulation (default: 20).
    n_samples : int, optional
        Number of samples to simulate (default: 20000).
    n_trials : int, optional
        Number of trials to simulate (default: 1).
    print_info : bool, optional
        Whether to print information during the simulation (default: True).
    boundary_fun : callable, optional
        Function that determines the decision boundary over time (default: None).
    boundary_params : dict, optional
        Parameters for the boundary function (default: {}).
    random_state : int or None, optional
        Seed for the random number generator (default: None).
    return_option : str, optional
        Determines the amount of data returned. Can be 'full' or 'minimal' (default: 'full').
    smooth_unif : bool, optional
        If True, applies uniform smoothing to reaction times (default: False).

    Returns:
    --------
    dict
        A dictionary containing simulated reaction times, choices, and metadata.
        The exact contents depend on the 'return_option' parameter.
    """
    # Check OpenMP availability for parallel execution
    if n_threads > 1:
        from cssm._openmp_status import check_parallel_request
        n_threads = check_parallel_request(n_threads)

    # Sequential path (n_threads=1)
    if n_threads == 1:
        return _ddm_flexbound_seq2_sequential(
            vh, vl1, vl2, zh, zl1, zl2, t, deadline, s,
            delta_t, max_t, n_samples, n_trials, print_info,
            boundary_fun, boundary_params, random_state,
            return_option, smooth_unif
        )

    # Parallel path
    # Param views
    cdef float[:] vh_view = vh
    cdef float[:] vl1_view = vl1
    cdef float[:] vl2_view = vl2
    cdef float[:] zh_view = zh
    cdef float[:] zl1_view = zl1
    cdef float[:] zl2_view = zl2
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s

    rts = np.zeros((n_samples, n_trials, 1), dtype=DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype=np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    # Trajectory disabled in parallel mode
    traj = np.zeros((int(max_t / delta_t) + 1, 3), dtype=DTYPE)
    traj[:, :] = -999
    cdef float[:, :] traj_view = traj

    cdef float delta_t_sqrt = sqrt(delta_t)
    cdef int num_steps = int((max_t / delta_t) + 1)
    cdef int c_n_samples = n_samples

    # Pre-compute boundaries for all trials (outside nogil)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundaries_all_np = np.zeros((n_trials, len(t_s)), dtype=DTYPE)
    deadlines_tmp = np.zeros(n_trials, dtype=DTYPE)
    sqrt_st_arr = np.zeros(n_trials, dtype=DTYPE)

    cdef Py_ssize_t k_precomp
    for k_precomp in range(n_trials):
        boundary_params_tmp = {key: boundary_params[key][k_precomp] for key in boundary_params.keys()}
        boundary_tmp = np.zeros(t_s.shape, dtype=DTYPE)
        compute_boundary(boundary_tmp, t_s, boundary_fun, boundary_params_tmp)
        boundaries_all_np[k_precomp, :] = boundary_tmp
        deadlines_tmp[k_precomp] = compute_deadline_tmp(max_t, deadline_view[k_precomp], t_view[k_precomp])
        sqrt_st_arr[k_precomp] = delta_t_sqrt * s_view[k_precomp]

    cdef float[:, :] boundaries_view = boundaries_all_np
    cdef float[:] deadlines_view = deadlines_tmp
    cdef float[:] sqrt_st_view = sqrt_st_arr

    # Per-thread RNG states for parallel execution
    cdef RngState[MAX_THREADS] rng_states
    cdef uint64_t base_seed = random_state if random_state is not None else np.random.randint(0, 2**31)
    cdef uint64_t combined_seed
    cdef int tid  # Thread ID
    cdef int i_thread
    cdef int c_n_threads = n_threads

    # Flattened parallel loop variables
    cdef Py_ssize_t total_iterations = <Py_ssize_t>n_trials * <Py_ssize_t>n_samples
    cdef Py_ssize_t flat_idx, k, n
    cdef int ix, ix1, ix2
    cdef float y_h, y_l1, y_l2, t_particle, t_particle1, t_particle2
    cdef float deadline_tmp_k, sqrt_st_k, bound_val, noise
    cdef int choice_val, decision_taken

    # Allocate per-thread GSL RNGs BEFORE parallel block
    for i_thread in range(c_n_threads):
        rng_alloc(&rng_states[i_thread])

    # Parallel execution over FLATTENED iteration space
    with nogil, parallel(num_threads=n_threads):
        for flat_idx in prange(total_iterations, schedule='dynamic'):
            # Get thread ID for per-thread RNG
            tid = threadid()

            k = flat_idx // c_n_samples  # trial index
            n = flat_idx % c_n_samples   # sample index

            # Re-seed per-thread RNG with unique seed for this (trial, sample)
            combined_seed = rng_mix_seed(base_seed, <uint64_t>k, <uint64_t>n)
            rng_seed(&rng_states[tid], combined_seed)

            deadline_tmp_k = deadlines_view[k]
            sqrt_st_k = sqrt_st_view[k]
            choice_val = 0
            decision_taken = 0

            # Stage 1: High-dimensional walker
            bound_val = boundaries_view[k, 0]
            y_h = (-1.0) * bound_val + (zh_view[k] * 2.0 * bound_val)
            t_particle = 0.0
            ix = 0

            while True:
                bound_val = boundaries_view[k, ix]
                if y_h < (-1.0) * bound_val or y_h > bound_val or t_particle > deadline_tmp_k:
                    break
                noise = rng_gaussian_f32(&rng_states[tid])
                y_h = y_h + (vh_view[k] * delta_t) + (sqrt_st_k * noise)
                t_particle = t_particle + delta_t
                ix = ix + 1
                if ix >= num_steps:
                    break

            # Determine high-dim choice
            bound_val = boundaries_view[k, ix] if ix < num_steps else 0.0
            if t_particle >= max_t:
                # At max_t, make stochastic choice
                if bound_val <= 0.0:
                    if rng_uniform(&rng_states[tid]) <= 0.5:
                        choice_val = 2
                elif rng_uniform(&rng_states[tid]) <= ((y_h + bound_val) / (2.0 * bound_val)):
                    choice_val = 2

                # Low dim choice random (a priori bias)
                if choice_val == 0:
                    if rng_uniform(&rng_states[tid]) <= zl1_view[k]:
                        choice_val = 1
                else:
                    if rng_uniform(&rng_states[tid]) <= zl2_view[k]:
                        choice_val = 3
                decision_taken = 1
            else:
                # High-dim choice based on position
                if bound_val <= 0.0:
                    if rng_uniform(&rng_states[tid]) <= 0.5:
                        choice_val = 2
                elif rng_uniform(&rng_states[tid]) <= ((y_h + bound_val) / (2.0 * bound_val)):
                    choice_val = 2

                # Stage 2: Low-dimensional walker (only run the one determined by high-dim choice)
                ix1 = ix
                ix2 = ix
                t_particle1 = t_particle
                t_particle2 = t_particle

                # Initialize low-dim walkers at current boundary
                y_l1 = (-1.0) * bound_val + (zl1_view[k] * 2.0 * bound_val)
                y_l2 = (-1.0) * bound_val + (zl2_view[k] * 2.0 * bound_val)

                if choice_val == 0:
                    # Check if already at boundary
                    if y_l1 >= bound_val or y_l1 <= (-1.0) * bound_val:
                        if rng_uniform(&rng_states[tid]) < zl1_view[k]:
                            choice_val = 1
                        decision_taken = 1
                    else:
                        # Run low-dim walker 1
                        while True:
                            bound_val = boundaries_view[k, ix1]
                            if y_l1 < (-1.0) * bound_val or y_l1 > bound_val or t_particle1 > deadline_tmp_k:
                                break
                            noise = rng_gaussian_f32(&rng_states[tid])
                            y_l1 = y_l1 + (vl1_view[k] * delta_t) + (sqrt_st_k * noise)
                            t_particle1 = t_particle1 + delta_t
                            ix1 = ix1 + 1
                            if ix1 >= num_steps:
                                break
                        t_particle = t_particle1
                        ix = ix1
                else:
                    # Check if already at boundary
                    if y_l2 >= bound_val or y_l2 <= (-1.0) * bound_val:
                        if rng_uniform(&rng_states[tid]) < zl2_view[k]:
                            choice_val = 3
                        decision_taken = 1
                    else:
                        # Run low-dim walker 2
                        while True:
                            bound_val = boundaries_view[k, ix2]
                            if y_l2 < (-1.0) * bound_val or y_l2 > bound_val or t_particle2 > deadline_tmp_k:
                                break
                            noise = rng_gaussian_f32(&rng_states[tid])
                            y_l2 = y_l2 + (vl2_view[k] * delta_t) + (sqrt_st_k * noise)
                            t_particle2 = t_particle2 + delta_t
                            ix2 = ix2 + 1
                            if ix2 >= num_steps:
                                break
                        t_particle = t_particle2
                        ix = ix2

            # Final low-dim choice if not yet decided
            if decision_taken == 0:
                bound_val = boundaries_view[k, ix] if ix < num_steps else 0.0
                if choice_val == 0:
                    # Low-dim choice based on y_l1 position
                    if bound_val <= 0.0:
                        if rng_uniform(&rng_states[tid]) <= 0.5:
                            choice_val = 1
                    elif rng_uniform(&rng_states[tid]) <= ((y_l1 + bound_val) / (2.0 * bound_val)):
                        choice_val = 1
                else:
                    # Low-dim choice based on y_l2 position
                    if bound_val <= 0.0:
                        if rng_uniform(&rng_states[tid]) <= 0.5:
                            choice_val = 3
                    elif rng_uniform(&rng_states[tid]) <= ((y_l2 + bound_val) / (2.0 * bound_val)):
                        choice_val = 3

            rts_view[n, k, 0] = t_particle + t_view[k]
            choices_view[n, k, 0] = choice_val

            # Enforce deadline
            if rts_view[n, k, 0] > deadline_view[k]:
                rts_view[n, k, 0] = -999.0

    # Free per-thread GSL RNGs AFTER parallel block
    for i_thread in range(c_n_threads):
        rng_free(&rng_states[i_thread])

    # Build minimal metadata first
    minimal_meta = build_minimal_metadata(
        simulator_name='ddm_flexbound_seq2',
        possible_choices=[0, 1, 2, 3],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )

    if return_option == 'full':
        sim_config = {'delta_t': delta_t, 'max_t': max_t, 'n_threads': n_threads}
        params = {
            'vh': vh, 'vl1': vl1, 'vl2': vl2,
            'zh': zh, 'zl1': zl1, 'zl2': zl2,
            't': t, 'deadline': deadline, 's': s
        }
        full_meta = build_full_metadata(
            minimal_metadata=minimal_meta,
            params=params,
            sim_config=sim_config,
            boundary_fun=boundary_fun,
            boundary=boundaries_all_np[0] if n_trials > 0 else np.array([]),
            traj=traj,
            boundary_params=boundary_params
        )
        return build_return_dict(rts, choices, full_meta)

    elif return_option == 'minimal':
        return build_return_dict(rts, choices, minimal_meta)

    else:
        raise ValueError('return_option must be either "full" or "minimal"')


def _ddm_flexbound_seq2_sequential(
    np.ndarray[float, ndim = 1] vh,
    np.ndarray[float, ndim = 1] vl1,
    np.ndarray[float, ndim = 1] vl2,
    np.ndarray[float, ndim = 1] zh,
    np.ndarray[float, ndim = 1] zl1,
    np.ndarray[float, ndim = 1] zl2,
    np.ndarray[float, ndim = 1] t,
    np.ndarray[float, ndim = 1] deadline,
    np.ndarray[float, ndim = 1] s,
    float delta_t,
    float max_t,
    int n_samples,
    int n_trials,
    print_info,
    boundary_fun,
    boundary_params,
    random_state,
    return_option,
    smooth_unif
):
    """Sequential implementation of ddm_flexbound_seq2 (original code path)."""

    set_seed(random_state)
    # Param views
    cdef float[:] vh_view = vh
    cdef float[:] vl1_view = vl1
    cdef float[:] vl2_view = vl2
    cdef float[:] zh_view = zh
    cdef float[:] zl1_view = zl1
    cdef float[:] zl2_view = zl2
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s
    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices
    cdef int decision_taken = 0

    # TD: Add Trajectory
    traj = np.zeros((int(max_t / delta_t) + 1, 3), dtype = DTYPE)
    traj[:, :] = -999
    cdef float[:, :] traj_view = traj

    cdef float delta_t_sqrt = sqrt(delta_t)

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef float y_h, t_particle, t_particle1, t_particle2, y_l, y_l1, y_l2, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n, ix, ix1, ix2, k
    cdef Py_ssize_t m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    cdef Py_ssize_t mu = 0
    cdef float[:] uniform_values = draw_uniform(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
        compute_boundary(boundary, t_s, boundary_fun,
                        boundary_params_tmp)

        deadline_tmp = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])
        sqrt_st = delta_t_sqrt * s_view[k]
        # Loop over samples
        for n in range(n_samples):
            decision_taken = 0
            t_particle = 0.0 # reset time
            ix = 0 # reset boundary index

            # Random walker 1 (high dimensional)
            y_h = (-1) * boundary_view[0] + (zh_view[k] * 2 * (boundary_view[0]))  # reset starting position

            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y_h

            while y_h >= (-1) * boundary_view[ix] and y_h <= boundary_view[ix] and t_particle <= deadline_tmp:
                y_h += (vh_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                t_particle += delta_t
                ix += 1
                m += 1

                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

                if n == 0:
                    if k == 0:
                        traj_view[ix, 0] = y_h

            # If we are already at maximum t, to generate a choice we just sample from a bernoulli
            if t_particle >= max_t:
                # High dim choice depends on position of particle
                if boundary_view[ix] <= 0:
                    if uniform_values[mu] <= 0.5:
                        choices_view[n, k, 0] += 2
                elif uniform_values[mu] <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
                        choices_view[n, k, 0] += 2
                mu += 1
                if mu == num_draws:
                    uniform_values = draw_uniform(num_draws)
                    mu = 0

                # Low dim choice random (didn't even get to process it if rt is at max after first choice)
                # so we just apply a priori bias
                if choices_view[n, k, 0] == 0:
                    if uniform_values[mu] <= zl1_view[k]:
                        choices_view[n, k, 0] += 1
                else:
                    if uniform_values[mu] <= zl2_view[k]:
                        choices_view[n, k, 0] += 1
                mu += 1
                if mu == num_draws:
                    uniform_values = draw_uniform(num_draws)
                    mu = 0
                rts_view[n, k, 0] = t_particle
                decision_taken = 1
            else:
                # If boundary is negative (or 0) already, we flip a coin
                if boundary_view[ix] <= 0:
                    if uniform_values[mu] <= 0.5:
                        choices_view[n, k, 0] += 2
                # Otherwise apply rule from above
                elif uniform_values[mu] <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
                    choices_view[n, k, 0] += 2
                mu += 1
                if mu == num_draws:
                    uniform_values = draw_uniform(num_draws)
                    mu = 0

                y_l1 = (-1) * boundary_view[ix] + (zl1_view[k] * 2 * (boundary_view[ix]))
                y_l2 = (-1) * boundary_view[ix] + (zl2_view[k] * 2 * (boundary_view[ix]))

                ix1 = ix
                t_particle1 = t_particle
                ix2 = ix
                t_particle2 = t_particle

                # Figure out negative bound for low level
                if choices_view[n, k, 0] == 0:
                    # In case boundary is negative already, we flip a coin with bias determined by w_l_ parameter
                    if (y_l1 >= boundary_view[ix]) or (y_l1 <= ((-1) * boundary_view[ix])):
                        if uniform_values[mu] < zl1_view[k]:
                            choices_view[n, k, 0] += 1
                        mu += 1
                        if mu == num_draws:
                            uniform_values = draw_uniform(num_draws)
                            mu = 0
                        decision_taken = 1

                    if n == 0:
                        if k == 0:
                            traj_view[ix, 1] = y_l1
                else:
                    # In case boundary is negative already, we flip a coin with bias determined by w_l_ parameter
                    if (y_l2 >= boundary_view[ix]) or (y_l2 <= ((-1) * boundary_view[ix])):
                        if uniform_values[mu] < zl2_view[k]:
                            choices_view[n, k, 0] += 1
                        mu += 1
                        if mu == num_draws:
                            uniform_values = draw_uniform(num_draws)
                            mu = 0
                        decision_taken = 1

                    if n == 0:
                        if k == 0:
                            traj_view[ix, 2] = y_l2

                # Random walker low level (1)
                if (choices_view[n, k, 0] == 0) | ((n == 0) & (k == 0)):
                    while (y_l1 >= ((-1) * boundary_view[ix1])) and (y_l1 <= boundary_view[ix1]) and (t_particle1 <= deadline_tmp):
                        y_l1 += (vl1_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                        t_particle1 += delta_t
                        ix1 += 1
                        m += 1
                        if m == num_draws:
                            gaussian_values = draw_gaussian(num_draws)
                            m = 0

                        if n == 0:
                            if k == 0:
                                traj_view[ix1, 1] = y_l1

                # Random walker low level (2)
                if (choices_view[n, k, 0] == 2) | ((n == 0) & (k == 0)):
                    while (y_l2 >= ((-1) * boundary_view[ix2])) and (y_l2 <= boundary_view[ix2]) and (t_particle2 <= deadline_tmp):
                        y_l2 += (vl2_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                        t_particle2 += delta_t
                        ix2 += 1
                        m += 1
                        if m == num_draws:
                            gaussian_values = draw_gaussian(num_draws)
                            m = 0

                        if n == 0:
                            if k == 0:
                                traj_view[ix2, 2] = y_l2

                # Get back to single t_particle
                if (choices_view[n, k, 0] == 0):
                    t_particle = t_particle1
                    ix = ix1
                    y_l = y_l1
                else:
                    t_particle = t_particle2
                    ix = ix2
                    y_l = y_l2

            smooth_u = compute_smooth_unif(smooth_unif, t_particle, deadline_tmp, delta_t, uniform_values[mu])
            mu += 1
            if mu == num_draws:
                uniform_values = draw_uniform(num_draws)
                mu = 0

            # Add nondecision time and smoothing of rt
            rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u

            # Take account of deadline
            enforce_deadline(rts_view, deadline_view, n, k, 0)

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add one deterministically
            # y at lower bound --> choice_view[n, k, 0] stays the same deterministically

            # If boundary is negative (or 0) already, we flip a coin
            if not decision_taken:
                if boundary_view[ix] <= 0:
                    if uniform_values[mu] <= 0.5:
                        choices_view[n, k, 0] += 1
                # Otherwise apply rule from above
                elif uniform_values[mu] <= ((y_l + boundary_view[ix]) / (2 * boundary_view[ix])):
                    choices_view[n, k, 0] += 1
                mu += 1
                if mu == num_draws:
                    uniform_values = draw_uniform(num_draws)
                    mu = 0

    # Build minimal metadata first
    minimal_meta = build_minimal_metadata(
        simulator_name='ddm_flexbound_seq2',
        possible_choices=[0, 1, 2, 3],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )

    if return_option == 'full':
        sim_config = {'delta_t': delta_t, 'max_t': max_t}
        params = {
            'vh': vh, 'vl1': vl1, 'vl2': vl2,
            'zh': zh, 'zl1': zl1, 'zl2': zl2,
            't': t, 'deadline': deadline, 's': s
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
# -----------------------------------------------------------------------------------------------

# NOTE: ddm_flexbound_par2 has been moved to parallel_models.pyx to avoid duplication
# Import it from there: from cssm.parallel_models import ddm_flexbound_par2

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flexbound_mic2_ornstein(np.ndarray[float, ndim = 1] vh,
                                np.ndarray[float, ndim = 1] vl1,
                                np.ndarray[float, ndim = 1] vl2,
                                np.ndarray[float, ndim = 1] a,
                                np.ndarray[float, ndim = 1] zh,
                                np.ndarray[float, ndim = 1] zl1,
                                np.ndarray[float, ndim = 1] zl2,
                                np.ndarray[float, ndim = 1] d, # damper (1 --> no drift on low level until high level done, 0 --> full drift on low level)
                                np.ndarray[float, ndim = 1] g, # inhibition parameter for the low dim choice procress while high dim is running
                                np.ndarray[float, ndim = 1] t,
                                np.ndarray[float, ndim = 1] deadline,
                                np.ndarray[float, ndim = 1] s_pre_high_level_choice,
                                np.ndarray[float, ndim = 1] s, # noise sigma
                                float delta_t = 0.001,
                                float max_t = 20,
                                int n_samples = 20000,
                                int n_trials = 1,
                                print_info = True,
                                boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                                boundary_params = {},
                                random_state = None,
                                return_option = 'full',
                                smooth_unif = False,
                                int n_threads = 1,
                                **kwargs):
    """
    Simulate (rt, choice) tuples from a DDM with flexible boundaries and Ornstein-Uhlenbeck process.

    Parameters:
    -----------
    vh, vl1, vl2 : np.ndarray, shape (n_trials,)
        Drift rates for high-level, low-level 1, and low-level 2 processes.
    a : np.ndarray, shape (n_trials,)
        Initial boundary separation.
    zh, zl1, zl2 : np.ndarray, shape (n_trials,)
        Starting points for high-level, low-level 1, and low-level 2 processes.
    d : np.ndarray, shape (n_trials,)
        Damping parameter (1: no drift on low level until high level done, 0: full drift on low level).
    g : np.ndarray, shape (n_trials,)
        Inhibition parameter for the low-dimensional choice process while high-dimensional is running.
    t : np.ndarray, shape (n_trials,)
        Non-decision time.
    deadline : np.ndarray, shape (n_trials,)
        Response deadline.
    s_pre_high_level_choice : np.ndarray, shape (n_trials,)
        Noise level before high-level choice is made.
    s : np.ndarray, shape (n_trials,)
        Noise level (sigma).
    delta_t : float, optional
        Size of time steps for simulation (default: 0.001).
    max_t : float, optional
        Maximum time for simulation (default: 20).
    n_samples : int, optional
        Number of samples to simulate (default: 20000).
    n_trials : int, optional
        Number of trials to simulate (default: 1).
    print_info : bool, optional
        Whether to print information during simulation (default: True).
    boundary_fun : callable, optional
        Boundary function of t and potentially other parameters (default: None).
    boundary_params : dict, optional
        Parameters for the boundary function (default: {}).
    random_state : int or None, optional
        Random seed for reproducibility (default: None).
    return_option : str, optional
        Determines what to return, either 'full' or 'minimal' (default: 'full').
    smooth_unif : bool, optional
        Whether to use smooth uniform distribution for RT jitter (default: False).
    n_threads : int, optional
        Number of threads for parallel execution (default: 1).

    Returns:
    --------
    dict
        Dictionary containing simulated data and metadata. The exact contents depend on the return_option.
    """
    # MIC2 models have complex simultaneous walker dynamics with on-the-fly bias computation
    # that prevent effective parallelization despite correct OpenMP code generation.
    # Always use sequential path with a warning if n_threads > 1 was requested.
    if n_threads > 1:
        warnings.warn(
            "ddm_flexbound_mic2_ornstein: n_threads > 1 does not provide speedup for this model "
            "due to complex simultaneous walker dynamics. Using sequential execution.",
            UserWarning
        )

    # Always use sequential path - parallel path doesn't provide speedup
    return _ddm_flexbound_mic2_ornstein_sequential(
        vh, vl1, vl2, a, zh, zl1, zl2, d, g, t, deadline,
        s_pre_high_level_choice, s, delta_t, max_t, n_samples, n_trials,
        print_info, boundary_fun, boundary_params, random_state,
        return_option, smooth_unif
    )


def _ddm_flexbound_mic2_ornstein_sequential(
    np.ndarray[float, ndim = 1] vh,
    np.ndarray[float, ndim = 1] vl1,
    np.ndarray[float, ndim = 1] vl2,
    np.ndarray[float, ndim = 1] a,
    np.ndarray[float, ndim = 1] zh,
    np.ndarray[float, ndim = 1] zl1,
    np.ndarray[float, ndim = 1] zl2,
    np.ndarray[float, ndim = 1] d,
    np.ndarray[float, ndim = 1] g,
    np.ndarray[float, ndim = 1] t,
    np.ndarray[float, ndim = 1] deadline,
    np.ndarray[float, ndim = 1] s_pre_high_level_choice,
    np.ndarray[float, ndim = 1] s,
    float delta_t,
    float max_t,
    int n_samples,
    int n_trials,
    print_info,
    boundary_fun,
    boundary_params,
    random_state,
    return_option,
    smooth_unif
):
    """Sequential implementation of ddm_flexbound_mic2_ornstein."""
    set_seed(random_state)
    # Param views
    cdef float[:] vh_view = vh
    cdef float[:] vl1_view = vl1
    cdef float[:] vl2_view = vl2
    cdef float[:] a_view = a
    cdef float[:] zh_view = zh
    cdef float[:] zl1_view = zl1
    cdef float[:] zl2_view = zl2
    cdef float[:] d_view = d
    cdef float[:] g_view = g
    cdef float[:] t_view = t
    cdef float[:] s_view = s
    cdef float[:] s_pre_high_level_choice_view = s_pre_high_level_choice
    cdef float[:] deadline_view = deadline

    # TD: Add trajectory --> same issue as with par2 model above... might need to make a separate simulator for trajectories
    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    rts_low = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    rts_high = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)

    cdef float[:, :, :] rts_view = rts
    cdef float[:, :, :] rts_high_view = rts_high
    cdef float[:, :, :] rts_low_view = rts_low
    cdef int[:, :, :] choices_view = choices

    traj = np.zeros((int(max_t / delta_t) + 1, 3), dtype = DTYPE)
    traj[:, :] = -999
    cdef float[:, :] traj_view = traj

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    #cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    # Y particle trace
    bias_trace_l1 = np.zeros(num_draws, dtype = DTYPE)
    bias_trace_l2 = np.zeros(num_draws, dtype = DTYPE)
    cdef float[:] bias_trace_l1_view = bias_trace_l1
    cdef float[:] bias_trace_l2_view = bias_trace_l2

    cdef float y_h, y_l, y_l1, y_l2
    cdef float v_l, v_l1, v_l2,
    cdef float t_h, t_l, t_l1, t_l2, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n, ix, ix1, ix2, ix_l, ix_tmp, ix1_tmp, ix2_tmp, k
    cdef Py_ssize_t m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    cdef Py_ssize_t mu = 0
    cdef float[:] uniform_values = draw_uniform(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
        compute_boundary(boundary, t_s, boundary_fun,
                        boundary_params_tmp)

        deadline_tmp = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])
        sqrt_st = delta_t_sqrt * s_view[k]
        # Loop over samples
        for n in range(n_samples):
            choices_view[n, k, 0] = 0 # reset choice
            t_h = 0 # reset time high dimension
            t_l = 0 # reset time low dimension
            t_l1 = 0 # reset time low dimension (1)
            t_l2 = 0 # reset time low dimension (2)
            ix = 0 # reset boundary index
            ix1 = 0 # reset boundary index (1)
            ix2 = 0 # reset boundary index (2)

            # Initialize walkers
            # Particle
            y_h = (-1) * boundary_view[0] + (zh_view[k] * 2 * (boundary_view[0]))
            # Relative particle position (used as resource allocator for low dim choice)
            bias_trace_l2_view[0] = ((y_h + boundary_view[0]) / (2 * boundary_view[0]))
            bias_trace_l1_view[0] = 1.0 - bias_trace_l2_view[0]

            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y_h

            # Random walks until y_h hits bound
            while (y_h >= ((-1) * boundary_view[ix])) and ((y_h <= boundary_view[ix])) and (t_h <= deadline_tmp):
                y_h += (vh_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                bias_trace_l2_view[ix] = ((y_h + boundary_view[ix]) / (2 * boundary_view[ix]))
                bias_trace_l1_view[ix] = 1.0 - bias_trace_l2_view[ix]
                t_h += delta_t
                ix += 1
                m += 1
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

                if n == 0:
                    if k == 0:
                        traj_view[ix, 0] = y_h

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add 2 deterministically
            # y at lower bound --> choice_view[n, k, 0] stay the same deterministically

            # If boundary is negative (or 0) already, we flip a coin
            if boundary_view[ix] <= 0:
                if uniform_values[mu] <= 0.5:
                    choices_view[n, k, 0] += 2
            # Otherwise, apply rule from above
            elif uniform_values[mu] <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 2
            mu += 1
            if mu == num_draws:
                uniform_values = draw_uniform(num_draws)
                mu = 0

            y_l2 = (- 1) * boundary_view[0] + (zl2_view[k] * 2 * (boundary_view[0]))
            y_l1 = (- 1) * boundary_view[0] + (zl1_view[k] * 2 * (boundary_view[0]))

            if choices_view[n, k, 0] == 0:
                 # Fill bias tracea until max_rt reached
                ix1_tmp = ix + 1
                while ix1_tmp < num_draws:
                    bias_trace_l2_view[ix1_tmp] = 0.0
                    bias_trace_l1_view[ix1_tmp] = 1.0
                    ix1_tmp += 1

            else: # Store intermediate choice
                # Fill bias tracea until max_rt reached
                ix2_tmp = ix + 1
                while ix2_tmp < num_draws:
                    bias_trace_l2_view[ix2_tmp] = 1.0
                    bias_trace_l1_view[ix2_tmp] = 0.0
                    ix2_tmp += 1

            # lower level random walker (1)
            if (choices_view[n, k, 0] == 0) | ((n == 0) & (k == 0)):
                while (y_l1 >= ((-1) * boundary_view[ix1])) and (y_l1 <= boundary_view[ix1]) and (t_l1 <= deadline_tmp):
                    if (bias_trace_l1_view[ix1] < 1) and (bias_trace_l1_view[ix1] > 0):
                        # main propagation if bias_trace is between 0 and 1 (high level choice is not yet made)
                        y_l1 += (((vl1_view[k] * bias_trace_l1_view[ix1] * (1 - d_view[k])) - (g_view[k] * y_l1)) * delta_t)
                        # add gaussian displacement
                        y_l1 += (sqrt_st * gaussian_values[m]) * s_pre_high_level_choice_view[k]
                    else:
                        # main propagation if bias_trace is not between 0 and 1 (high level choice is already made)
                        y_l1 += (vl1_view[k] * delta_t)
                        # add gaussian displacement
                        y_l1 += (sqrt_st * gaussian_values[m])


                    # propagate time and indices
                    t_l1 += delta_t
                    ix1 += 1
                    m += 1
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                    if n == 0:
                        if k == 0:
                            traj_view[ix1, 1] = y_l1

            # lower level random walker (2)
            if (choices_view[n, k, 0] == 2) | ((n == 0) & (k == 0)):
                while (y_l2 >= ((-1) * boundary_view[ix2])) and (y_l2 <= boundary_view[ix2]) and (t_l2 <= deadline_tmp):
                    if (bias_trace_l2_view[ix2] < 1) and (bias_trace_l2_view[ix2] > 0):
                        # main propagation if bias_trace is between 0 and 1 (high level choice is not yet made)
                        y_l2 += (((vl2_view[k] * bias_trace_l2_view[ix2] * (1 - d_view[k])) - (g_view[k] * y_l2)) * delta_t)
                        # add gaussian displacement
                        y_l2 += (sqrt_st * gaussian_values[m]) * s_pre_high_level_choice_view[k]
                    else:
                        # main propagation if bias_trace is not between 0 and 1 (high level choice is already made)
                        y_l2 += (vl2_view[k] * delta_t)
                        # add gaussian displacement
                        y_l2 += (sqrt_st * gaussian_values[m])

                    # propagate time and indices
                    t_l2 += delta_t
                    ix2 += 1
                    m += 1
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                    if n == 0:
                        if k == 0:
                            traj_view[ix2, 2] = y_l2

            # Get back to single y_l and t_l
            if (choices_view[n, k, 0] == 0):
                t_l = t_l1
                y_l = y_l1
                ix_l = ix1
            else:
                t_l = t_l2
                y_l = y_l2
                ix_l = ix2

            smooth_u = compute_smooth_unif(smooth_unif, fmax(t_h, t_l), deadline_tmp, delta_t, uniform_values[mu])
            mu += 1
            if mu == num_draws:
                uniform_values = draw_uniform(num_draws)
                mu = 0

            rts_view[n, k, 0] = fmax(t_h, t_l) + t_view[k]
            rts_high_view[n, k, 0] = t_h + t_view[k]
            rts_low_view[n, k, 0] = t_l + t_view[k]

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add one deterministically
            # y at lower bound --> choice_view[n, k, 0] stays the same deterministically

            # If boundary is negative (or 0) already, we flip a coin
            if boundary_view[ix] <= 0:
                if uniform_values[mu] <= 0.5:
                    choices_view[n, k, 0] += 1
            # Otherwise apply rule from above
            elif uniform_values[mu] <= ((y_l + boundary_view[ix_l]) / (2 * boundary_view[ix_l])):
                choices_view[n, k, 0] += 1
            mu += 1
            if mu == num_draws:
                uniform_values = draw_uniform(num_draws)
                mu = 0

            enforce_deadline(rts_view, deadline_view, n, k, 0)

    # Build minimal metadata first
    minimal_meta = build_minimal_metadata(
        simulator_name='ddm_flexbound_mic2_ornstein',
        possible_choices=[0, 1, 2, 3],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )

    # Extra arrays for this model
    extra_arrays_dict = {'rts_high': rts_high, 'rts_low': rts_low}

    if return_option == 'full':
        sim_config = {'delta_t': delta_t, 'max_t': max_t}
        params = {
            'vh': vh, 'vl1': vl1, 'vl2': vl2,
            'a': a, 'zh': zh, 'zl1': zl1, 'zl2': zl2,
            'd': d, 'g': g, 't': t, 'deadline': deadline,
            's_pre_high_level_choice': s_pre_high_level_choice,
            's': s
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
        return build_return_dict(rts, choices, full_meta, extra_arrays=extra_arrays_dict)

    elif return_option == 'minimal':
        return build_return_dict(rts, choices, minimal_meta, extra_arrays=extra_arrays_dict)

    else:
        raise ValueError('return_option must be either "full" or "minimal"')

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flexbound_mic2_multinoise(np.ndarray[float, ndim = 1] vh,
                                  np.ndarray[float, ndim = 1] vl1,
                                  np.ndarray[float, ndim = 1] vl2,
                                  np.ndarray[float, ndim = 1] a,
                                  np.ndarray[float, ndim = 1] zh,
                                  np.ndarray[float, ndim = 1] zl1,
                                  np.ndarray[float, ndim = 1] zl2,
                                  np.ndarray[float, ndim = 1] d,
                                  np.ndarray[float, ndim = 1] t,
                                  np.ndarray[float, ndim = 1] deadline,
                                  np.ndarray[float, ndim = 1] s,
                                  float delta_t = 0.001,
                                  float max_t = 20,
                                  int n_samples = 20000,
                                  int n_trials = 1,
                                  print_info = True,
                                  boundary_fun = None,
                                  boundary_params = {},
                                  random_state = None,
                                  return_option = 'full',
                                  smooth_unif = False,
                                  int n_threads = 1,
                                  **kwargs):
    """
    Simulates a multi-level decision-making process using a drift-diffusion model with flexible boundaries.

    Parameters:
    -----------
    vh, vl1, vl2 : np.ndarray, shape (n_trials,)
        Drift rates for high-level, low-level 1, and low-level 2 processes.
    a : np.ndarray, shape (n_trials,)
        Initial boundary separation.
    zh, zl1, zl2 : np.ndarray, shape (n_trials,)
        Starting points for high-level, low-level 1, and low-level 2 processes.
    d : np.ndarray, shape (n_trials,)
        Damping parameter (1: no drift on low level until high level done, 0: full drift on low level).
    t : np.ndarray, shape (n_trials,)
        Non-decision time.
    deadline : np.ndarray, shape (n_trials,)
        Response deadline.
    s : np.ndarray, shape (n_trials,)
        Noise level (standard deviation).
    delta_t : float, optional
        Size of time steps for simulation (default: 0.001).
    max_t : float, optional
        Maximum time for each trial (default: 20).
    n_samples : int, optional
        Number of samples to simulate (default: 20000).
    n_trials : int, optional
        Number of trials to simulate (default: 1).
    print_info : bool, optional
        Whether to print information during simulation (default: True).
    boundary_fun : callable, optional
        Function defining the decision boundary (default: None).
    boundary_params : dict, optional
        Parameters for the boundary function (default: {}).
    random_state : int or None, optional
        Seed for random number generator (default: None).
    return_option : str, optional
        Determines what to return, either 'full' or 'minimal' (default: 'full').
    smooth_unif : bool, optional
        Whether to use smooth uniform distribution for certain calculations (default: False).

    Returns:
    --------
    dict
        A dictionary containing simulation results. The exact contents depend on the return_option:
        - 'full': Contains 'rts', 'choices', 'rts_high', 'rts_low', and detailed 'metadata'.
        - 'minimal': Contains 'rts', 'choices', 'rts_high', 'rts_low', and minimal 'metadata'.

    Raises:
    -------
    ValueError
        If an invalid return_option is provided.

    Notes:
    ------
    This function implements a complex drift-diffusion model for multi-level decision-making,
    incorporating flexible boundaries and multiple noise sources.

    Note: n_threads parameter is accepted for API consistency but parallel execution shows
    limited speedup for MIC2 models due to bias trace array dependencies.
    """
    # Warn if n_threads > 1 (limited benefit for MIC2 models)
    if n_threads > 1:
        warnings.warn(
            "ddm_flexbound_mic2_multinoise: n_threads > 1 has limited speedup due to "
            "bias trace dependencies. Using sequential execution.",
            UserWarning
        )

    set_seed(random_state)
    # Param views
    cdef float[:] vh_view = vh
    cdef float[:] vl1_view = vl1
    cdef float[:] vl2_view = vl2
    cdef float[:] a_view = a
    cdef float[:] zh_view = zh
    cdef float[:] zl1_view = zl1
    cdef float[:] zl2_view = zl2
    cdef float[:] d_view = d
    cdef float[:] t_view = t
    cdef float[:] s_view = s
    cdef float[:] deadline_view = deadline

    # TD: Add trajectory --> same issue as with par2 model above... might need to make a separate simulator for trajectories
    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    rts_low = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    rts_high = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)

    cdef float[:, :, :] rts_view = rts
    cdef float[:, :, :] rts_high_view = rts_high
    cdef float[:, :, :] rts_low_view = rts_low
    cdef int[:, :, :] choices_view = choices

    traj = np.zeros((int(max_t / delta_t) + 1, 3), dtype = DTYPE)
    traj[:, :] = -999
    cdef float[:, :] traj_view = traj

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    #cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    # Y particle trace
    bias_trace_l1 = np.zeros(num_draws, dtype = DTYPE)
    bias_trace_l2 = np.zeros(num_draws, dtype = DTYPE)
    cdef float[:] bias_trace_l1_view = bias_trace_l1
    cdef float[:] bias_trace_l2_view = bias_trace_l2

    cdef float y_h, y_l, y_l1, y_l2
    cdef float v_l, v_l1, v_l2
    cdef float t_h, t_l, t_l1, t_l2, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n, ix, ix1, ix2, ix_l, ix_tmp, ix1_tmp, ix2_tmp, k
    cdef Py_ssize_t m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    cdef Py_ssize_t mu = 0
    cdef float[:] uniform_values = draw_uniform(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
        compute_boundary(boundary, t_s, boundary_fun,
                        boundary_params_tmp)

        deadline_tmp = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])
        sqrt_st = delta_t_sqrt * s_view[k]
        # Loop over samples
        for n in range(n_samples):
            choices_view[n, k, 0] = 0 # reset choice
            t_h = 0 # reset time high dimension
            t_l = 0 # reset time low dimension
            t_l1 = 0 # reset time low dimension (1)
            t_l2 = 0 # reset time low dimension (2)
            ix = 0 # reset boundary index
            ix1 = 0 # reset boundary index (1)
            ix2 = 0 # reset boundary index (2)

            # Initialize walkers
            # Particle
            y_h = (-1) * boundary_view[0] + (zh_view[k] * 2 * (boundary_view[0]))
            # Relative particle position (used as resource allocator for low dim choice)
            bias_trace_l2_view[0] = ((y_h + boundary_view[0]) / (2 * boundary_view[0]))
            bias_trace_l1_view[0] = 1.0 - bias_trace_l2_view[0]

            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y_h

            # Random walks until y_h hits bound
            while (y_h >= ((-1) * boundary_view[ix])) and ((y_h <= boundary_view[ix])) and (t_h <= deadline_tmp):
                y_h += (vh_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                bias_trace_l2_view[ix] = ((y_h + boundary_view[ix]) / (2 * boundary_view[ix]))
                bias_trace_l1_view[ix] = 1.0 - bias_trace_l2_view[ix]
                t_h += delta_t
                ix += 1
                m += 1
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

                if n == 0:
                    if k == 0:
                        traj_view[ix, 0] = y_h

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add 2 deterministically
            # y at lower bound --> choice_view[n, k, 0] stay the same deterministically

            # If boundary is negative (or 0) already, we flip a coin
            if boundary_view[ix] <= 0:
                if uniform_values[mu] <= 0.5:
                    choices_view[n, k, 0] += 2
            # Otherwise, apply rule from above
            elif uniform_values[mu] <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 2
            mu += 1
            if mu == num_draws:
                uniform_values = draw_uniform(num_draws)
                mu = 0

            y_l2 = (- 1) * boundary_view[0] + (zl2_view[k] * 2 * (boundary_view[0]))
            y_l1 = (- 1) * boundary_view[0] + (zl1_view[k] * 2 * (boundary_view[0]))

            if choices_view[n, k, 0] == 0:
                 # Fill bias tracea until max_rt reached
                ix1_tmp = ix + 1
                while ix1_tmp < num_draws:
                    bias_trace_l2_view[ix1_tmp] = 0.0
                    bias_trace_l1_view[ix1_tmp] = 1.0
                    ix1_tmp += 1

            else: # Store intermediate choice
                # Fill bias tracea until max_rt reached
                ix2_tmp = ix + 1
                while ix2_tmp < num_draws:
                    bias_trace_l2_view[ix2_tmp] = 1.0
                    bias_trace_l1_view[ix2_tmp] = 0.0
                    ix2_tmp += 1

            # lower level random walker (1)
            if (choices_view[n, k, 0] == 0) | ((n == 0) & (k == 0)):
                while (y_l1 >= ((-1) * boundary_view[ix1])) and (y_l1 <= boundary_view[ix1]) and (t_l1 <= deadline_tmp):
                    if (bias_trace_l1_view[ix1] < 1) and (bias_trace_l1_view[ix1] > 0):
                        # main propagation if bias_trace is between 0 and 1 (high level choice is not yet made)
                        y_l1 += (((vl1_view[k] * bias_trace_l1_view[ix1] * (1 - d_view[k]))) * delta_t)
                        # add gaussian displacement
                        # we multiply by bias_trace_view to make low level variance depend on high level trace
                        y_l1 += (sqrt_st * gaussian_values[m]) * bias_trace_l1_view[ix1]
                    else:
                        # main propagation if bias_trace is not between 0 and 1 (high level choice is already made)
                        y_l1 += (vl1_view[k] * delta_t)
                        # add gaussian displacement
                        y_l1 += (sqrt_st * gaussian_values[m])


                    # propagate time and indices
                    t_l1 += delta_t
                    ix1 += 1
                    m += 1
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                    if n == 0:
                        if k == 0:
                            traj_view[ix1, 1] = y_l1

            # lower level random walker (2)
            if (choices_view[n, k, 0] == 2) | ((n == 0) & (k == 0)):
                while (y_l2 >= ((-1) * boundary_view[ix2])) and (y_l2 <= boundary_view[ix2]) and (t_l2 <= deadline_tmp):
                    if (bias_trace_l2_view[ix2] < 1) and (bias_trace_l2_view[ix2] > 0):
                        # main propagation if bias_trace is between 0 and 1 (high level choice is not yet made)
                        y_l2 += (((vl2_view[k] * bias_trace_l2_view[ix2] * (1 - d_view[k]))) * delta_t)
                        # add gaussian displacement
                        # we multiply by bias_trace_view to make low level variance depend on high level trace
                        y_l2 += (sqrt_st * gaussian_values[m]) * bias_trace_l2_view[ix2]
                    else:
                        # main propagation if bias_trace is not between 0 and 1 (high level choice is already made)
                        y_l2 += (vl2_view[k] * delta_t)
                        # add gaussian displacement
                        y_l2 += (sqrt_st * gaussian_values[m])


                    # propagate time and indices
                    t_l2 += delta_t
                    ix2 += 1
                    m += 1
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                    if n == 0:
                        if k == 0:
                            traj_view[ix2, 2] = y_l2

            # Get back to single y_l and t_l
            if (choices_view[n, k, 0] == 0):
                t_l = t_l1
                y_l = y_l1
                ix_l = ix1
            else:
                t_l = t_l2
                y_l = y_l2
                ix_l = ix2

            smooth_u = compute_smooth_unif(smooth_unif, fmax(t_h, t_l), deadline_tmp, delta_t, uniform_values[mu])
            mu += 1
            if mu == num_draws:
                uniform_values = draw_uniform(num_draws)
                mu = 0

            rts_view[n, k, 0] = fmax(t_h, t_l) + t_view[k]
            rts_high_view[n, k, 0] = t_h + t_view[k]
            rts_low_view[n, k, 0] = t_l + t_view[k]

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add one deterministically
            # y at lower bound --> choice_view[n, k, 0] stays the same deterministically

            # If boundary is negative (or 0) already, we flip a coin
            if boundary_view[ix] <= 0:
                if uniform_values[mu] <= 0.5:
                    choices_view[n, k, 0] += 1
            # Otherwise apply rule from above
            elif uniform_values[mu] <= ((y_l + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 1
            mu += 1
            if mu == num_draws:
                uniform_values = draw_uniform(num_draws)
                mu = 0

            enforce_deadline(rts_view, deadline_view, n, k, 0)

    # Build minimal metadata first
    minimal_meta = build_minimal_metadata(
        simulator_name='ddm_flexbound_mic2_adj',
        possible_choices=[0, 1, 2, 3],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )

    # Extra arrays for this model
    extra_arrays_dict = {'rts_high': rts_high, 'rts_low': rts_low}

    if return_option == 'full':
        sim_config = {'delta_t': delta_t, 'max_t': max_t}
        params = {
            'vh': vh, 'vl1': vl1, 'vl2': vl2,
            'a': a, 'zh': zh, 'zl1': zl1, 'zl2': zl2,
            'd': d, 't': t, 'deadline': deadline, 's': s
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
        return build_return_dict(rts, choices, full_meta, extra_arrays=extra_arrays_dict)

    elif return_option == 'minimal':
        return build_return_dict(rts, choices, minimal_meta, extra_arrays=extra_arrays_dict)

    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# ----------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flexbound_mic2_ornstein_multinoise(np.ndarray[float, ndim = 1] vh,
                                           np.ndarray[float, ndim = 1] vl1,
                                           np.ndarray[float, ndim = 1] vl2,
                                           np.ndarray[float, ndim = 1] a,
                                           np.ndarray[float, ndim = 1] zh,
                                           np.ndarray[float, ndim = 1] zl1,
                                           np.ndarray[float, ndim = 1] zl2,
                                           np.ndarray[float, ndim = 1] d, # damper (1 --> no drift on low level until high level done, 0 --> full drift on low level)
                                           np.ndarray[float, ndim = 1] g, # inhibition parameter for the low dim choice procress while high dim is running
                                           np.ndarray[float, ndim = 1] t,
                                           np.ndarray[float, ndim = 1] deadline,
                                           np.ndarray[float, ndim = 1] s, # noise sigma
                                           float delta_t = 0.001,
                                           float max_t = 20,
                                           int n_samples = 20000,
                                           int n_trials = 1,
                                           print_info = True,
                                           boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                                           boundary_params = {},
                                           random_state = None,
                                           return_option = 'full',
                                           smooth_unif = False,
                                           int n_threads = 1,
                                           **kwargs):
    """
    Simulate reaction times and choices from a DDM with flexible boundaries and multiple noise sources.

    This function implements a drift diffusion model (DDM) with flexible boundaries, incorporating
    both high-dimensional and low-dimensional choice processes, and multiple noise sources.

    Parameters:
    -----------
    vh, vl1, vl2 : np.ndarray[float, ndim=1]
        Drift rates for high-dimensional and two low-dimensional processes.
    a : np.ndarray[float, ndim=1]
        Initial boundary separation.
    zh, zl1, zl2 : np.ndarray[float, ndim=1]
        Starting points for high-dimensional and two low-dimensional processes.
    d : np.ndarray[float, ndim=1]
        Damping parameter for low-dimensional drift.
    g : np.ndarray[float, ndim=1]
        Inhibition parameter for low-dimensional choice process.
    t : np.ndarray[float, ndim=1]
        Non-decision time.
    deadline : np.ndarray[float, ndim=1]
        Maximum allowed decision time.
    s : np.ndarray[float, ndim=1]
        Noise standard deviation.
    delta_t : float, optional
        Time step for simulation (default is 0.001).
    max_t : float, optional
        Maximum time to simulate (default is 20).
    n_samples : int, optional
        Number of samples to generate (default is 20000).
    n_trials : int, optional
        Number of trials to simulate (default is 1).
    print_info : bool, optional
        Whether to print simulation information (default is True).
    boundary_fun : callable, optional
        Function defining the decision boundary over time.
    boundary_params : dict, optional
        Parameters for the boundary function.
    random_state : int or None, optional
        Seed for random number generator.
    return_option : str, optional
        Determines the amount of data returned ('full' or 'minimal', default is 'full').
    smooth_unif : bool, optional
        Whether to use smooth uniform distribution for certain calculations (default is False).
    **kwargs : dict
        Additional keyword arguments.

    Returns:
    --------
    dict
        A dictionary containing simulated reaction times, choices, and metadata.
        The exact contents depend on the 'return_option' parameter.

    Notes:
    ------
    This function implements a complex DDM with multiple interacting processes and flexible
    boundaries. It's designed for advanced cognitive modeling scenarios where both
    high-dimensional and low-dimensional choice processes are of interest.

    Note: n_threads parameter is accepted for API consistency but parallel execution shows
    limited speedup for MIC2 models due to bias trace array dependencies.
    """
    # Warn if n_threads > 1 (limited benefit for MIC2 models)
    if n_threads > 1:
        warnings.warn(
            "ddm_flexbound_mic2_ornstein_multinoise: n_threads > 1 has limited speedup due to "
            "bias trace dependencies. Using sequential execution.",
            UserWarning
        )

    set_seed(random_state)
    # Param views
    cdef float[:] vh_view = vh
    cdef float[:] vl1_view = vl1
    cdef float[:] vl2_view = vl2
    cdef float[:] a_view = a
    cdef float[:] zh_view = zh
    cdef float[:] zl1_view = zl1
    cdef float[:] zl2_view = zl2
    cdef float[:] d_view = d
    cdef float[:] g_view = g
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s
    # TD: Add trajectory --> same issue as with par2 model above... might need to make a separate simulator for trajectories
    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    rts_low = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    rts_high = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)

    cdef float[:, :, :] rts_view = rts
    cdef float[:, :, :] rts_high_view = rts_high
    cdef float[:, :, :] rts_low_view = rts_low
    cdef int[:, :, :] choices_view = choices

    traj = np.zeros((int(max_t / delta_t) + 1, 3), dtype = DTYPE)
    traj[:, :] = -999
    cdef float[:, :] traj_view = traj

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    #cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    # Y particle trace
    bias_trace_l1 = np.zeros(num_draws, dtype = DTYPE)
    bias_trace_l2 = np.zeros(num_draws, dtype = DTYPE)
    cdef float[:] bias_trace_l1_view = bias_trace_l1
    cdef float[:] bias_trace_l2_view = bias_trace_l2

    cdef float y_h, y_l, y_l1, y_l2
    cdef float v_l, v_l1, v_l2,
    cdef float t_h, t_l, t_l1, t_l2, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n, ix, ix1, ix2, ix_l, ix_tmp, ix1_tmp, ix2_tmp, k
    cdef Py_ssize_t m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    cdef Py_ssize_t mu = 0
    cdef float[:] uniform_values = draw_uniform(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
        compute_boundary(boundary, t_s, boundary_fun,
                        boundary_params_tmp)

        deadline_tmp = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])
        sqrt_st = delta_t_sqrt * s_view[k]
        # Loop over samples
        for n in range(n_samples):
            choices_view[n, k, 0] = 0 # reset choice
            t_h = 0 # reset time high dimension
            t_l = 0 # reset time low dimension
            t_l1 = 0 # reset time low dimension (1)
            t_l2 = 0 # reset time low dimension (2)
            ix = 0 # reset boundary index
            ix1 = 0 # reset boundary index (1)
            ix2 = 0 # reset boundary index (2)

            # Initialize walkers
            # Particle
            y_h = (-1) * boundary_view[0] + (zh_view[k] * 2 * (boundary_view[0]))
            # Relative particle position (used as resource allocator for low dim choice)
            bias_trace_l2_view[0] = ((y_h + boundary_view[0]) / (2 * boundary_view[0]))
            bias_trace_l1_view[0] = 1.0 - bias_trace_l2_view[0]

            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y_h

            # Random walks until y_h hits bound
            while (y_h >= ((-1) * boundary_view[ix])) and ((y_h <= boundary_view[ix])) and (t_h <= deadline_tmp):
                y_h += (vh_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                bias_trace_l2_view[ix] = ((y_h + boundary_view[ix]) / (2 * boundary_view[ix]))
                bias_trace_l1_view[ix] = 1.0 - bias_trace_l2_view[ix]
                t_h += delta_t
                ix += 1
                m += 1
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add 2 deterministically
            # y at lower bound --> choice_view[n, k, 0] stay the same deterministically

            # If boundary is negative (or 0) already, we flip a coin
            if boundary_view[ix] <= 0:
                if uniform_values[mu] <= 0.5:
                    choices_view[n, k, 0] += 2
            # Otherwise, apply rule from above
            elif uniform_values[mu] <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 2
            mu += 1
            if mu == num_draws:
                uniform_values = draw_uniform(num_draws)
                mu = 0

            y_l2 = (- 1) * boundary_view[0] + (zl2_view[k] * 2 * (boundary_view[0]))
            y_l1 = (- 1) * boundary_view[0] + (zl1_view[k] * 2 * (boundary_view[0]))

            if choices_view[n, k, 0] == 0:
                 # Fill bias tracea until max_rt reached
                ix1_tmp = ix + 1
                while ix1_tmp < num_draws:
                    bias_trace_l2_view[ix1_tmp] = 0.0
                    bias_trace_l1_view[ix1_tmp] = 1.0
                    ix1_tmp += 1

            else: # Store intermediate choice
                # Fill bias tracea until max_rt reached
                ix2_tmp = ix + 1
                while ix2_tmp < num_draws:
                    bias_trace_l2_view[ix2_tmp] = 1.0
                    bias_trace_l1_view[ix2_tmp] = 0.0
                    ix2_tmp += 1

            # lower level random walker (1)
            if (choices_view[n, k, 0] == 0) | ((n == 0) & (k == 0)):
                while (y_l1 >= ((-1) * boundary_view[ix1])) and (y_l1 <= boundary_view[ix1]) and (t_l1 <= deadline_tmp):
                    if (bias_trace_l1_view[ix1] < 1) and (bias_trace_l1_view[ix1] > 0):
                        # main propagation if bias_trace is between 0 and 1 (high level choice is not yet made)
                        y_l1 += (((vl1_view[k] * bias_trace_l1_view[ix1] * (1 - d_view[k])) - (g_view[k] * y_l1)) * delta_t)
                        # add gaussian displacement
                        y_l1 += (sqrt_st * gaussian_values[m]) * bias_trace_l1_view[ix1]
                    else:
                        # main propagation if bias_trace is not between 0 and 1 (high level choice is already made)
                        y_l1 += (vl1_view[k] * delta_t)
                        # add gaussian displacement
                        y_l1 += (sqrt_st * gaussian_values[m])


                    # propagate time and indices
                    t_l1 += delta_t
                    ix1 += 1
                    m += 1
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                    if n == 0:
                        if k == 0:
                            traj_view[ix1, 1] = y_l1

            # lower level random walker (2)
            if (choices_view[n, k, 0] == 2) | ((n == 0) & (k == 0)):
                while (y_l2 >= ((-1) * boundary_view[ix2])) and (y_l2 <= boundary_view[ix2]) and (t_l2 <= deadline_tmp):
                    if (bias_trace_l2_view[ix2] < 1) and (bias_trace_l2_view[ix2] > 0):
                        # main propagation if bias_trace is between 0 and 1 (high level choice is not yet made)
                        y_l2 += (((vl2_view[k] * bias_trace_l2_view[ix2] * (1 - d_view[k])) - (g_view[k] * y_l2)) * delta_t)
                        # add gaussian displacement
                        y_l2 += (sqrt_st * gaussian_values[m]) * bias_trace_l2_view[ix2]
                    else:
                        # main propagation if bias_trace is not between 0 and 1 (high level choice is already made)
                        y_l2 += (vl2_view[k] * delta_t)
                        # add gaussian displacement
                        y_l2 += (sqrt_st * gaussian_values[m])

                    # propagate time and indices
                    t_l2 += delta_t
                    ix2 += 1
                    m += 1
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                    if n == 0:
                        if k == 0:
                            traj_view[ix2, 2] = y_l2

            # Get back to single y_l and t_l
            if (choices_view[n, k, 0] == 0):
                t_l = t_l1
                y_l = y_l1
                ix_l = ix1
            else:
                t_l = t_l2
                y_l = y_l2
                ix_l = ix2

            smooth_u = compute_smooth_unif(smooth_unif, fmax(t_h, t_l), deadline_tmp, delta_t, uniform_values[mu])
            mu += 1
            if mu == num_draws:
                uniform_values = draw_uniform(num_draws)
                mu = 0

            rts_view[n, k, 0] = fmax(t_h, t_l) + t_view[k]
            rts_high_view[n, k, 0] = t_h + t_view[k]
            rts_low_view[n, k, 0] = t_l + t_view[k]

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add one deterministically
            # y at lower bound --> choice_view[n, k, 0] stays the same deterministically

            # If boundary is negative (or 0) already, we flip a coin
            if boundary_view[ix] <= 0:
                if uniform_values[mu] <= 0.5:
                    choices_view[n, k, 0] += 1
            # Otherwise apply rule from above
            elif uniform_values[mu] <= ((y_l + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 1
            mu += 1
            if mu == num_draws:
                uniform_values = draw_uniform(num_draws)
                mu = 0

            enforce_deadline(rts_view, deadline_view, n, k, 0)

    # Build minimal metadata first
    minimal_meta = build_minimal_metadata(
        simulator_name='ddm_flexbound_mic2_adj',
        possible_choices=[0, 1, 2, 3],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )

    # Extra arrays for this model
    extra_arrays_dict = {'rts_high': rts_high, 'rts_low': rts_low}

    if return_option == 'full':
        sim_config = {'delta_t': delta_t, 'max_t': max_t}
        params = {
            'vh': vh, 'vl1': vl1, 'vl2': vl2,
            'a': a, 'zh': zh, 'zl1': zl1, 'zl2': zl2,
            'd': d, 't': t, 'deadline': deadline, 's': s
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
        return build_return_dict(rts, choices, full_meta, extra_arrays=extra_arrays_dict)

    elif return_option == 'minimal':
        return build_return_dict(rts, choices, minimal_meta, extra_arrays=extra_arrays_dict)

    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# ----------------------------------------------------------------------------------------------------


# Simulate (rt, choice) tuples from: Vanilla LBA Model without ndt -----------------------------
def lba_vanilla(np.ndarray[float, ndim = 2] v,
        np.ndarray[float, ndim = 2] a,
        np.ndarray[float, ndim = 2] z,
        np.ndarray[float, ndim = 1] deadline,
        np.ndarray[float, ndim = 2] sd, # noise sigma
        np.ndarray[float, ndim = 1] t, # non-decision time
        int nact = 3,
        int n_samples = 2000,
        int n_trials = 1,
        float max_t = 20,
        int n_threads = 1,
        **kwargs
        ):
    """
    Simulate reaction times and choices from a vanilla Linear Ballistic Accumulator (LBA) model.

    Parameters:
    -----------
    v : np.ndarray[float, ndim=2]
        Drift rate for each accumulator.
    a : np.ndarray[float, ndim=2]
        Starting point of the decision boundary.
    z : np.ndarray[float, ndim=2]
        Starting point distribution.
    deadline : np.ndarray[float, ndim=1]
        Maximum allowed decision time.
    sd : np.ndarray[float, ndim=1]
        Standard deviation of the drift rate distribution.
    t : np.ndarray[float, ndim=1]
        Non-decision time.
    nact : int, optional
        Number of accumulators (default is 3).
    n_samples : int, optional
        Number of samples to generate (default is 2000).
    n_trials : int, optional
        Number of trials to simulate (default is 1).
    max_t : float, optional
        Maximum time to simulate (default is 20).
    **kwargs : dict
        Additional keyword arguments.

    Returns:
    --------
    dict
        A dictionary containing:
        - 'rts': simulated reaction times
        - 'choices': simulated choices
        - 'metadata': dictionary with model parameters and simulation details
    """

    # Param views
    cdef float[:, :] v_view = v
    cdef float[:, :] a_view = a
    cdef float[:, :] z_view = z
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline
    cdef float[:, :] sd_view = sd

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    cdef float[:, :, :] rts_view = rts

    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    cdef int[:, :, :] choices_view = choices

    cdef Py_ssize_t n, k, i

    for k in range(n_trials):

        for n in range(n_samples):
            zs = np.random.uniform(0, z_view[k], nact)

            vs = np.abs(np.random.normal(v_view[k], sd_view[k])) # np.abs() to avoid negative vs

            x_t = ([a_view[k]]*nact - zs)/vs

            choices_view[n, k, 0] = np.argmin(x_t) # store choices for sample n
            rts_view[n, k, 0] = np.min(x_t) + t_view[k]  # store reaction time for sample n

            # If the rt exceeds the deadline, set rt to -999
            enforce_deadline(rts_view, deadline_view, n, k, 0)


    v_dict = {}
    for i in range(nact):
        v_dict['v_' + str(i)] = v[:, i]

    return {'rts': rts, 'choices': choices, 'metadata': {**v_dict,
                                                         'a': a,
                                                         'z': z,
                                                         'deadline': deadline,
                                                         'sd': sd,
                                                         't': t,
                                                         'n_samples': n_samples,
                                                         'simulator' : 'lba_vanilla',
                                                         'possible_choices': list(np.arange(0, nact, 1)),
                                                         'max_t': max_t,
                                                         }}



# Simulate (rt, choice) tuples from: Collapsing bound angle LBA Model -----------------------------
def lba_angle(np.ndarray[float, ndim = 2] v,
        np.ndarray[float, ndim = 2] a,
        np.ndarray[float, ndim = 2] z,
        np.ndarray[float, ndim = 2] theta,
        np.ndarray[float, ndim = 1] deadline,
        np.ndarray[float, ndim = 2] sd, # noise sigma
        np.ndarray[float, ndim = 1] t, # non-decision time
        int nact = 3,
        int n_samples = 2000,
        int n_trials = 1,
        float max_t = 20,
        int n_threads = 1,
        **kwargs
        ):
    """
    Simulate reaction times and choices from a Linear Ballistic Accumulator (LBA) model with collapsing bounds.

    Parameters:
    -----------
    v : np.ndarray[float, ndim=2]
        Drift rate for each accumulator.
    a : np.ndarray[float, ndim=2]
        Starting point of the decision boundary.
    z : np.ndarray[float, ndim=2]
        Starting point distribution.
    theta : np.ndarray[float, ndim=2]
        Angle parameter for the collapsing bound.
    deadline : np.ndarray[float, ndim=1]
        Maximum allowed decision time.
    sd : np.ndarray[float, ndim=1]
        Standard deviation of the drift rate distribution.
    t : np.ndarray[float, ndim=1]
        Non-decision time.
    nact : int, optional
        Number of accumulators (default is 3).
    n_samples : int, optional
        Number of samples to generate (default is 2000).
    n_trials : int, optional
        Number of trials to simulate (default is 1).
    max_t : float, optional
        Maximum time to simulate (default is 20).

    Returns:
    --------
    dict
        A dictionary containing:
        - 'rts': simulated reaction times
        - 'choices': simulated choices
        - 'metadata': additional information about the simulation
    """

    # Param views
    cdef float[:, :] v_view = v
    cdef float[:, :] a_view = a
    cdef float[:, :] z_view = z
    cdef float[:, :] theta_view = theta
    cdef float[:] t_view = t

    cdef float[:] deadline_view = deadline
    cdef float[:, :] sd_view = sd

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    cdef float[:, :, :] rts_view = rts

    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    cdef int[:, :, :] choices_view = choices

    cdef Py_ssize_t n, k, i

    for k in range(n_trials):
        for n in range(n_samples):
            zs = np.random.uniform(0, z_view[k], nact)

            vs = np.abs(np.random.normal(v_view[k], sd_view[k])) # np.abs() to avoid negative vs
            x_t = ([a_view[k]]*nact - zs)/(vs + np.tan(theta_view[k, 0]))

            choices_view[n, k, 0] = np.argmin(x_t) # store choices for sample n
            rts_view[n, k, 0] = np.min(x_t) + t_view[k] # store reaction time for sample n

            # If the rt exceeds the deadline, set rt to -999
            enforce_deadline(rts_view, deadline_view, n, k, 0)

            # if np.min(x_t) <= 0:
            #     print("\n ssms sim error: ", a[k], zs, vs, np.tan(theta[k]))

    v_dict = {}
    for i in range(nact):
        v_dict['v_' + str(i)] = v[:, i]

    return {'rts': rts, 'choices': choices, 'metadata': {**v_dict,
                                                         'a': a,
                                                         'z': z,
                                                         'theta': theta,
                                                         'deadline': deadline,
                                                         'sd': sd,
                                                         't': t,
                                                         'n_samples': n_samples,
                                                         'simulator' : 'lba_angle',
                                                         'possible_choices': list(np.arange(0, nact, 1)),
                                                         'max_t': max_t,
                                                         }}


# Simulate (rt, choice) tuples from LBA piece-wise model  -----------------------------
def rlwm_lba_pw_v1(np.ndarray[float, ndim = 2] vRL,
        np.ndarray[float, ndim = 2] vWM,
        np.ndarray[float, ndim = 2] a,
        np.ndarray[float, ndim = 2] z,
        np.ndarray[float, ndim = 2] tWM,
        np.ndarray[float, ndim = 1] deadline,
        np.ndarray[float, ndim = 2] sd, # std dev
        np.ndarray[float, ndim = 1] t, # ndt is supposed to be 0 by default because of parameter identifiability issues
        int nact = 3,
        int n_samples = 2000,
        int n_trials = 1,
        float max_t = 20,
        int n_threads = 1,
        **kwargs
        ):

    # Note: LBA models have multiple accumulators. Parallel execution not yet supported.
    if n_threads > 1:
        warnings.warn(
            "rlwm_lba_pw_v1 does not yet support parallel execution. Running with n_threads=1.",
            UserWarning
        )

    # Param views
    cdef float[:, :] v_RL_view = vRL
    cdef float[:, :] v_WM_view = vWM
    cdef float[:, :] a_view = a
    cdef float[:, :] z_view = z
    cdef float[:, :] t_WM_view = tWM
    cdef float[:] t_view = t

    cdef float[:] deadline_view = deadline
    cdef float[:, :] sd_view = sd

    cdef np.ndarray[float, ndim = 1] zs
    cdef np.ndarray[double, ndim = 2] x_t_RL
    cdef np.ndarray[double, ndim = 2] x_t_WM
    cdef np.ndarray[double, ndim = 1] vs_RL
    cdef np.ndarray[double, ndim = 1] vs_WM

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    cdef float[:, :, :] rts_view = rts

    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    cdef int[:, :, :] choices_view = choices

    cdef Py_ssize_t n, k, i

    for k in range(n_trials):

        for n in range(n_samples):
            zs = np.random.uniform(0, z_view[k], nact).astype(DTYPE)

            vs_RL = np.abs(np.random.normal(v_RL_view[k], sd_view[k])) # np.abs() to avoid negative vs
            vs_WM = np.abs(np.random.normal(v_WM_view[k], sd_view[k])) # np.abs() to avoid negative vs

            x_t_RL = ([a_view[k]]*nact - zs)/vs_RL
            # x_t_WM = ([a_view[k]]*nact - zs)/vs_WM

            if np.min(x_t_RL) < t_WM_view[k]:
                x_t = x_t_RL
            else:
                x_t = t_WM_view[k] + ( [a_view[k]]*nact - zs - ([t_WM_view[k]]*nact)*vs_RL ) / ( vs_RL + vs_WM )

            choices_view[n, k, 0] = np.argmin(x_t) # store choices for sample n
            rts_view[n, k, 0] = np.min(x_t) + t_view[k] # store reaction time for sample n

            # If the rt exceeds the deadline, set rt to -999
            enforce_deadline(rts_view, deadline_view, n, k, 0)


    v_dict = {}
    for i in range(nact):
        v_dict['vRL' + str(i)] = vRL[:, i]
        v_dict['vWM' + str(i)] = vWM[:, i]

    return {'rts': rts, 'choices': choices, 'metadata': {**v_dict,
                                                         'a': a,
                                                         'z': z,
                                                         'tWM': tWM,
                                                         't': t,
                                                         'deadline': deadline,
                                                         'sd': sd,
                                                         'n_samples': n_samples,
                                                         'simulator' : 'rlwm_lba_pw_v1',
                                                         'possible_choices': list(np.arange(0, nact, 1)),
                                                         'max_t': max_t,
                                                         }}

# Simulate (rt, choice) tuples from: RLWM LBA Race Model without ndt -----------------------------
def rlwm_lba_race(np.ndarray[float, ndim = 2] vRL, # RL drift parameters (np.array expect: one column of floats)
        np.ndarray[float, ndim = 2] vWM, # WM drift parameters (np.array expect: one column of floats)
        np.ndarray[float, ndim = 2] a, # criterion height
        np.ndarray[float, ndim = 2] z, # initial bias parameters (np.array expect: one column of floats)
        np.ndarray[float, ndim = 1] deadline,
        np.ndarray[float, ndim = 2] sd, # noise sigma
        np.ndarray[float, ndim = 1] t, # non-decision time
        int nact = 3,
        int n_samples = 2000,
        int n_trials = 1,
        float max_t = 20,
        int n_threads = 1,
        **kwargs
        ):
    """
    Simulate reaction times and choices from a Reinforcement Learning Working Memory (RLWM) Linear Ballistic Accumulator (LBA) race model.

    Parameters:
    -----------
    vRL : np.ndarray[float, ndim=2]
        Drift rate for the Reinforcement Learning (RL) component.
    vWM : np.ndarray[float, ndim=2]
        Drift rate for the Working Memory (WM) component.
    a : np.ndarray[float, ndim=2]
        Decision threshold (criterion height).
    z : np.ndarray[float, ndim=2]
        Starting point distribution.
    deadline : np.ndarray[float, ndim=1]
        Maximum allowed decision time.
    sd : np.ndarray[float, ndim=1]
        Standard deviation of the drift rate distribution.
    t : np.ndarray[float, ndim=1]
        Non-decision time.
    nact : int, optional
        Number of accumulators (default is 3).
    n_samples : int, optional
        Number of samples to generate (default is 2000).
    n_trials : int, optional
        Number of trials to simulate (default is 1).
    max_t : float, optional
        Maximum time to simulate (default is 20).

    Returns:
    --------
    dict
        A dictionary containing:
        - 'rts': simulated reaction times
        - 'choices': simulated choices
        - 'metadata': additional information about the simulation
    """

    # Param views
    cdef float[:, :] v_RL_view = vRL
    cdef float[:, :] v_WM_view = vWM
    cdef float[:, :] a_view = a
    cdef float[:, :] z_view = z
    cdef float[:] t_view = t

    cdef float[:] deadline_view = deadline
    cdef float[:, :] sd_view = sd
    cdef np.ndarray[float, ndim = 1] zs
    cdef np.ndarray[double, ndim = 2] x_t_RL
    cdef np.ndarray[double, ndim = 2] x_t_WM
    cdef np.ndarray[double, ndim = 1] vs_RL
    cdef np.ndarray[double, ndim = 1] vs_WM

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    cdef float[:, :, :] rts_view = rts

    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    cdef int[:, :, :] choices_view = choices

    cdef Py_ssize_t n, k, i

    for k in range(n_trials):

        for n in range(n_samples):
            zs = np.random.uniform(0, z_view[k], nact).astype(DTYPE)

            vs_RL = np.abs(np.random.normal(v_RL_view[k], sd_view[k])) # np.abs() to avoid negative vs
            vs_WM = np.abs(np.random.normal(v_WM_view[k], sd_view[k])) # np.abs() to avoid negative vs

            x_t_RL = ([a_view[k]]*nact - zs)/vs_RL
            x_t_WM = ([a_view[k]]*nact - zs)/vs_WM

            if np.min(x_t_RL) <= np.min(x_t_WM):
                rts_view[n, k, 0] = np.min(x_t_RL) + t_view[k]  # store reaction time for sample n
                choices_view[n, k, 0] = np.argmin(x_t_RL) # store choices for sample n
            else:
                rts_view[n, k, 0] = np.min(x_t_WM) + t_view[k]  # store reaction time for sample n
                choices_view[n, k, 0] = np.argmin(x_t_WM) # store choices for sample n

            # If the rt exceeds the deadline, set rt to -999
            enforce_deadline(rts_view, deadline_view, n, k, 0)


    v_dict = {}
    for i in range(nact):
        v_dict['vRL' + str(i)] = vRL[:, i]
        v_dict['vWM' + str(i)] = vWM[:, i]

    return {'rts': rts, 'choices': choices, 'metadata': {**v_dict,
                                                         'a': a,
                                                         'z': z,
                                                         't': 0,
                                                         'deadline': deadline,
                                                         'sd': sd,
                                                         't': t,
                                                         'n_samples': n_samples,
                                                         'simulator' : 'rlwm_lba_race',
                                                         'possible_choices': list(np.arange(0, nact, 1)),
                                                         'max_t': max_t,
                                                         }}
# ----------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flexbound_mic2_unnormalized_ornstein_multinoise(np.ndarray[float, ndim = 1] vh,
                                                        np.ndarray[float, ndim = 1] vl1,
                                                        np.ndarray[float, ndim = 1] vl2,
                                                        np.ndarray[float, ndim = 1] a,
                                                        np.ndarray[float, ndim = 1] zh,
                                                        np.ndarray[float, ndim = 1] zl1,
                                                        np.ndarray[float, ndim = 1] zl2,
                                                        np.ndarray[float, ndim = 1] d, # damper (1 --> no drift on low level until high level done, 0 --> full drift on low level)
                                                        np.ndarray[float, ndim = 1] g, # inhibition parameter for the low dim choice procress while high dim is running
                                                        np.ndarray[float, ndim = 1] t,
                                                        np.ndarray[float, ndim = 1] deadline,
                                                        np.ndarray[float, ndim = 1] s, # noise sigma
                                                        float delta_t = 0.001,
                                                        float max_t = 20,
                                                        int n_samples = 20000,
                                                        int n_trials = 1,
                                                        print_info = True,
                                                        boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                                                        boundary_params = {},
                                                        random_state = None,
                                                        return_option = 'full',
                                                        smooth_unif = False,
                                                        int n_threads = 1,
                                                        **kwargs):
    """
    Simulate a Drift Diffusion Model (DDM) with flexible boundaries for a multi-level decision process.

    This function simulates a two-stage decision process where a high-dimensional choice influences
    two low-dimensional choices through a bias trace. The process incorporates an Ornstein-Uhlenbeck
    process and multiple noise parameters.

    Parameters:
    -----------
    vh, vl1, vl2 : np.ndarray[float, ndim=1]
        Drift rates for high-dimensional and two low-dimensional choices.
    a : np.ndarray[float, ndim=1]
        Initial boundary separation.
    zh, zl1, zl2 : np.ndarray[float, ndim=1]
        Starting points for high-dimensional and two low-dimensional choices.
    d : np.ndarray[float, ndim=1]
        Damping factor for drift rate on low-level choices.
    g : np.ndarray[float, ndim=1]
        Inhibition parameter for low-dimensional choices while high-dimensional is running.
    t : np.ndarray[float, ndim=1]
        Non-decision time.
    deadline : np.ndarray[float, ndim=1]
        Time limit for each trial.
    s : np.ndarray[float, ndim=1]
        Noise standard deviation.
    delta_t : float, optional
        Time step for simulation (default: 0.001).
    max_t : float, optional
        Maximum time for simulation (default: 20).
    n_samples : int, optional
        Number of samples per trial (default: 20000).
    n_trials : int, optional
        Number of trials to simulate (default: 1).
    print_info : bool, optional
        Whether to print simulation information (default: True).
    boundary_fun : callable, optional
        Function defining the decision boundary over time.
    boundary_params : dict, optional
        Parameters for the boundary function.
    random_state : int or None, optional
        Seed for random number generator (default: None).
    return_option : str, optional
        Determines the amount of data returned ('full' or 'minimal', default: 'full').
    smooth_unif : bool, optional
        If True, applies uniform smoothing to reaction times (default: False).

    Returns:
    --------
    dict
        A dictionary containing simulated reaction times, choices, and metadata.
        The exact contents depend on the 'return_option' parameter.

    Notes:
    ------
    This function implements a complex DDM with multiple interacting decision processes,
    flexible boundaries, and Ornstein-Uhlenbeck dynamics. It's particularly suited for
    modeling hierarchical decision-making scenarios.

    Note: n_threads parameter is accepted for API consistency but parallel execution shows
    limited speedup for MIC2 models due to bias trace array dependencies.
    """
    # Warn if n_threads > 1 (limited benefit for MIC2 models)
    if n_threads > 1:
        warnings.warn(
            "ddm_flexbound_mic2_unnormalized_ornstein_multinoise: n_threads > 1 has limited speedup due to "
            "bias trace dependencies. Using sequential execution.",
            UserWarning
        )

    set_seed(random_state)
    # Param views
    cdef float[:] vh_view = vh
    cdef float[:] vl1_view = vl1
    cdef float[:] vl2_view = vl2
    cdef float[:] a_view = a
    cdef float[:] zh_view = zh
    cdef float[:] zl1_view = zl1
    cdef float[:] zl2_view = zl2
    cdef float[:] d_view = d
    cdef float[:] g_view = g
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s
    # TD: Add trajectory --> same issue as with par2 model above... might need to make a separate simulator for trajectories
    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    rts_low = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    rts_high = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)

    cdef float[:, :, :] rts_view = rts
    cdef float[:, :, :] rts_high_view = rts_high
    cdef float[:, :, :] rts_low_view = rts_low
    cdef int[:, :, :] choices_view = choices

    traj = np.zeros((int(max_t / delta_t) + 1, 3), dtype = DTYPE)
    traj[:, :] = -999
    cdef float[:, :] traj_view = traj

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    #cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    # Y particle trace
    bias_trace_l1 = np.zeros(num_draws, dtype = DTYPE)
    bias_trace_l2 = np.zeros(num_draws, dtype = DTYPE)
    cdef float[:] bias_trace_l1_view = bias_trace_l1
    cdef float[:] bias_trace_l2_view = bias_trace_l2

    cdef float y_h, y_l, y_l1, y_l2
    cdef float v_l, v_l1, v_l2,
    cdef float t_h, t_l, t_l1, t_l2, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n, ix, ix1, ix2, ix_l, ix_tmp, ix1_tmp, ix2_tmp, k
    cdef Py_ssize_t m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    cdef Py_ssize_t mu = 0
    cdef float[:] uniform_values = draw_uniform(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
        compute_boundary(boundary, t_s, boundary_fun,
                        boundary_params_tmp)

        deadline_tmp = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])
        sqrt_st = delta_t_sqrt * s_view[k]
        # Loop over samples
        for n in range(n_samples):
            choices_view[n, k, 0] = 0 # reset choice
            t_h = 0 # reset time high dimension
            t_l = 0 # reset time low dimension
            t_l1 = 0 # reset time low dimension (1)
            t_l2 = 0 # reset time low dimension (2)
            ix = 0 # reset boundary index
            ix1 = 0 # reset boundary index (1)
            ix2 = 0 # reset boundary index (2)

            # Initialize walkers
            # Particle
            y_h = (-1) * boundary_view[0] + (zh_view[k] * 2 * (boundary_view[0]))
            # Relative particle position (used as resource allocator for low dim choice)
            bias_trace_l2_view[0] = ((y_h + boundary_view[0]) / (2))
            bias_trace_l1_view[0] = boundary_view[0] - bias_trace_l2_view[0]

            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y_h

            # Random walks until y_h hits bound
            while (y_h >= ((-1) * boundary_view[ix])) and ((y_h <= boundary_view[ix])) and (t_h <= deadline_tmp):
                y_h += (vh_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                bias_trace_l2_view[ix] = ((y_h + boundary_view[ix]) / (2))
                bias_trace_l1_view[ix] = boundary_view[ix] - bias_trace_l2_view[ix]
                t_h += delta_t
                ix += 1
                m += 1
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add 2 deterministically
            # y at lower bound --> choice_view[n, k, 0] stay the same deterministically

            # If boundary is negative (or 0) already, we flip a coin
            if boundary_view[ix] <= 0:
                if uniform_values[mu] <= 0.5:
                    choices_view[n, k, 0] += 2
            # Otherwise, apply rule from above
            elif uniform_values[mu] <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 2
            mu += 1
            if mu == num_draws:
                uniform_values = draw_uniform(num_draws)
                mu = 0

            y_l2 = (- 1) * boundary_view[0] + (zl2_view[k] * 2 * (boundary_view[0]))
            y_l1 = (- 1) * boundary_view[0] + (zl1_view[k] * 2 * (boundary_view[0]))

            if choices_view[n, k, 0] == 0:
                 # Fill bias trace a until max_rt reached
                ix1_tmp = ix + 1
                while ix1_tmp < num_draws:
                    bias_trace_l2_view[ix1_tmp] = 0.0
                    bias_trace_l1_view[ix1_tmp] = 1.0
                    ix1_tmp += 1

            else: # Store intermediate choice
                # Fill bias tracea until max_rt reached
                ix2_tmp = ix + 1
                while ix2_tmp < num_draws:
                    bias_trace_l2_view[ix2_tmp] = 1.0
                    bias_trace_l1_view[ix2_tmp] = 0.0
                    ix2_tmp += 1

            # lower level random walker (1)
            if (choices_view[n, k, 0] == 0) | ((n == 0) & (k == 0)):
                while (y_l1 >= ((-1) * boundary_view[ix1])) and (y_l1 <= boundary_view[ix1]) and (t_l1 <= deadline_tmp):
                    if (bias_trace_l1_view[ix1] < boundary_view[ix1]) and (bias_trace_l1_view[ix1] > 0):
                        # main propagation if bias_trace is between 0 and 1 (high level choice is not yet made)
                        y_l1 += (((vl1_view[k] * bias_trace_l1_view[ix1] * (1 - d_view[k])) - (g_view[k] * y_l1)) * delta_t)
                        # add gaussian displacement
                        y_l1 += (sqrt_st * gaussian_values[m]) * bias_trace_l1_view[ix1]
                    else:
                        # main propagation if bias_trace is not between 0 and 1 (high level choice is already made)
                        y_l1 += (vl1_view[k] * delta_t)
                        # add gaussian displacement
                        y_l1 += (sqrt_st * gaussian_values[m])

                    # propagate time and indices
                    t_l1 += delta_t
                    ix1 += 1
                    m += 1
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                    if n == 0:
                        if k == 0:
                            traj_view[ix1, 1] = y_l1

            # lower level random walker (2)
            if (choices_view[n, k, 0] == 2) | ((n == 0) & (k == 0)):
                while (y_l2 >= ((-1) * boundary_view[ix2])) and (y_l2 <= boundary_view[ix2]) and (t_l2 <= deadline_tmp):
                    if (bias_trace_l2_view[ix2] < boundary_view[ix2]) and (bias_trace_l2_view[ix2] > 0):
                        # main propagation if bias_trace is between 0 and 1 (high level choice is not yet made)
                        y_l2 += (((vl2_view[k] * bias_trace_l2_view[ix2] * (1 - d_view[k])) - (g_view[k] * y_l2)) * delta_t)
                        # add gaussian displacement
                        y_l2 += (sqrt_st * gaussian_values[m]) * bias_trace_l2_view[ix2]
                    else:
                        # main propagation if bias_trace is not between 0 and 1 (high level choice is already made)
                        y_l2 += (vl2_view[k] * delta_t)
                        # add gaussian displacement
                        y_l2 += (sqrt_st * gaussian_values[m])

                    # propagate time and indices
                    t_l2 += delta_t
                    ix2 += 1
                    m += 1
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                    if n == 0:
                        if k == 0:
                            traj_view[ix2, 2] = y_l2

            # Get back to single y_l and t_l
            if (choices_view[n, k, 0] == 0):
                t_l = t_l1
                y_l = y_l1
                ix_l = ix1
            else:
                t_l = t_l2
                y_l = y_l2
                ix_l = ix2

            smooth_u = compute_smooth_unif(smooth_unif, fmax(t_h, t_l), deadline_tmp, delta_t, uniform_values[mu])
            mu += 1
            if mu == num_draws:
                uniform_values = draw_uniform(num_draws)
                mu = 0

            rts_view[n, k, 0] = fmax(t_h, t_l) + t_view[k]
            rts_high_view[n, k, 0] = t_h + t_view[k]
            rts_low_view[n, k, 0] = t_l + t_view[k]

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add one deterministically
            # y at lower bound --> choice_view[n, k, 0] stays the same deterministically

            # If boundary is negative (or 0) already, we flip a coin
            if boundary_view[ix] <= 0:
                if uniform_values[mu] <= 0.5:
                    choices_view[n, k, 0] += 1
            # Otherwise apply rule from above
            elif uniform_values[mu] <= ((y_l + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 1
            mu += 1
            if mu == num_draws:
                uniform_values = draw_uniform(num_draws)
                mu = 0

            enforce_deadline(rts_view, deadline_view, n, k, 0)

    # Build minimal metadata first
    minimal_meta = build_minimal_metadata(
        simulator_name='ddm_flexbound_mic2_adj',
        possible_choices=[0, 1, 2, 3],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )

    # Extra arrays for this model
    extra_arrays_dict = {'rts_high': rts_high, 'rts_low': rts_low}

    if return_option == 'full':
        sim_config = {'delta_t': delta_t, 'max_t': max_t}
        params = {
            'vh': vh, 'vl1': vl1, 'vl2': vl2,
            'a': a, 'zh': zh, 'zl1': zl1, 'zl2': zl2,
            'd': d, 't': t, 'deadline': deadline, 's': s
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
        return build_return_dict(rts, choices, full_meta, extra_arrays=extra_arrays_dict)

    elif return_option == 'minimal':
        return build_return_dict(rts, choices, minimal_meta, extra_arrays=extra_arrays_dict)

    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# ----------------------------------------------------------------------------------------------------


# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
