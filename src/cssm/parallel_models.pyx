# Globaly settings for cython
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: initializedcheck=False

"""
Parallel Sampling Models

This module contains simulator functions for parallel decision models.
These models involve simultaneous (parallel) accumulation of evidence across
multiple dimensions that combine to form a single decision variable.
"""

import cython
from libc.math cimport sqrt, log, fmax
from libc.stdint cimport uint64_t

import numpy as np
cimport numpy as np

# OpenMP imports
from cython.parallel cimport prange, parallel, threadid

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

include "_rng_wrappers.pxi"

DTYPE = np.float32

# Include shared constants (MAX_THREADS, etc.)
include "_constants.pxi"

# Parallel Models ------------------------------------

def ddm_flexbound_par2(np.ndarray[float, ndim = 1] vh,
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
    Simulate a parallel diffusion decision model with flexible boundaries.

    This function simulates a decision process where three independent walkers
    (high-dimensional and two low-dimensional) evolve in parallel. The final
    decision combines outcomes from all three processes.

    Parameters:
    -----------
    vh, vl1, vl2 : np.ndarray
        Drift rates for high-dimensional and two low-dimensional choices.
    zh, zl1, zl2 : np.ndarray
        Starting points for high-dimensional and two low-dimensional choices.
    t : np.ndarray
        Non-decision time.
    deadline : np.ndarray
        Time limit for each trial.
    s : np.ndarray
        Noise standard deviation.
    delta_t : float, optional
        Size of time steps in simulation. Default is 0.001.
    max_t : float, optional
        Maximum time for each trial. Default is 20.
    n_samples : int, optional
        Number of simulations per trial. Default is 20000.
    n_trials : int, optional
        Number of trials to simulate. Default is 1.
    print_info : bool, optional
        Whether to print information during simulation. Default is True.
    boundary_fun : callable, optional
        Function defining the decision boundary over time.
    boundary_params : dict, optional
        Additional parameters for the boundary function.
    random_state : int or None, optional
        Seed for random number generator. Default is None.
    return_option : str, optional
        Determines the content of the returned dictionary. Can be 'full' or 'minimal'. Default is 'full'.
    smooth_unif : bool, optional
        If True, adds uniform noise to simulate continuous time. Default is False.
    n_threads : int, optional
        Number of threads for parallel execution. Default is 1.

    Returns:
    --------
    dict
        A dictionary containing simulation results. The exact contents depend on the return_option.
        'full' returns all simulation data and parameters, while 'minimal' returns only essential outputs.
    """
    # Check OpenMP availability for parallel execution
    if n_threads > 1:
        from cssm._openmp_status import check_parallel_request
        n_threads = check_parallel_request(n_threads)

    # Sequential path (n_threads=1)
    if n_threads == 1:
        return _ddm_flexbound_par2_sequential(
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

    # Trajectory storage - disabled in parallel mode (would require first sample only)
    traj = np.zeros((int(max_t / delta_t) + 1, 3), dtype=DTYPE)
    traj[:, :] = -999
    cdef float[:, :] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype=DTYPE)
    rts_high = np.zeros((n_samples, n_trials, 1), dtype=DTYPE)
    rts_low = np.zeros((n_samples, n_trials, 1), dtype=DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype=np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef float[:, :, :] rts_high_view = rts_high
    cdef float[:, :, :] rts_low_view = rts_low
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t)
    cdef int num_steps = int((max_t / delta_t) + 1)
    cdef float c_max_t = max_t
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
    cdef Py_ssize_t flat_idx
    cdef int ix, ix1, ix2
    cdef float y_h, y_l1, y_l2, t_h, t_l1, t_l2, t_particle
    cdef float deadline_tmp_k, sqrt_st_k
    cdef int choice_val
    cdef float bound_h, bound_l1, bound_l2, noise

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

            # Initialize all three walkers
            t_h = 0.0
            t_l1 = 0.0
            t_l2 = 0.0
            ix = 0
            ix1 = 0
            ix2 = 0
            choice_val = 0

            # High-dimensional walker initialization
            bound_h = boundaries_view[k, 0]
            y_h = (-1.0) * bound_h + (zh_view[k] * 2.0 * bound_h)

            # Simulate high-dimensional walker
            while True:
                bound_h = boundaries_view[k, ix]
                if y_h < (-1.0) * bound_h or y_h > bound_h or t_h > deadline_tmp_k:
                    break
                noise = rng_gaussian_f32(&rng_states[tid])
                y_h = y_h + (vh_view[k] * delta_t) + (sqrt_st_k * noise)
                t_h = t_h + delta_t
                ix = ix + 1
                if ix >= num_steps:
                    break

            # Determine high-dimensional choice
            bound_h = boundaries_view[k, ix] if ix < num_steps else 0.0
            if bound_h <= 0.0:
                if rng_uniform_f32(&rng_states[tid]) <= 0.5:
                    choice_val = 2
            elif rng_uniform_f32(&rng_states[tid]) <= ((y_h + bound_h) / (2.0 * bound_h)):
                choice_val = 2

            # OPTIMIZATION: Only simulate the RELEVANT low-dimensional walker
            # based on high-dim choice (saves ~33% computation)
            # choice_val mapping: 0=high0_low0, 1=high0_low1, 2=high1_low0, 3=high1_low1
            if choice_val == 0:
                # High-dim chose lower bound -> simulate low-dim walker 1
                y_l1 = (-1.0) * bound_h + (zl1_view[k] * 2.0 * bound_h)
                while True:
                    bound_l1 = boundaries_view[k, ix1]
                    if y_l1 < (-1.0) * bound_l1 or y_l1 > bound_l1 or t_l1 > deadline_tmp_k:
                        break
                    noise = rng_gaussian_f32(&rng_states[tid])
                    y_l1 = y_l1 + (vl1_view[k] * delta_t) + (sqrt_st_k * noise)
                    t_l1 = t_l1 + delta_t
                    ix1 = ix1 + 1
                    if ix1 >= num_steps:
                        break

                t_particle = fmax(t_h, t_l1)
                rts_low_view[n, k, 0] = t_l1 + t_view[k]
                # Low-dim choice based on y_l1
                bound_l1 = boundaries_view[k, ix1] if ix1 < num_steps else 0.0
                if bound_l1 <= 0.0:
                    if rng_uniform_f32(&rng_states[tid]) <= 0.5:
                        choice_val = 1
                elif rng_uniform_f32(&rng_states[tid]) <= ((y_l1 + bound_l1) / (2.0 * bound_l1)):
                    choice_val = 1
                # else choice_val remains 0
            else:
                # High-dim chose upper bound (choice_val == 2) -> simulate low-dim walker 2
                y_l2 = (-1.0) * bound_h + (zl2_view[k] * 2.0 * bound_h)
                while True:
                    bound_l2 = boundaries_view[k, ix2]
                    if y_l2 < (-1.0) * bound_l2 or y_l2 > bound_l2 or t_l2 > deadline_tmp_k:
                        break
                    noise = rng_gaussian_f32(&rng_states[tid])
                    y_l2 = y_l2 + (vl2_view[k] * delta_t) + (sqrt_st_k * noise)
                    t_l2 = t_l2 + delta_t
                    ix2 = ix2 + 1
                    if ix2 >= num_steps:
                        break

                t_particle = fmax(t_h, t_l2)
                rts_low_view[n, k, 0] = t_l2 + t_view[k]
                # Low-dim choice based on y_l2: result will be 2 or 3
                bound_l2 = boundaries_view[k, ix2] if ix2 < num_steps else 0.0
                if bound_l2 <= 0.0:
                    if rng_uniform_f32(&rng_states[tid]) <= 0.5:
                        choice_val = 3  # high=1, low=1
                    # else choice_val stays 2
                elif rng_uniform_f32(&rng_states[tid]) <= ((y_l2 + bound_l2) / (2.0 * bound_l2)):
                    choice_val = 3  # high=1, low=1
                # else choice_val stays 2

            rts_view[n, k, 0] = t_particle + t_view[k]
            rts_high_view[n, k, 0] = t_h + t_view[k]
            choices_view[n, k, 0] = choice_val

            # Enforce deadline
            if rts_view[n, k, 0] >= deadline_view[k]:
                rts_view[n, k, 0] = -999.0

    # Free per-thread GSL RNGs AFTER parallel block
    for i_thread in range(c_n_threads):
        rng_free(&rng_states[i_thread])

    # Build minimal metadata first
    minimal_meta = build_minimal_metadata(
        simulator_name='ddm_flexbound_par2',
        possible_choices=[0, 1, 2, 3],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )

    # Extra arrays for this model
    extra_arrays_dict = {'rts_low': rts_low, 'rts_high': rts_high}

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
        return build_return_dict(rts, choices, full_meta, extra_arrays=extra_arrays_dict)

    elif return_option == 'minimal':
        return build_return_dict(rts, choices, minimal_meta, extra_arrays=extra_arrays_dict)

    else:
        raise ValueError('return_option must be either "full" or "minimal"')


def _ddm_flexbound_par2_sequential(
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
    """Sequential implementation of ddm_flexbound_par2 (original code path)."""

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

    # TD: Add trajectory --> Tricky here because the simulator is optimized to include only two instead of three particles (high dimension choice determines which low dimension choice will matter for ultimate choice)
    # TD: Add Trajectory
    traj = np.zeros((int(max_t / delta_t) + 1, 3), dtype = DTYPE)
    traj[:, :] = -999
    cdef float[:, :] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    rts_high = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    rts_low = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef float[:, :, :] rts_high_view = rts_high
    cdef float[:, :, :] rts_low_view = rts_low
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    #cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)

    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef float y_h, y_l, y_l1, y_l2, v_l, v_l1, v_l2, t_h, t_l, t_l1, t_l2, smooth_u, deadline_tmp, sqrt_st
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
            t_h = 0.0 # reset time high dimension
            t_l1 = 0.0 # reset time low dimension (1)
            t_l2 = 0.0 # reset time low dimension (2)
            t_l = 0.0 # reset time low dimension (1 or 2)
            ix = 0 # reset boundary index

            # Initialize walkers
            y_h = (-1) * boundary_view[0] + (zh_view[k] * 2 * (boundary_view[0]))

            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y_h

            # Random walks until y_h hits bound
            while (y_h >= (-1) * boundary_view[ix]) and (y_h <= boundary_view[ix]) and (t_h <= deadline_tmp):
                y_h += (vh_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
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
            # y at upper bound --> choices_view[n, k, 0] add 2 deterministically (correct)
            # y at lower bound --> choice_view[n, k, 0] stay the same deterministically (mistake)

            # if boundary is negative (or 0) already, we flip a coin
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

            # Initialize lower level walkers
            y_l1 = (-1) * boundary_view[0] + (zl1_view[k] * 2 * (boundary_view[0]))
            y_l2 = (-1) * boundary_view[0] + (zl2_view[k] * 2 * (boundary_view[0]))

            # Random walker lower level (1)
            if (choices_view[n, k, 0] == 0) | ((n == 0) & (k == 0)):
                ix1 = 0
                while (y_l1 >= (-1) * boundary_view[ix1]) and (y_l1 <= boundary_view[ix1]) and (t_l1 <= deadline_tmp):
                    y_l1 += (vl1_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                    t_l1 += delta_t
                    ix1 += 1
                    m += 1
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                    if n == 0:
                        if k == 0:
                            traj_view[ix1, 1] = y_l1

            # Random walker lower level (2)
            if (choices_view[n, k, 0] == 2) | ((n == 0) & (k == 0)):
                ix2 = 0
                while (y_l2 >= (-1) * boundary_view[ix2]) and (y_l2 <= boundary_view[ix2]) and (t_l2 <= deadline_tmp):
                    y_l2 += (vl2_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                    t_l2 += delta_t
                    ix2 += 1
                    m += 1
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                    if n == 0:
                        if k == 0:
                            traj_view[ix2, 2] = y_l2

            # Consider only relevant lower-dim walker for final rt
            if (choices_view[n, k, 0] == 0):
                t_l = t_l1
                y_l = y_l1
                ix = ix1
            else:
                t_l = t_l2
                y_l = y_l2
                ix = ix2

            t_particle = fmax(t_h, t_l)  # Use max time for parallel model
            smooth_u = compute_smooth_unif(smooth_unif, t_particle, deadline_tmp, delta_t, uniform_values[mu])
            mu += 1
            if mu == num_draws:
                uniform_values = draw_uniform(num_draws)
                mu = 0

            rts_view[n, k, 0] = fmax(t_h, t_l) + t_view[k] + smooth_u
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
        simulator_name='ddm_flexbound_par2',
        possible_choices=[0, 1, 2, 3],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )

    # Extra arrays for this model
    extra_arrays_dict = {'rts_low': rts_low, 'rts_high': rts_high}

    if return_option == 'full':
        sim_config = {'delta_t': delta_t, 'max_t': max_t, 'n_threads': 1}
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
        return build_return_dict(rts, choices, full_meta, extra_arrays=extra_arrays_dict)

    elif return_option == 'minimal':
        return build_return_dict(rts, choices, minimal_meta, extra_arrays=extra_arrays_dict)

    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# -----------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
