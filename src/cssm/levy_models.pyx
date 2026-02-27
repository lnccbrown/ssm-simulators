# Globaly settings for cython
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: initializedcheck=False

"""
Levy Flight Models

This module contains simulator functions for Levy flight diffusion models,
which generalize the Brownian motion assumption of standard diffusion models
by allowing for heavy-tailed jump distributions.
"""

import cython
from libc.math cimport log, sqrt, pow, fmax, fmin, tan, sin, cos
from cython.parallel cimport prange, parallel, threadid
from libc.stdint cimport uint64_t

import numpy as np
cimport numpy as np

# Import utility functions from the _utils module
from cssm._utils import (
    set_seed,
    draw_uniform,
    draw_random_stable,
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

include "_rng_wrappers.pxi"
include "_constants.pxi"

# Import OpenMP status check
from cssm._openmp_status import check_parallel_request

DTYPE = np.float32

def levy_flexbound(np.ndarray[float, ndim = 1] v,
                   np.ndarray[float, ndim = 1] z,
                   np.ndarray[float, ndim = 1] alpha,
                   np.ndarray[float, ndim = 1] t,
                   np.ndarray[float, ndim = 1] deadline,
                   np.ndarray[float, ndim = 1] s, # noise sigma
                   float delta_t = 0.001,
                   float max_t = 20,
                   int n_samples = 20000,
                   int n_trials = 1,
                   boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                   boundary_params = {},
                   random_state = None,
                   return_option = 'full',
                   smooth_unif = False,
                   int n_threads = 1,
                   **kwargs):
    """
    Simulate reaction times and choices from a Levy Flight model with flexible boundaries.

    Args:
        v (np.ndarray): Drift rate for each trial.
        z (np.ndarray): Starting point (between 0 and 1) for each trial.
        alpha (np.ndarray): Stability parameter for each trial (0 < alpha <= 2).
        t (np.ndarray): Non-decision time for each trial.
        deadline (np.ndarray): Maximum reaction time allowed for each trial.
        s (np.ndarray): Noise scale parameter for each trial.
        delta_t (float): Time step size for simulation (default: 0.001).
        max_t (float): Maximum time for simulation (default: 20).
        n_samples (int): Number of samples to simulate per trial (default: 20000).
        n_trials (int): Number of trials to simulate (default: 1).
        boundary_fun (callable): Function defining the shape of the boundary over time.
        boundary_params (dict): Parameters for the boundary function.
        random_state (int or None): Seed for random number generator (default: None).
        return_option (str): 'full' for complete output, 'minimal' for basic output (default: 'full').
        smooth_unif (bool): Whether to apply uniform smoothing to reaction times (default: False).
        n_threads (int): Number of threads for parallel execution (default: 1).
            If > 1 and OpenMP/GSL are available, uses parallel execution with GSL's
            validated Levy distribution sampler. Maximum supported: 256 threads.
        **kwargs: Additional keyword arguments.

    Returns:
        dict: A dictionary containing simulated reaction times, choices, and metadata.
              The exact contents depend on the return_option.

    Raises:
        ValueError: If return_option is neither 'full' nor 'minimal'.
        ValueError: If n_threads exceeds the maximum supported (256).
    """

    # Validate and clamp n_threads (handles <=0, >MAX_THREADS, missing OpenMP/GSL)
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
    t_s = setup['t_s']
    cdef int num_draws = setup['num_draws']
    cdef float delta_t_sqrt = setup['delta_t_sqrt']
    cdef float[:] uniform_values = setup['uniform_values']
    cdef int num_steps = int((max_t / delta_t) + 1)
    cdef float c_max_t = max_t

    # Param views
    cdef float[:] v_view  = v
    cdef float[:] z_view = z
    cdef float[:] alpha_view = alpha
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s

    # Boundary storage for the upper bound
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef float delta_t_alpha
    cdef float y, t_particle, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0
    cdef float[:] alpha_stable_values
    cdef Py_ssize_t mu = 0

    # Variables for parallel execution - per-thread RNG states
    cdef float[:, :] boundaries_all
    cdef float[:] deadlines_tmp_all
    cdef float[:] delta_t_alpha_all
    cdef RngState[MAX_THREADS] rng_states  # Per-thread RNG states
    cdef uint64_t combined_seed
    cdef float z_k, v_k, alpha_k, t_k, s_k, deadline_k, delta_t_alpha_k
    cdef float bound_val, neg_bound_val, noise
    cdef int choice
    cdef int tid  # Thread ID
    cdef int i_thread  # Loop variable for alloc/free

    # Flattened parallelization variables
    cdef Py_ssize_t flat_idx
    cdef Py_ssize_t total_iterations
    cdef int c_n_samples = n_samples
    cdef int c_n_threads = n_threads

    # =========================================================================
    # SEQUENTIAL PATH (n_threads == 1): Original algorithm
    # =========================================================================
    if n_threads == 1:
        alpha_stable_values = draw_random_stable(num_draws, alpha_view[0])

        for k in range(n_trials):
            # Scale for a discretised alpha-stable increment over delta_t:
            # by the self-similarity property X(delta_t) ~ S_alpha(s * delta_t^(1/alpha)).
            # Ref: Samorodnitsky & Taqqu (1994), "Stable Non-Gaussian Random Processes", ch. 1.
            # Same formula is reused in the parallel precomputation block below.
            delta_t_alpha = s_view[k] * pow(delta_t, 1.0 / alpha_view[k])
            boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

            # Precompute boundary evaluations
            compute_boundary(boundary,
                             t_s,
                             boundary_fun,
                             boundary_params_tmp
                             )
            deadline_tmp = compute_deadline_tmp(max_t,
                                                deadline_view[k],
                                                t_view[k]
                                                )

            # Loop over samples
            for n in range(n_samples):
                y = (-1) * boundary_view[0] + (z_view[k] * 2 * (boundary_view[0]))  # reset starting position
                t_particle = 0.0 # reset time
                ix = 0 # reset boundary index
                if n == 0:
                    if k == 0:
                        traj_view[0, 0] = y

                # Random walker
                while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t_particle <= deadline_tmp:
                    y += (v_view[k] * delta_t) + (delta_t_alpha * alpha_stable_values[m])
                    t_particle += delta_t
                    ix += 1
                    m += 1
                    if n == 0:
                        if k == 0:
                            traj_view[ix, 0] = y
                    if m == num_draws:
                        alpha_stable_values = draw_random_stable(num_draws, alpha_view[k])
                        m = 0

                smooth_u = compute_smooth_unif(smooth_unif,
                                               t_particle,
                                               deadline_tmp,
                                               delta_t,
                                               uniform_values[mu]
                                               )
                mu += 1
                if mu == num_draws:
                    uniform_values = draw_uniform(num_draws)
                    mu = 0

                rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u # Store rt
                choices_view[n, k, 0] = sign(y) # Store choice
                enforce_deadline(rts_view,
                                 deadline_view,
                                 n,
                                 k,
                                 0
                                )

    # =========================================================================
    # PARALLEL PATH (n_threads > 1): FLATTENED OpenMP parallelization with C RNG
    # Parallelizes over (n_trials × n_samples) for optimal efficiency
    # =========================================================================
    else:
        # Precompute ALL trial data outside nogil
        boundaries_all_np = np.zeros((n_trials, num_steps), dtype=DTYPE)
        deadlines_tmp_all_np = np.zeros(n_trials, dtype=DTYPE)
        delta_t_alpha_all_np = np.zeros(n_trials, dtype=DTYPE)

        for k in range(n_trials):
            boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
            compute_boundary(boundary, t_s, boundary_fun, boundary_params_tmp)
            boundaries_all_np[k, :] = boundary
            deadlines_tmp_all_np[k] = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])
            # Same self-similarity scaling as in the sequential path above.
            delta_t_alpha_all_np[k] = s_view[k] * pow(delta_t, 1.0 / alpha_view[k])

        boundaries_all = boundaries_all_np
        deadlines_tmp_all = deadlines_tmp_all_np
        delta_t_alpha_all = delta_t_alpha_all_np

        # Total iterations = n_trials × n_samples
        total_iterations = <Py_ssize_t>n_trials * <Py_ssize_t>n_samples

        # Allocate per-thread GSL RNGs BEFORE parallel block
        for i_thread in range(c_n_threads):
            rng_alloc(&rng_states[i_thread])

        with nogil, parallel(num_threads=n_threads):
            for flat_idx in prange(total_iterations, schedule='dynamic'):
                # Compute (k, n) from flat index
                k = flat_idx // c_n_samples
                n = flat_idx % c_n_samples

                # Get this thread's dedicated RNG state
                tid = threadid()

                z_k = z_view[k]
                v_k = v_view[k]
                alpha_k = alpha_view[k]
                t_k = t_view[k]
                deadline_k = deadline_view[k]
                delta_t_alpha_k = delta_t_alpha_all[k]
                deadline_tmp = deadlines_tmp_all[k]

                # Re-seed per-thread RNG (no allocation, safe in parallel)
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

                    # Generate alpha-stable noise using GSL's Levy sampler
                    noise = rng_levy_f32(&rng_states[tid], alpha_k)
                    y = y + (v_k * delta_t) + (delta_t_alpha_k * noise)
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
        simulator_name='levy_flexbound',
        possible_choices=[-1, 1],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )

    if return_option == 'full':
        sim_config = {'delta_t': delta_t, 'max_t': max_t, 'n_threads': n_threads}
        params = {
            'v': v, 'z': z,
            't': t, 'alpha': alpha, 's': s,
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
