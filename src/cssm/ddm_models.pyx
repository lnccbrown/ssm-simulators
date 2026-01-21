# Globaly settings for cython
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: initializedcheck=False

"""
Drift Diffusion Model (DDM) Simulators

This module contains simulator functions for various drift diffusion models,
the most widely used sequential sampling models in cognitive psychology and neuroscience.
These models simulate the accumulation of noisy evidence toward decision boundaries.

Parallelization (n_threads parameter):
- n_threads=1: Sequential execution (default, preserves original behavior including trajectory)
- n_threads>1: OpenMP parallel execution (requires OpenMP, no trajectory recording)
"""

import cython
from cython.parallel import prange, parallel
from libc.math cimport sqrt, log, exp, fmax, fmin
from libc.stdint cimport uint64_t

import numpy as np
cimport numpy as np

# Import utility functions from the _utils module
from cssm._utils import (
    set_seed,
    random_uniform,
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

# Import OpenMP status checker for graceful degradation
from cssm._openmp_status import check_parallel_request

# Import thread ID for per-thread RNG
from cython.parallel cimport threadid

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

# Type alias
ctypedef ssms_rng_state RngState

# Include shared constants (MAX_THREADS, etc.)
include "_constants.pxi"

# Wrapper functions
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

# =============================================================================

DTYPE = np.float32

# Simulate (rt, choice) tuples from: FULL DDM (HDDM BASE) ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def full_ddm_hddm_base(np.ndarray[float, ndim = 1] v, # = 0,
                       np.ndarray[float, ndim = 1] a, # = 1.0,
                       np.ndarray[float, ndim = 1] z, # = 0.5,
                       np.ndarray[float, ndim = 1] t, # = 0.0,
                       np.ndarray[float, ndim = 1] sz, # = 0.05,
                       np.ndarray[float, ndim = 1] sv, # = 0.1,
                       np.ndarray[float, ndim = 1] st, # = 0.0,
                       np.ndarray[float, ndim = 1] deadline, # = 0.0,
                       np.ndarray[float, ndim = 1] s, # = 1,
                       float delta_t = 0.001,
                       float max_t = 20,
                       int n_samples = 20000,
                       int n_trials = 1,
                       random_state = None,
                       smooth_unif  = False,
                       return_option = 'full', # 'full' or 'minimal'
                       int n_threads = 1,
                       **kwargs,
                       ):
    """
    Simulate reaction times and choices from a full drift diffusion model with flexible bounds.

    Args:
        v (np.ndarray): Drift rate for each trial.
        a (np.ndarray): Boundary separation (threshold) for each trial.
        z (np.ndarray): Starting point bias for each trial (between 0 and 1).
        t (np.ndarray): Non-decision time for each trial.
        sz (np.ndarray): Variability in starting point for each trial.
        sv (np.ndarray): Variability in drift rate for each trial.
        st (np.ndarray): Variability in non-decision time for each trial.
        deadline (np.ndarray): Maximum allowed reaction time for each trial.
        s (np.ndarray): Diffusion coefficient (noise) for each trial.
        delta_t (float): Time step for simulation.
        max_t (float): Maximum time for simulation.
        n_samples (int): Number of samples to simulate per trial.
        n_trials (int): Number of trials to simulate.
        random_state (int or None): Seed for random number generator.
        smooth_unif (bool): Whether to apply uniform smoothing to reaction times.
        return_option (str): 'full' for complete output, 'minimal' for basic output.
        **kwargs: Additional keyword arguments.

    Returns:
        dict: A dictionary containing simulated reaction times, choices, and metadata.
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
    t_s = setup['t_s']
    cdef int num_draws = setup['num_draws']
    cdef float delta_t_sqrt = setup['delta_t_sqrt']
    cdef int num_steps = int((max_t / delta_t) + 1)
    cdef float c_max_t = max_t

    # Param views
    cdef float[:] v_view = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] sz_view = sz
    cdef float[:] sv_view = sv
    cdef float[:] st_view = st
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s

    cdef float y, t_particle, t_tmp, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0
    cdef float drift_increment = 0.0

    # Variables for parallel execution - per-thread RNG states
    cdef RngState[MAX_THREADS] rng_states
    cdef int tid
    cdef uint64_t combined_seed
    cdef float z_k, v_k, a_k, t_k, s_k, sz_k, sv_k, st_k, sqrt_st_k, deadline_k
    cdef float noise, y_disp, v_disp, t_disp
    cdef int choice
    cdef int i_thread

    # Flattened parallelization variables
    cdef Py_ssize_t flat_idx
    cdef Py_ssize_t total_iterations
    cdef int c_n_samples = n_samples
    cdef int c_n_threads = n_threads

    # =========================================================================
    # SEQUENTIAL PATH (n_threads == 1): Original algorithm
    # =========================================================================
    if n_threads == 1:
        for k in range(n_trials):
            # Loop over samples
            deadline_tmp = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])
            sqrt_st = delta_t_sqrt * s_view[k]
            for n in range(n_samples):
                # initialize starting point
                y = (z_view[k] * (a_view[k]))  # reset starting position

                # get drift by random displacement of v
                drift_increment = (v_view[k] + sv_view[k] * gaussian_values[m]) * delta_t
                t_tmp = t_view[k] + (2 * (random_uniform() - 0.5) * st_view[k])

                # apply uniform displacement on y
                y += 2 * (random_uniform() - 0.5) * sz_view[k]

                # increment m appropriately
                m += 1
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

                t_particle = 0.0 # reset time
                ix = 0 # reset boundary index

                if n == 0:
                    if k == 0:
                        traj_view[0, 0] = y

                # Random walker
                while y >= 0 and y <= a_view[k] and t_particle <= deadline_tmp:
                    y += drift_increment + (sqrt_st * gaussian_values[m])
                    t_particle += delta_t
                    ix += 1
                    m += 1

                    if n == 0:
                        if k == 0:
                            traj_view[ix, 0] = y
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                # Apply smoothing with uniform if desired
                smooth_u = compute_smooth_unif(smooth_unif, t_particle, deadline_tmp, delta_t)

                rts_view[n, k, 0] = t_particle + t_tmp + smooth_u # Store rt

                if y < 0:
                    choices_view[n, k, 0] = 0 # Store choice
                else:
                    choices_view[n, k, 0] = 1

                # If the rt exceeds the deadline, set rt to -999 and choice to -1
                enforce_deadline(rts_view, deadline_view, n, k, 0)

    # =========================================================================
    # PARALLEL PATH (n_threads > 1): FLATTENED OpenMP parallelization with GSL RNG
    # Uses GSL's validated Ziggurat implementation for Gaussian sampling.
    # Per-thread RNG states are allocated before parallel block and freed after.
    # =========================================================================
    else:
        total_iterations = <Py_ssize_t>n_trials * <Py_ssize_t>n_samples

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

                # Cache trial parameters
                z_k = z_view[k]
                v_k = v_view[k]
                a_k = a_view[k]
                t_k = t_view[k]
                s_k = s_view[k]
                sz_k = sz_view[k]
                sv_k = sv_view[k]
                st_k = st_view[k]
                sqrt_st_k = delta_t_sqrt * s_k
                deadline_k = deadline_view[k]

                # Compute deadline
                deadline_tmp = fmin(c_max_t, deadline_k - t_k)
                if deadline_tmp < 0:
                    deadline_tmp = c_max_t

                # Re-seed per-thread RNG with unique seed for this (trial, sample)
                combined_seed = rng_mix_seed(seed, <uint64_t>k, <uint64_t>n)
                rng_seed(&rng_states[tid], combined_seed)

                # Generate variability using GSL Ziggurat
                v_disp = sv_k * rng_gaussian_f32(&rng_states[tid])

                # Uniform for starting point and t variability: 2 * (U - 0.5) = range [-1, 1]
                y_disp = 2.0 * (<float>ssms_uniform(&rng_states[tid]) - 0.5) * sz_k
                t_disp = 2.0 * (<float>ssms_uniform(&rng_states[tid]) - 0.5) * st_k

                # Starting point: z * a + displacement
                y = z_k * a_k + y_disp

                drift_increment = (v_k + v_disp) * delta_t
                t_tmp = t_k + t_disp

                t_particle = 0.0
                ix = 0

                # Random walker with constant boundaries [0, a]
                while y >= 0 and y <= a_k and t_particle <= deadline_tmp:
                    noise = rng_gaussian_f32(&rng_states[tid])
                    y = y + drift_increment + sqrt_st_k * noise
                    t_particle = t_particle + delta_t
                    ix = ix + 1

                    if ix >= num_steps:
                        break

                rts_view[n, k, 0] = t_particle + t_tmp

                if y < 0:
                    choice = 0
                else:
                    choice = 1
                choices_view[n, k, 0] = choice

                if rts_view[n, k, 0] >= deadline_k or deadline_k <= 0:
                    rts_view[n, k, 0] = -999.0

        # Free per-thread GSL RNGs AFTER parallel block
        for i_thread in range(c_n_threads):
            rng_free(&rng_states[i_thread])

    # Build minimal metadata first
    minimal_meta = build_minimal_metadata(
        simulator_name='full_ddm_hddm_base',
        possible_choices=[0, 1],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name='constant'
    )

    if return_option == 'full':
        # Augment minimal with full details
        sim_config = {'delta_t': delta_t, 'max_t': max_t}
        params = {
            'v': v, 'a': a, 'z': z, 't': t,
            'sz': sz, 'sv': sv, 'st': st,
            'deadline': deadline, 's': s
        }
        full_meta = build_full_metadata(
            minimal_metadata=minimal_meta,
            params=params,
            sim_config=sim_config,
            traj=traj
        )
        return build_return_dict(rts, choices, full_meta)

    elif return_option == 'minimal':
        return build_return_dict(rts, choices, minimal_meta)

    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# -------------------------------------------------------------------------------------------------


# Simulate (rt, choice) tuples from: SIMPLE DDM -----------------------------------------------
# Simplest algorithm
#@cython.boundscheck(False)
#@cython.wraparound(False)
def ddm(np.ndarray[float, ndim = 1] v, # drift by timestep 'delta_t'
        np.ndarray[float, ndim = 1] a, # boundary separation
        np.ndarray[float, ndim = 1] z,  # between 0 and 1
        np.ndarray[float, ndim = 1] t, # non-decision time
        np.ndarray[float, ndim = 1] deadline, # maximum rt allowed
        np.ndarray[float, ndim = 1] s, # noise sigma
        max_t = 20, # maximum rt allowed
        float delta_t = 0.001, # timesteps fraction of seconds
        int n_samples = 20000, # number of samples considered
        int n_trials = 10,
        random_state = None,
        return_option = 'full', # 'full' or 'minimal'
        smooth_unif  = False,
        int n_threads = 1,
        **kwargs):
    """
    Simulate reaction times and choices from a simple drift diffusion model (DDM).

    Args:
        v (np.ndarray): Drift rate for each trial.
        a (np.ndarray): Boundary separation for each trial.
        z (np.ndarray): Starting point (between 0 and 1) for each trial.
        t (np.ndarray): Non-decision time for each trial.
        deadline (np.ndarray): Maximum reaction time allowed for each trial.
        s (np.ndarray): Noise standard deviation for each trial.
        max_t (float): Maximum simulation time (default: 20).
        delta_t (float): Time step size (default: 0.001).
        n_samples (int): Number of samples per trial (default: 20000).
        n_trials (int): Number of trials to simulate (default: 10).
        random_state (int or None): Seed for random number generator (default: None).
        return_option (str): 'full' or 'minimal' return format (default: 'full').
        smooth_unif (bool): Whether to apply uniform smoothing to reaction times (default: False).
        n_threads (int): Number of threads for parallel execution (default: 1).
        **kwargs: Additional keyword arguments.

    Returns:
        dict: A dictionary containing simulated reaction times, choices, and metadata.
              The exact contents depend on the return_option.

    Raises:
        ValueError: If return_option is neither 'full' nor 'minimal'.
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
    t_s = setup['t_s']
    cdef int num_draws = setup['num_draws']
    cdef float delta_t_sqrt = setup['delta_t_sqrt']

    # Param views
    cdef float[:] v_view = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] s_view = s
    cdef float[:] deadline_view = deadline

    cdef float y, t_particle, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n, ix, k
    cdef int m = 0

    # Variables for parallel execution - per-thread RNG states
    cdef RngState[MAX_THREADS] rng_states
    cdef uint64_t combined_seed
    cdef float v_k, a_k, z_k, t_k, s_k, sqrt_st_k, deadline_tmp_k
    cdef float drift_inc, noise
    cdef int choice
    cdef float max_t_c = <float>max_t  # Cache max_t for nogil
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
            # Loop over samples
            deadline_tmp = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])
            sqrt_st = delta_t_sqrt * s_view[k]
            for n in range(n_samples):
                y = z_view[k] * a_view[k] # reset starting point
                t_particle = 0.0 # reset time
                ix = 0 # reset boundary index

                if n == 0:
                    if k == 0:
                        traj_view[0, 0] = y

                # Random walker
                while y <= a_view[k] and y >= 0 and t_particle <= deadline_tmp:
                    y += v_view[k] * delta_t + sqrt_st * gaussian_values[m] # update particle position
                    t_particle += delta_t
                    m += 1
                    ix += 1

                    if n == 0:
                        if k == 0:
                            traj_view[ix, 0] = y

                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                # Note that for purposes of consistency with Navarro and Fuss,
                # the choice corresponding the lower barrier is +1, higher barrier is -1

                # Apply smoothing with uniform if desired
                smooth_u = compute_smooth_unif(smooth_unif, t_particle, deadline_tmp, delta_t)

                rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u # store rt
                choices_view[n, k, 0] = sign(y) # store choice

                # If the rt exceeds the deadline, set rt to -999 and choice to -1
                enforce_deadline(rts_view, deadline_view, n, k, 0)

    # =========================================================================
    # PARALLEL PATH (n_threads > 1): FLATTENED OpenMP parallelization with C RNG
    # Parallelizes over (n_trials × n_samples) for optimal efficiency
    # =========================================================================
    else:
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

                # Cache trial parameters
                v_k = v_view[k]
                a_k = a_view[k]
                z_k = z_view[k]
                t_k = t_view[k]
                s_k = s_view[k]
                sqrt_st_k = delta_t_sqrt * s_k
                deadline_tmp_k = fmin(max_t_c, deadline_view[k] - t_k)
                drift_inc = v_k * delta_t

                # Re-seed per-thread RNG with unique seed for this (trial, sample)
                combined_seed = rng_mix_seed(seed, <uint64_t>k, <uint64_t>n)
                rng_seed(&rng_states[tid], combined_seed)

                # Initialize particle
                y = z_k * a_k
                t_particle = 0.0

                # Random walk with inline C RNG
                while y > 0.0 and y < a_k and t_particle <= deadline_tmp_k:
                    noise = rng_gaussian_f32(&rng_states[tid])
                    y = y + drift_inc + sqrt_st_k * noise
                    t_particle = t_particle + delta_t

                # Store results
                rts_view[n, k, 0] = t_particle + t_k

                # Choice based on final position
                if y >= a_k:
                    choice = 1
                else:
                    choice = -1
                choices_view[n, k, 0] = choice

                # Deadline enforcement
                if rts_view[n, k, 0] >= deadline_view[k] or deadline_view[k] <= 0:
                    rts_view[n, k, 0] = -999.0

        # Free per-thread GSL RNGs AFTER parallel block
        for i_thread in range(c_n_threads):
            rng_free(&rng_states[i_thread])

        # Note: smooth_unif is not supported in parallel mode

    # Build minimal metadata first
    minimal_meta = build_minimal_metadata(
        simulator_name='ddm',
        possible_choices=[-1, 1],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name='constant'
    )

    if return_option == 'full':
        sim_config = {'delta_t': delta_t, 'max_t': max_t}
        params = {
            'v': v, 'a': a, 'z': z, 't': t,
            'deadline': deadline, 's': s
        }
        full_meta = build_full_metadata(
            minimal_metadata=minimal_meta,
            params=params,
            sim_config=sim_config,
            traj=traj
        )
        return build_return_dict(rts, choices, full_meta)

    elif return_option == 'minimal':
        return build_return_dict(rts, choices, minimal_meta)

    else:
        raise ValueError('return_option must be either "full" or "minimal"')



# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flexbound(np.ndarray[float, ndim = 1] v,
                  np.ndarray[float, ndim = 1] a,
                  np.ndarray[float, ndim = 1] z,
                  np.ndarray[float, ndim = 1] t,
                  np.ndarray[float, ndim = 1] deadline,
                  np.ndarray[float, ndim = 1] s, # noise sigma
                  float max_t = 20,
                  float delta_t = 0.001,
                  int n_samples = 20000,
                  int n_trials = 1,
                  boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                  boundary_params = {},
                  random_state = None,
                  return_option = 'full',
                  smooth_unif  = False,
                  int n_threads = 1,
                  **kwargs,
                  ):
    """
    Simulate reaction times and choices from a drift diffusion model with flexible boundaries.

    Args:
        v (np.ndarray): Drift rate for each trial.
        a (np.ndarray): Boundary separation for each trial (used for metadata).
        z (np.ndarray): Starting point bias for each trial (between 0 and 1).
        t (np.ndarray): Non-decision time for each trial.
        deadline (np.ndarray): Maximum allowed reaction time for each trial.
        s (np.ndarray): Noise (sigma) for each trial.
        max_t (float): Maximum time for simulation.
        delta_t (float): Time step for simulation.
        n_samples (int): Number of samples to simulate per trial.
        n_trials (int): Number of trials to simulate.
        boundary_fun (callable): Function defining the shape of the boundary.
        boundary_params (dict): Parameters for the boundary function.
        random_state (int or None): Seed for random number generator.
        return_option (str): 'full' for complete output, 'minimal' for basic output.
        smooth_unif (bool): Whether to apply uniform smoothing to reaction times.
        n_threads (int): Number of threads for parallel execution. Default is 1 (sequential).
                        If > 1 and OpenMP is available, uses parallel execution.
                        Note: Trajectory recording is only available with n_threads=1.
        **kwargs: Additional keyword arguments.

    Returns:
        dict: A dictionary containing simulated reaction times, choices, and metadata.
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
    t_s = setup['t_s']
    cdef int num_draws = setup['num_draws']
    cdef float delta_t_sqrt = setup['delta_t_sqrt']

    # Param views
    cdef float[:] v_view = v
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s

    # Boundary storage for the upper bound
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    # Number of timesteps
    cdef int num_steps = int((max_t / delta_t) + 1)

    cdef float y, t_particle, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0

    # Variables for parallel execution - per-thread RNG states
    cdef float[:, :] boundaries_all  # 2D: [n_trials, num_steps]
    cdef float[:] deadlines_tmp      # 1D: [n_trials]
    cdef float[:] sqrt_st_all        # 1D: [n_trials]
    cdef float[:] drift_inc_all      # 1D: [n_trials]
    cdef RngState[MAX_THREADS] rng_states
    cdef int tid
    cdef uint64_t combined_seed
    cdef float v_k, z_k, t_k, s_k, sqrt_st_k, deadline_tmp_k
    cdef float drift_inc, noise, bound_val, neg_bound_val
    cdef int choice
    cdef int i_thread

    # Flattened parallelization variables
    cdef Py_ssize_t flat_idx
    cdef Py_ssize_t total_iterations
    cdef int c_n_samples = n_samples
    cdef int c_n_threads = n_threads

    # =========================================================================
    # SEQUENTIAL PATH (n_threads == 1): Original algorithm, preserves all behavior
    # =========================================================================
    if n_threads == 1:
        # Loop over samples
        for k in range(n_trials):
            # Precompute boundary evaluations
            boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
            compute_boundary(boundary,
                             t_s,
                             boundary_fun,
                             boundary_params_tmp)

            deadline_tmp = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])
            sqrt_st = delta_t_sqrt * s_view[k]
            for n in range(n_samples):
                y = (-1) * boundary_view[0] + (z_view[k] * 2 * (boundary_view[0]))  # reset starting position
                t_particle = 0.0 # reset time
                ix = 0 # reset boundary index
                # if deadline >> max_t, then deadline_tmp = max_t, regardless of t-value, otherwise deadline applies
                # Can improve with less checks
                if n == 0:
                    if k == 0:
                        traj_view[0, 0] = y

                # Random walker
                while (y >= (-1) * boundary_view[ix]) and (y <= boundary_view[ix]) and (t_particle <= deadline_tmp):
                    y += (v_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                    t_particle += delta_t
                    ix += 1
                    m += 1

                    # Can improve with less checks
                    if n == 0:
                        if k == 0:
                            traj_view[ix, 0] = y

                    # Can improve with less checks
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                if smooth_unif :
                    if t_particle == 0.0:
                        smooth_u = random_uniform() * 0.5 * delta_t
                    elif t_particle < deadline_tmp:
                        smooth_u = (0.5 - random_uniform()) * delta_t
                    else:
                        smooth_u = 0.0
                else:
                    smooth_u = 0.0

                rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u # Store rt
                choices_view[n, k, 0] = sign(y) # Store choice

                enforce_deadline(rts_view, deadline_view, n, k, 0)

    # =========================================================================
    # PARALLEL PATH (n_threads > 1): FLATTENED OpenMP parallelization with C RNG
    # Parallelizes over (n_trials × n_samples) for optimal efficiency
    # =========================================================================
    else:
        # Precompute ALL trial data outside nogil
        boundaries_all_np = np.zeros((n_trials, num_steps), dtype=DTYPE)
        deadlines_tmp_np = np.zeros(n_trials, dtype=DTYPE)
        sqrt_st_all_np = np.zeros(n_trials, dtype=DTYPE)
        drift_inc_all_np = np.zeros(n_trials, dtype=DTYPE)

        for k in range(n_trials):
            boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
            compute_boundary(boundary, t_s, boundary_fun, boundary_params_tmp)
            boundaries_all_np[k, :] = boundary
            deadlines_tmp_np[k] = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])
            sqrt_st_all_np[k] = delta_t_sqrt * s_view[k]
            drift_inc_all_np[k] = v_view[k] * delta_t

        boundaries_all = boundaries_all_np
        deadlines_tmp = deadlines_tmp_np
        sqrt_st_all = sqrt_st_all_np
        drift_inc_all = drift_inc_all_np

        # Total iterations = n_trials × n_samples
        total_iterations = <Py_ssize_t>n_trials * <Py_ssize_t>n_samples

        # Allocate per-thread GSL RNGs BEFORE parallel block
        for i_thread in range(c_n_threads):
            rng_alloc(&rng_states[i_thread])

        # Parallel execution over FLATTENED iteration space
        with nogil, parallel(num_threads=n_threads):
            for flat_idx in prange(total_iterations, schedule='dynamic'):
                # Get thread ID for per-thread RNG
                tid = threadid()

                # Compute (k, n) from flat index
                k = flat_idx // c_n_samples
                n = flat_idx % c_n_samples

                # Access pre-computed trial parameters
                z_k = z_view[k]
                t_k = t_view[k]
                sqrt_st_k = sqrt_st_all[k]
                deadline_tmp_k = deadlines_tmp[k]
                drift_inc = drift_inc_all[k]

                # Re-seed per-thread RNG with unique seed for this (trial, sample) pair
                combined_seed = rng_mix_seed(seed, <uint64_t>k, <uint64_t>n)
                rng_seed(&rng_states[tid], combined_seed)

                # Initialize particle position using boundary at t=0
                bound_val = boundaries_all[k, 0]
                y = (-1.0) * bound_val + (z_k * 2.0 * bound_val)
                t_particle = 0.0
                ix = 0

                # Random walk with GSL Ziggurat RNG
                while True:
                    bound_val = boundaries_all[k, ix]
                    neg_bound_val = -bound_val

                    if y < neg_bound_val or y > bound_val or t_particle > deadline_tmp_k:
                        break

                    noise = rng_gaussian_f32(&rng_states[tid])
                    y = y + drift_inc + sqrt_st_k * noise
                    t_particle = t_particle + delta_t
                    ix = ix + 1

                    # Safety: don't exceed boundary array
                    if ix >= num_steps:
                        break

                # Store results
                rts_view[n, k, 0] = t_particle + t_k

                # Choice based on final position (sign of y)
                # Same logic as sequential: sign(y) = 1 if y > 0, -1 if y < 0
                if y > 0.0:
                    choice = 1
                elif y < 0.0:
                    choice = -1
                else:
                    choice = 0
                choices_view[n, k, 0] = choice

                # Deadline enforcement (inline for nogil)
                if rts_view[n, k, 0] >= deadline_view[k] or deadline_view[k] <= 0:
                    rts_view[n, k, 0] = -999.0

        # Free per-thread GSL RNGs AFTER parallel block
        for i_thread in range(c_n_threads):
            rng_free(&rng_states[i_thread])

        # Note: smooth_unif is not supported in parallel mode

    # Build minimal metadata first
    minimal_meta = build_minimal_metadata(
        simulator_name='ddm_flexbound',
        possible_choices=[-1, 1],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )

    if return_option == 'full':
        sim_config = {'delta_t': delta_t, 'max_t': max_t, 'n_threads': n_threads}
        params = {
            'v': v, 'a': a, 'z': z, 't': t,
            's': s, 'deadline': deadline
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
## ----------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES AND FLEXIBLE SLOPE -----------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flex(np.ndarray[float, ndim = 1] v,
             np.ndarray[float, ndim = 1] a,
             np.ndarray[float, ndim = 1] z,
             np.ndarray[float, ndim = 1] t,
             np.ndarray[float, ndim = 1] deadline,
             np.ndarray[float, ndim = 1] s, # noise sigma
             float delta_t = 0.001,
             float max_t = 20,
             int n_samples = 20000,
             int n_trials = 1,
             boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
             drift_fun = None,
             boundary_params = {},
             drift_params = {},
             random_state = None,
             return_option = 'full',
             smooth_unif = False,
             int n_threads = 1,
             **kwargs):
    """
    Simulate reaction times and choices from a drift diffusion model with flexible boundaries and flexible drift.

    Args:
        v (np.ndarray): Drift rate for each trial.
        a (np.ndarray): Boundary separation for each trial.
        z (np.ndarray): Starting point (between 0 and 1) for each trial.
        t (np.ndarray): Non-decision time for each trial.
        deadline (np.ndarray): Maximum reaction time allowed for each trial.
        s (np.ndarray): Noise standard deviation for each trial.
        delta_t (float): Time step size (default: 0.001).
        max_t (float): Maximum simulation time (default: 20).
        n_samples (int): Number of samples per trial (default: 20000).
        n_trials (int): Number of trials to simulate (default: 1).
        boundary_fun (callable): Function defining the decision boundary over time.
        drift_fun (callable): Function defining the drift rate over time.
        boundary_params (dict): Parameters for the boundary function.
        drift_params (dict): Parameters for the drift function.
        random_state (int or None): Seed for random number generator (default: None).
        return_option (str): 'full' or 'minimal' return format (default: 'full').
        smooth_unif (bool): Whether to apply uniform smoothing to reaction times (default: False).
        **kwargs: Additional keyword arguments.

    Returns:
        dict: A dictionary containing simulated reaction times, choices, and metadata.
              The exact contents depend on the return_option.

    Raises:
        ValueError: If return_option is neither 'full' nor 'minimal'.
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
    t_s = setup['t_s']
    cdef int num_draws = setup['num_draws']
    cdef float delta_t_sqrt = setup['delta_t_sqrt']
    cdef int num_steps = int((max_t / delta_t) + 1)

    # Param views
    cdef float[:] v_view = v
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s

    # Boundary and drift storage
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    drift = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary
    cdef float[:] drift_view = drift

    cdef float y, t_particle, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0

    # Variables for parallel execution - per-thread RNG states
    cdef float[:, :] boundaries_all
    cdef float[:, :] drifts_all
    cdef float[:] deadlines_tmp
    cdef float[:] sqrt_st_all
    cdef RngState[MAX_THREADS] rng_states
    cdef uint64_t combined_seed
    cdef float z_k, t_k, s_k, sqrt_st_k, deadline_tmp_k
    cdef float drift_val, noise, bound_val, neg_bound_val
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
            # Precompute boundary evaluations and drift evaluations

            # Drift - drift functions now return final drift value (v is included in drift_params)
            drift_params_tmp = {key: drift_params[key][k] for key in drift_params.keys()}
            drift[:] = drift_fun(t = t_s, **drift_params_tmp).astype(DTYPE)

            # Boundary
            boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
            compute_boundary(boundary,
                             t_s,
                             boundary_fun,
                             boundary_params_tmp)

            deadline_tmp = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])

            sqrt_st = delta_t_sqrt * s_view[k]
            for n in range(n_samples):
                y = (-1) * boundary_view[0] + (z_view[k] * 2 * (boundary_view[0]))  # reset starting position
                t_particle = 0.0 # reset time
                ix = 0 # reset boundary index

                # Can improve with less checks
                if n == 0:
                    if k == 0:
                        traj_view[0, 0] = y

                # Random walker
                while (y >= (-1) * boundary_view[ix]) and (y <= boundary_view[ix]) and (t_particle <= deadline_tmp):
                    y += (drift_view[ix] * delta_t) + (sqrt_st * gaussian_values[m])
                    t_particle += delta_t
                    ix += 1
                    m += 1

                    # Can improve with less checks
                    if n == 0:
                        if k == 0:
                            traj_view[ix, 0] = y

                    # Can improve with less checks
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                if smooth_unif :
                    if t_particle == 0.0:
                        smooth_u = random_uniform() * 0.5 * delta_t
                    elif t_particle < deadline_tmp:
                        smooth_u = (0.5 - random_uniform()) * delta_t
                    else:
                        smooth_u = 0.0
                else:
                    smooth_u = 0.0

                rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u # Store rt
                choices_view[n, k, 0] = sign(y) # Store choice

                enforce_deadline(rts_view, deadline_view, n, k, 0)

    # =========================================================================
    # PARALLEL PATH (n_threads > 1): FLATTENED OpenMP parallelization with C RNG
    # Parallelizes over (n_trials × n_samples) for optimal efficiency
    # =========================================================================
    else:
        # Precompute ALL trial data outside nogil
        boundaries_all_np = np.zeros((n_trials, num_steps), dtype=DTYPE)
        drifts_all_np = np.zeros((n_trials, num_steps), dtype=DTYPE)
        deadlines_tmp_np = np.zeros(n_trials, dtype=DTYPE)
        sqrt_st_all_np = np.zeros(n_trials, dtype=DTYPE)

        for k in range(n_trials):
            # Drift
            drift_params_tmp = {key: drift_params[key][k] for key in drift_params.keys()}
            drifts_all_np[k, :] = drift_fun(t=t_s, **drift_params_tmp).astype(DTYPE)

            # Boundary
            boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
            compute_boundary(boundary, t_s, boundary_fun, boundary_params_tmp)
            boundaries_all_np[k, :] = boundary
            deadlines_tmp_np[k] = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])
            sqrt_st_all_np[k] = delta_t_sqrt * s_view[k]

        boundaries_all = boundaries_all_np
        drifts_all = drifts_all_np
        deadlines_tmp = deadlines_tmp_np
        sqrt_st_all = sqrt_st_all_np

        # Total iterations = n_trials × n_samples
        total_iterations = <Py_ssize_t>n_trials * <Py_ssize_t>n_samples
        c_n_threads = n_threads

        # Allocate per-thread GSL RNGs BEFORE parallel block
        for i_thread in range(c_n_threads):
            rng_alloc(&rng_states[i_thread])

        # Parallel execution over FLATTENED iteration space
        with nogil, parallel(num_threads=n_threads):
            for flat_idx in prange(total_iterations, schedule='dynamic'):
                # Get thread ID for per-thread RNG
                tid = threadid()

                # Compute (k, n) from flat index
                k = flat_idx // c_n_samples
                n = flat_idx % c_n_samples

                # Access pre-computed trial parameters
                z_k = z_view[k]
                t_k = t_view[k]
                sqrt_st_k = sqrt_st_all[k]
                deadline_tmp_k = deadlines_tmp[k]

                # Re-seed per-thread RNG with unique seed for this (trial, sample)
                combined_seed = rng_mix_seed(seed, <uint64_t>k, <uint64_t>n)
                rng_seed(&rng_states[tid], combined_seed)

                # Initialize particle position
                bound_val = boundaries_all[k, 0]
                y = (-1.0) * bound_val + (z_k * 2.0 * bound_val)
                t_particle = 0.0
                ix = 0

                # Random walk with inline C RNG
                while True:
                    bound_val = boundaries_all[k, ix]
                    neg_bound_val = -bound_val

                    if y < neg_bound_val or y > bound_val or t_particle > deadline_tmp_k:
                        break

                    drift_val = drifts_all[k, ix]
                    noise = rng_gaussian_f32(&rng_states[tid])
                    y = y + (drift_val * delta_t) + sqrt_st_k * noise
                    t_particle = t_particle + delta_t
                    ix = ix + 1

                    if ix >= num_steps:
                        break

                # Store results
                rts_view[n, k, 0] = t_particle + t_k

                # Choice based on sign of y (same as sequential path)
                if y > 0.0:
                    choice = 1
                elif y < 0.0:
                    choice = -1
                else:
                    choice = 0
                choices_view[n, k, 0] = choice

                if rts_view[n, k, 0] >= deadline_view[k] or deadline_view[k] <= 0:
                    rts_view[n, k, 0] = -999.0

        # Free per-thread GSL RNGs AFTER parallel block
        for i_thread in range(c_n_threads):
            rng_free(&rng_states[i_thread])

    # Build minimal metadata first
    minimal_meta = build_minimal_metadata(
        simulator_name='ddm_flex',
        possible_choices=[-1, 1],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )
    # Add drift_fun_type for this model
    minimal_meta['drift_fun_type'] = boundary_fun.__name__

    if return_option == 'full':
        sim_config = {'delta_t': delta_t, 'max_t': max_t}
        params = {
            'v': v, 'a': a, 'z': z, 't': t,
            'deadline': deadline, 's': s
        }
        # Add drift array to extra params
        extra_params_dict = {'drift': drift}
        full_meta = build_full_metadata(
            minimal_metadata=minimal_meta,
            params=params,
            sim_config=sim_config,
            boundary_fun=boundary_fun,
            boundary=boundary,
            traj=traj,
            boundary_params=boundary_params,
            drift_params=drift_params,
            extra_params=extra_params_dict
        )
        return build_return_dict(rts, choices, full_meta)

    elif return_option == 'minimal':
        return build_return_dict(rts, choices, minimal_meta)

    else:
        raise ValueError('return_option must be either "full" or "minimal"')


# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES, FLEXIBLE SLOPE, AND DECAY ----------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flex_leak(np.ndarray[float, ndim = 1] v,
             np.ndarray[float, ndim = 1] a,
             np.ndarray[float, ndim = 1] z,
             np.ndarray[float, ndim = 1] g,
             np.ndarray[float, ndim = 1] t,
             np.ndarray[float, ndim = 1] deadline,
             np.ndarray[float, ndim = 1] s, # noise sigma
             float delta_t = 0.001,
             float max_t = 20,
             int n_samples = 20000,
             int n_trials = 1,
             boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
             drift_fun = None,
             boundary_params = {},
             drift_params = {},
             random_state = None,
             return_option = 'full',
             smooth_unif  = False,
             int n_threads = 1,
             **kwargs):
    """
    Simulate reaction times and choices from a drift diffusion model with flexible boundaries, flexible drift, and decay.

    Args:
        v (np.ndarray): Drift rate for each trial.
        a (np.ndarray): Boundary separation for each trial.
        z (np.ndarray): Starting point (between 0 and 1) for each trial.
        g (np.ndarray): Decay parameter for each trial.
        t (np.ndarray): Non-decision time for each trial.
        deadline (np.ndarray): Maximum reaction time allowed for each trial.
        s (np.ndarray): Noise standard deviation for each trial.
        delta_t (float): Time step size (default: 0.001).
        max_t (float): Maximum simulation time (default: 20).
        n_samples (int): Number of samples per trial (default: 20000).
        n_trials (int): Number of trials to simulate (default: 1).
        boundary_fun (callable): Function defining the decision boundary over time.
        drift_fun (callable): Function defining the drift rate over time.
        boundary_params (dict): Parameters for the boundary function.
        drift_params (dict): Parameters for the drift function.
        random_state (int or None): Seed for random number generator (default: None).
        return_option (str): 'full' or 'minimal' return format (default: 'full').
        smooth_unif (bool): Whether to apply uniform smoothing to reaction times (default: False).
        **kwargs: Additional keyword arguments.

    Returns:
        dict: A dictionary containing simulated reaction times, choices, and metadata.
              The exact contents depend on the return_option.

    Raises:
        ValueError: If return_option is neither 'full' nor 'minimal'.
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
    t_s = setup['t_s']
    cdef int num_draws = setup['num_draws']
    cdef float delta_t_sqrt = setup['delta_t_sqrt']
    cdef int num_steps = int((max_t / delta_t) + 1)

    # Param views
    cdef float[:] v_view = v
    cdef float[:] z_view = z
    cdef float[:] g_view = g
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s

    # Boundary and drift storage
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    drift = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary
    cdef float[:] drift_view = drift

    cdef float y, t_particle, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0

    # Variables for parallel execution - per-thread RNG states
    cdef float[:, :] boundaries_all
    cdef float[:, :] drifts_all
    cdef float[:] deadlines_tmp
    cdef float[:] sqrt_st_all
    cdef RngState[MAX_THREADS] rng_states
    cdef uint64_t combined_seed
    cdef float z_k, t_k, g_k, s_k, sqrt_st_k, deadline_tmp_k
    cdef float drift_val, noise, bound_val, neg_bound_val
    cdef int choice
    cdef int tid
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
            # Precompute boundary evaluations and drift evaluations

            # Drift - drift functions now return final drift value (v is included in drift_params)
            drift_params_tmp = {key: drift_params[key][k] for key in drift_params.keys()}
            drift[:] = drift_fun(t = t_s, **drift_params_tmp).astype(DTYPE)

            # Boundary
            boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
            compute_boundary(boundary,
                             t_s,
                             boundary_fun,
                             boundary_params_tmp)

            deadline_tmp = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])
            sqrt_st = delta_t_sqrt * s_view[k]
            for n in range(n_samples):
                y = (-1) * boundary_view[0] + (z_view[k] * 2 * (boundary_view[0]))  # reset starting position
                t_particle = 0.0 # reset time
                ix = 0 # reset boundary index


                # Can improve with less checks
                if n == 0:
                    if k == 0:
                        traj_view[0, 0] = y

                # Random walker
                while (y >= (-1) * boundary_view[ix]) and (y <= boundary_view[ix]) and (t_particle <= deadline_tmp):
                    y += ((drift_view[ix] - (g_view[k] * y)) * delta_t) + (sqrt_st * gaussian_values[m])
                    t_particle += delta_t
                    ix += 1
                    m += 1

                    # Can improve with less checks
                    if n == 0:
                        if k == 0:
                            traj_view[ix, 0] = y

                    # Can improve with less checks
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                if smooth_unif :
                    if t_particle == 0.0:
                        smooth_u = random_uniform() * 0.5 * delta_t
                    elif t_particle < deadline_tmp:
                        smooth_u = (0.5 - random_uniform()) * delta_t
                    else:
                        smooth_u = 0.0
                else:
                    smooth_u = 0.0

                rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u # Store rt
                choices_view[n, k, 0] = sign(y) # Store choice

                enforce_deadline(rts_view, deadline_view, n, k, 0)

    # =========================================================================
    # PARALLEL PATH (n_threads > 1): FLATTENED OpenMP parallelization with C RNG
    # Parallelizes over (n_trials × n_samples) for optimal efficiency
    # =========================================================================
    else:
        # Precompute ALL trial data outside nogil
        boundaries_all_np = np.zeros((n_trials, num_steps), dtype=DTYPE)
        drifts_all_np = np.zeros((n_trials, num_steps), dtype=DTYPE)
        deadlines_tmp_np = np.zeros(n_trials, dtype=DTYPE)
        sqrt_st_all_np = np.zeros(n_trials, dtype=DTYPE)

        for k in range(n_trials):
            drift_params_tmp = {key: drift_params[key][k] for key in drift_params.keys()}
            drifts_all_np[k, :] = drift_fun(t=t_s, **drift_params_tmp).astype(DTYPE)

            boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
            compute_boundary(boundary, t_s, boundary_fun, boundary_params_tmp)
            boundaries_all_np[k, :] = boundary
            deadlines_tmp_np[k] = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])
            sqrt_st_all_np[k] = delta_t_sqrt * s_view[k]

        boundaries_all = boundaries_all_np
        drifts_all = drifts_all_np
        deadlines_tmp = deadlines_tmp_np
        sqrt_st_all = sqrt_st_all_np

        # Total iterations = n_trials × n_samples
        total_iterations = <Py_ssize_t>n_trials * <Py_ssize_t>n_samples
        c_n_threads = n_threads

        # Allocate per-thread GSL RNGs BEFORE parallel block
        for i_thread in range(c_n_threads):
            rng_alloc(&rng_states[i_thread])

        with nogil, parallel(num_threads=n_threads):
            for flat_idx in prange(total_iterations, schedule='dynamic'):
                tid = threadid()

                # Compute (k, n) from flat index
                k = flat_idx // c_n_samples
                n = flat_idx % c_n_samples

                z_k = z_view[k]
                g_k = g_view[k]
                t_k = t_view[k]
                sqrt_st_k = sqrt_st_all[k]
                deadline_tmp_k = deadlines_tmp[k]

                combined_seed = rng_mix_seed(seed, <uint64_t>k, <uint64_t>n)
                rng_seed(&rng_states[tid], combined_seed)

                bound_val = boundaries_all[k, 0]
                y = (-1.0) * bound_val + (z_k * 2.0 * bound_val)
                t_particle = 0.0
                ix = 0

                while True:
                    bound_val = boundaries_all[k, ix]
                    neg_bound_val = -bound_val

                    if y < neg_bound_val or y > bound_val or t_particle > deadline_tmp_k:
                        break

                    drift_val = drifts_all[k, ix]
                    noise = rng_gaussian_f32(&rng_states[tid])
                    # Leak term: drift_val - g_k * y
                    y = y + ((drift_val - g_k * y) * delta_t) + sqrt_st_k * noise
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

                if rts_view[n, k, 0] >= deadline_view[k] or deadline_view[k] <= 0:
                    rts_view[n, k, 0] = -999.0

        # Free per-thread GSL RNGs AFTER parallel block
        for i_thread in range(c_n_threads):
            rng_free(&rng_states[i_thread])

    # Build minimal metadata first
    minimal_meta = build_minimal_metadata(
        simulator_name='ddm_flex_leak',
        possible_choices=[-1, 1],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )
    # Add drift_fun_type for this model
    minimal_meta['drift_fun_type'] = boundary_fun.__name__

    if return_option == 'full':
        sim_config = {'delta_t': delta_t, 'max_t': max_t}
        params = {
            'v': v, 'a': a, 'z': z, 'g': g, 't': t,
            'deadline': deadline, 's': s
        }
        # Add drift and boundary arrays to extra params
        extra_params_dict = {'drift': drift}
        full_meta = build_full_metadata(
            minimal_metadata=minimal_meta,
            params=params,
            sim_config=sim_config,
            boundary_fun=boundary_fun,
            boundary=boundary,
            traj=traj,
            boundary_params=boundary_params,
            drift_params=drift_params,
            extra_params=extra_params_dict
        )
        return build_return_dict(rts, choices, full_meta)

    elif return_option == 'minimal':
        return build_return_dict(rts, choices, minimal_meta)

    else:
        raise ValueError('return_option must be either "full" or "minimal"')



# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES, FLEXIBLE SLOPE, AND DUAL LEAK ----------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flex_leak2(
    np.ndarray[float, ndim = 1] vt,
    np.ndarray[float, ndim = 1] vd,
    np.ndarray[float, ndim = 1] a,
    np.ndarray[float, ndim = 1] z,
    np.ndarray[float, ndim = 1] gt,
    np.ndarray[float, ndim = 1] gd,
    np.ndarray[float, ndim = 1] t,
    np.ndarray[float, ndim = 1] deadline,
    np.ndarray[float, ndim = 1] s, # noise sigma
    float delta_t = 0.001,
    float max_t = 20,
    int n_samples = 20000,
    int n_trials = 1,
    boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
    drift_fun = None,
    boundary_params = {},
    drift_params = {},
    random_state = None,
    return_option = 'full',
    smooth_unif  = False,
    int n_threads = 1,
    **kwargs):
    """
    Simulate reaction times and choices from a sequential sampling model that pools choice evidence across two sensory
    input dimensions, one 'target' and one 'distractor', with flexible boundaries, flexible drifts, and separate decay
    parameters.

    This model assumes two accumulators for two sensory input dimensions, a 'target' and 'distractor' dimension. Each
    accumulator has its own drift rate and decay parameter, but fully share noise. The decision variable (DV) from each
    accumulator, `y_t` and `y_d`, is summed to form a single DV, `y`, which is compared to the decision boundaries.
    Individual accumulators are always initiated with no starting-point bias; instead, the starting point bias is
    applied to the combined DV `y`.

    Args:
        vt (np.ndarray): Drift rate for target input for each trial.
        vd (np.ndarray): Drift rate for distractor input for each trial.
        a (np.ndarray): Boundary separation for each trial.
        z (np.ndarray): Starting point (between 0 and 1) for each trial.
        gt (np.ndarray): Decay parameter for target input for each trial.
        gd (np.ndarray): Decay parameter for distractor input for each trial.
        t (np.ndarray): Non-decision time for each trial.
        deadline (np.ndarray): Maximum reaction time allowed for each trial.
        s (np.ndarray): Noise standard deviation for each trial.
        delta_t (float): Time step size (default: 0.001).
        max_t (float): Maximum simulation time (default: 20).
        n_samples (int): Number of samples per trial (default: 20000).
        n_trials (int): Number of trials to simulate (default: 1).
        boundary_fun (callable): Function defining the decision boundary over time.
        drift_fun (callable): Function defining the drift rate over time.
        boundary_params (dict): Parameters for the boundary function.
        drift_params (dict): Parameters for the drift function.
        random_state (int or None): Seed for random number generator (default: None).
        return_option (str): 'full' or 'minimal' return format (default: 'full').
        smooth_unif (bool): Whether to apply uniform smoothing to reaction times (default: False).
        **kwargs: Additional keyword arguments.

    Returns:
        dict: A dictionary containing simulated reaction times, choices, and metadata.
              The exact contents depend on the return_option.

    Raises:
        ValueError: If return_option is neither 'full' nor 'minimal'.
    """

    # Check if parallel execution is requested and available
    n_threads = check_parallel_request(n_threads)

    # Get seed for reproducibility
    cdef uint64_t seed = random_state if random_state is not None else np.random.randint(0, 2**31)

    setup = setup_simulation(n_samples, n_trials, max_t, delta_t, random_state)

    # Extract basic setup
    rts = setup['rts']
    choices = setup['choices']
    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices
    cdef float[:] gaussian_values = setup['gaussian_values']
    t_s = setup['t_s']
    cdef int num_draws = setup['num_draws']
    cdef float delta_t_sqrt = setup['delta_t_sqrt']
    cdef int num_steps = int((max_t / delta_t) + 1)

    # Custom traj array for this model (3 columns)
    traj = np.zeros((num_draws, 3), dtype = DTYPE)
    traj[:, :] = -999
    cdef float[:, :] traj_view = traj

    # Param views
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] g_t_view = gt
    cdef float[:] g_d_view = gd
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s

    # Boundary and drift storage
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    drift = np.zeros((t_s.shape[0], 2), dtype = DTYPE)
    cdef float[:] boundary_view = boundary
    cdef float[:, :] drift_view = drift

    cdef float y_t, y_d, y_start, y, t_particle, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0

    # Variables for parallel execution
    cdef float[:, :] boundaries_all
    cdef float[:, :, :] drifts_all
    cdef float[:] deadlines_tmp
    cdef float[:] sqrt_st_all
    cdef float[:] half_sqrt_st_all
    cdef RngState[MAX_THREADS] rng_states
    cdef uint64_t combined_seed
    cdef float z_k, t_k, g_t_k, g_d_k, sqrt_st_k, deadline_tmp_k
    cdef float drift_t, drift_d, noise, bound_val, neg_bound_val, half_sqrt_st
    cdef int choice
    cdef int tid
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
            # Precompute boundary evaluations and drift evaluations

            # Drift
            drift_params_tmp = {key: drift_params[key][k] for key in drift_params.keys()}
            drift[:, :] = drift_fun(t = t_s, **drift_params_tmp).astype(DTYPE)

            # Boundary
            boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
            compute_boundary(boundary,
                             t_s,
                             boundary_fun,
                             boundary_params_tmp)

            deadline_tmp = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])
            sqrt_st = delta_t_sqrt * s_view[k]
            for n in range(n_samples):
                y_start = (-1) * boundary_view[0] + (z_view[k] * 2 * (boundary_view[0]))  # reset starting position
                y = y_start
                y_t = 0.0
                y_d = 0.0
                t_particle = 0.0 # reset time
                ix = 0 # reset boundary index

                # Can improve with less checks
                if n == 0:
                    if k == 0:
                        traj_view[0, 0] = y
                        traj_view[0, 1] = y_t
                        traj_view[0, 2] = y_d

                # Random walker
                while (y >= (-1) * boundary_view[ix]) and (y <= boundary_view[ix]) and (t_particle <= deadline_tmp):
                    y_t += ((drift_view[ix, 0] - (g_t_view[k] * y_t)) * delta_t) + (sqrt_st/2 * gaussian_values[m])
                    y_d += ((drift_view[ix, 1] - (g_d_view[k] * y_d)) * delta_t) + (sqrt_st/2 * gaussian_values[m])
                    y = y_start + y_t + y_d

                    t_particle += delta_t
                    ix += 1
                    m += 1

                    # Can improve with less checks
                    if n == 0:
                        if k == 0:
                            traj_view[ix, 0] = y
                            traj_view[ix, 1] = y_t
                            traj_view[ix, 2] = y_d

                    # Can improve with less checks
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                if smooth_unif :
                    if t_particle == 0.0:
                        smooth_u = random_uniform() * 0.5 * delta_t
                    elif t_particle < deadline_tmp:
                        smooth_u = (0.5 - random_uniform()) * delta_t
                    else:
                        smooth_u = 0.0
                else:
                    smooth_u = 0.0

                rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u # Store rt
                choices_view[n, k, 0] = sign(y) # Store choice

                enforce_deadline(rts_view, deadline_view, n, k, 0)

    # =========================================================================
    # PARALLEL PATH (n_threads > 1): FLATTENED OpenMP parallelization with C RNG
    # Parallelizes over (n_trials × n_samples) for optimal efficiency
    # =========================================================================
    else:
        # Precompute ALL trial data outside nogil
        boundaries_all_np = np.zeros((n_trials, num_steps), dtype=DTYPE)
        drifts_all_np = np.zeros((n_trials, num_steps, 2), dtype=DTYPE)
        deadlines_tmp_np = np.zeros(n_trials, dtype=DTYPE)
        sqrt_st_all_np = np.zeros(n_trials, dtype=DTYPE)
        half_sqrt_st_all_np = np.zeros(n_trials, dtype=DTYPE)

        for k in range(n_trials):
            drift_params_tmp = {key: drift_params[key][k] for key in drift_params.keys()}
            drifts_all_np[k, :, :] = drift_fun(t=t_s, **drift_params_tmp).astype(DTYPE)

            boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
            compute_boundary(boundary, t_s, boundary_fun, boundary_params_tmp)
            boundaries_all_np[k, :] = boundary
            deadlines_tmp_np[k] = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])
            sqrt_st_all_np[k] = delta_t_sqrt * s_view[k]
            half_sqrt_st_all_np[k] = sqrt_st_all_np[k] / 2.0

        boundaries_all = boundaries_all_np
        drifts_all = drifts_all_np
        deadlines_tmp = deadlines_tmp_np
        sqrt_st_all = sqrt_st_all_np
        half_sqrt_st_all = half_sqrt_st_all_np

        # Total iterations = n_trials × n_samples
        total_iterations = <Py_ssize_t>n_trials * <Py_ssize_t>n_samples
        c_n_threads = n_threads

        # Allocate per-thread GSL RNGs BEFORE parallel block
        for i_thread in range(c_n_threads):
            rng_alloc(&rng_states[i_thread])

        with nogil, parallel(num_threads=n_threads):
            for flat_idx in prange(total_iterations, schedule='dynamic'):
                tid = threadid()

                # Compute (k, n) from flat index
                k = flat_idx // c_n_samples
                n = flat_idx % c_n_samples

                z_k = z_view[k]
                g_t_k = g_t_view[k]
                g_d_k = g_d_view[k]
                t_k = t_view[k]
                half_sqrt_st = half_sqrt_st_all[k]
                deadline_tmp_k = deadlines_tmp[k]

                combined_seed = rng_mix_seed(seed, <uint64_t>k, <uint64_t>n)
                rng_seed(&rng_states[tid], combined_seed)

                bound_val = boundaries_all[k, 0]
                y_start = (-1.0) * bound_val + (z_k * 2.0 * bound_val)
                y = y_start
                y_t = 0.0
                y_d = 0.0
                t_particle = 0.0
                ix = 0

                while True:
                    bound_val = boundaries_all[k, ix]
                    neg_bound_val = -bound_val

                    if y < neg_bound_val or y > bound_val or t_particle > deadline_tmp_k:
                        break

                    drift_t = drifts_all[k, ix, 0]
                    drift_d = drifts_all[k, ix, 1]
                    noise = rng_gaussian_f32(&rng_states[tid])

                    # Both accumulators use the same noise (shared variance)
                    y_t = y_t + ((drift_t - g_t_k * y_t) * delta_t) + half_sqrt_st * noise
                    y_d = y_d + ((drift_d - g_d_k * y_d) * delta_t) + half_sqrt_st * noise
                    y = y_start + y_t + y_d

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

                if rts_view[n, k, 0] >= deadline_view[k] or deadline_view[k] <= 0:
                    rts_view[n, k, 0] = -999.0

        # Free per-thread GSL RNGs AFTER parallel block
        for i_thread in range(c_n_threads):
            rng_free(&rng_states[i_thread])

    # Build minimal metadata first
    minimal_meta = build_minimal_metadata(
        simulator_name='ddm_flex_leak2',
        possible_choices=[-1, 1],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )
    # Add drift_fun_type for this model
    minimal_meta['drift_fun_type'] = boundary_fun.__name__

    if return_option == 'full':
        sim_config = {'delta_t': delta_t, 'max_t': max_t}
        params = {
            'vt': vt, 'vd': vd, 'a': a, 'z': z,
            'gt': gt, 'gd': gd, 't': t,
            'deadline': deadline, 's': s
        }
        extra_params_dict = {'drift': drift}
        full_meta = build_full_metadata(
            minimal_metadata=minimal_meta,
            params=params,
            sim_config=sim_config,
            boundary_fun=boundary_fun,
            boundary=boundary,
            traj=traj,
            boundary_params=boundary_params,
            drift_params=drift_params,
            extra_params=extra_params_dict
        )
        return build_return_dict(rts, choices, full_meta)

    elif return_option == 'minimal':
        return build_return_dict(rts, choices, minimal_meta)
    else:
        raise ValueError('return_option must be either "full" or "minimal"')


# Simulate (rt, choice) tuples from: Levy Flight with Flex Bound -------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def full_ddm_rv(np.ndarray[float, ndim = 1] v, # = 0,
                np.ndarray[float, ndim = 1] a, # = 1,
                np.ndarray[float, ndim = 1] z, # = 0.5,
                np.ndarray[float, ndim = 1] t, # = 0.0,
                z_dist, # = 0.05,
                v_dist, # = 0.1,
                t_dist, # = 0.0,
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
    Simulate reaction times and choices from a full drift diffusion model with flexible boundaries and random variability.

    Args:
        v (np.ndarray): Drift rate for each trial.
        a (np.ndarray): Boundary separation for each trial.
        z (np.ndarray): Starting point (between 0 and 1) for each trial.
        t (np.ndarray): Non-decision time for each trial.
        z_dist: Distribution function for starting point variability.
        v_dist: Distribution function for drift rate variability.
        t_dist: Distribution function for non-decision time variability.
        deadline (np.ndarray): Maximum reaction time allowed for each trial.
        s (np.ndarray): Noise standard deviation for each trial.
        delta_t (float): Time step size for simulation (default: 0.001).
        max_t (float): Maximum simulation time (default: 20).
        n_samples (int): Number of samples per trial (default: 20000).
        n_trials (int): Number of trials to simulate (default: 1).
        boundary_fun (callable): Function defining the decision boundary over time.
        boundary_params (dict): Parameters for the boundary function.
        random_state (int or None): Seed for random number generator (default: None).
        return_option (str): 'full' or 'minimal' return format (default: 'full').
        smooth_unif (bool): Whether to apply uniform smoothing to reaction times (default: False).
        **kwargs: Additional keyword arguments.

    Returns:
        dict: A dictionary containing simulated reaction times, choices, and metadata.
              The exact contents depend on the return_option.

    Raises:
        ValueError: If return_option is neither 'full' nor 'minimal'.
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
    t_s = setup['t_s']
    cdef int num_draws = setup['num_draws']
    cdef float delta_t_sqrt = setup['delta_t_sqrt']
    cdef int num_steps = int((max_t / delta_t) + 1)
    cdef float c_max_t = max_t

    # Param views
    cdef float[:] v_view = v
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s

    # Distribution sampling arrays for starting point, drift, and non-decision time variability
    sz_samplewise = np.zeros((n_trials, n_samples), dtype = DTYPE)
    sv_samplewise = np.zeros((n_trials, n_samples), dtype = DTYPE)
    st_samplewise = np.zeros((n_trials, n_samples), dtype = DTYPE)
    cdef float[:, :] sz_samplewise_view = sz_samplewise
    cdef float[:, :] sv_samplewise_view = sv_samplewise
    cdef float[:, :] st_samplewise_view = st_samplewise

    # Boundary storage
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef float y, t_particle, t_tmp, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0
    cdef float drift_increment = 0.0

    # Variables for parallel execution - per-thread RNG states
    cdef float[:, :] boundaries_all
    cdef float[:] sqrt_st_all
    cdef RngState[MAX_THREADS] rng_states
    cdef uint64_t combined_seed
    cdef float z_k, v_k, t_k, s_k, sqrt_st_k, deadline_k
    cdef float sz_n, sv_n, st_n, bound_val, neg_bound_val, noise
    cdef int choice
    cdef int tid
    cdef int i_thread
    cdef int c_n_threads

    # Flattened parallelization variables
    cdef Py_ssize_t flat_idx
    cdef Py_ssize_t total_iterations
    cdef int c_n_samples = n_samples

    # Pre-generate all distribution samples (this is done before the main loop for both paths)
    sv_samplewise[:, :] = v_dist(size = (n_samples, n_trials)).T
    sz_samplewise[:, :] = z_dist(size = (n_samples, n_trials)).T
    st_samplewise[:, :] = t_dist(size = (n_samples, n_trials)).T

    # =========================================================================
    # SEQUENTIAL PATH (n_threads == 1): Original algorithm
    # =========================================================================
    if n_threads == 1:
        for k in range(n_trials):
            boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

            # Precompute boundary evaluations
            boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
            compute_boundary(boundary,
                             t_s,
                             boundary_fun,
                             boundary_params_tmp)

            sqrt_st = delta_t_sqrt * s_view[k]

            # Loop over samples
            for n in range(n_samples):
                # displaced_starting_point
                y = (-1) * boundary_view[0] + ((z_view[k] + sz_samplewise_view[k, n]) * 2.0 * (boundary_view[0]))

                # displaced drift
                drift_increment = (v_view[k] + sv_samplewise_view[k, n]) * delta_t

                # displaced t
                t_tmp = t_view[k] + st_samplewise_view[k, n]
                deadline_tmp = compute_deadline_tmp(max_t, deadline_view[k], t_tmp)

                # increment m appropriately
                m += 1
                if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                t_particle = 0.0 # reset time
                ix = 0 # reset boundary index

                if n == 0:
                    if k == 0:
                        traj_view[0, 0] = y

                # Random walker
                while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t_particle <= deadline_tmp:
                    y += drift_increment + (sqrt_st * gaussian_values[m])
                    t_particle += delta_t
                    ix += 1
                    m += 1

                    if n == 0:
                        if k == 0:
                            traj_view[ix, 0] = y
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                smooth_u = compute_smooth_unif(smooth_unif, t_particle, deadline_tmp, delta_t)

                rts_view[n, k, 0] = t_particle + t_tmp + smooth_u # Store rt
                choices_view[n, k, 0] = np.sign(y) # Store choice

                enforce_deadline(rts_view, deadline_view, n, k, 0)

    # =========================================================================
    # PARALLEL PATH (n_threads > 1): FLATTENED OpenMP parallelization with C RNG
    # Parallelizes over (n_trials × n_samples) for optimal efficiency
    # =========================================================================
    else:
        # Precompute ALL trial data outside nogil
        boundaries_all_np = np.zeros((n_trials, num_steps), dtype=DTYPE)
        sqrt_st_all_np = np.zeros(n_trials, dtype=DTYPE)

        for k in range(n_trials):
            boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
            compute_boundary(boundary, t_s, boundary_fun, boundary_params_tmp)
            boundaries_all_np[k, :] = boundary
            sqrt_st_all_np[k] = delta_t_sqrt * s_view[k]

        boundaries_all = boundaries_all_np
        sqrt_st_all = sqrt_st_all_np

        # Total iterations = n_trials × n_samples
        total_iterations = <Py_ssize_t>n_trials * <Py_ssize_t>n_samples
        c_n_threads = n_threads

        # Allocate per-thread GSL RNGs BEFORE parallel block
        for i_thread in range(c_n_threads):
            rng_alloc(&rng_states[i_thread])

        with nogil, parallel(num_threads=n_threads):
            for flat_idx in prange(total_iterations, schedule='dynamic'):
                tid = threadid()

                # Compute (k, n) from flat index
                k = flat_idx // c_n_samples
                n = flat_idx % c_n_samples

                z_k = z_view[k]
                v_k = v_view[k]
                t_k = t_view[k]
                sqrt_st_k = sqrt_st_all[k]
                deadline_k = deadline_view[k]

                combined_seed = rng_mix_seed(seed, <uint64_t>k, <uint64_t>n)
                rng_seed(&rng_states[tid], combined_seed)

                # Get per-sample variability
                sz_n = sz_samplewise_view[k, n]
                sv_n = sv_samplewise_view[k, n]
                st_n = st_samplewise_view[k, n]

                bound_val = boundaries_all[k, 0]
                y = (-1.0) * bound_val + ((z_k + sz_n) * 2.0 * bound_val)

                drift_increment = (v_k + sv_n) * delta_t
                t_tmp = t_k + st_n

                # Compute deadline with variability
                deadline_tmp = fmin(c_max_t, deadline_k - t_tmp)
                if deadline_tmp < 0:
                    deadline_tmp = c_max_t

                t_particle = 0.0
                ix = 0

                while True:
                    bound_val = boundaries_all[k, ix]
                    neg_bound_val = -bound_val

                    if y < neg_bound_val or y > bound_val or t_particle > deadline_tmp:
                        break

                    noise = rng_gaussian_f32(&rng_states[tid])
                    y = y + drift_increment + sqrt_st_k * noise
                    t_particle = t_particle + delta_t
                    ix = ix + 1

                    if ix >= num_steps:
                        break

                rts_view[n, k, 0] = t_particle + t_tmp

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
        simulator_name='full_ddm_rv',
        possible_choices=[-1, 1],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )

    if return_option == 'full':
        sim_config = {'delta_t': delta_t, 'max_t': max_t}
        params = {
            'v': v, 'a': a, 'z': z, 't': t,
            'z_dist': z_dist, 'v_dist': v_dist, 't_dist': t_dist,
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

# -------------------------------------------------------------------------------------------------


# Simulate (rt, choice) tuples from: Full DDM with flexible bounds --------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def full_ddm(np.ndarray[float, ndim = 1] v, # = 0,
             np.ndarray[float, ndim = 1] a, # = 1,
             np.ndarray[float, ndim = 1] z, # = 0.5,
             np.ndarray[float, ndim = 1] t, # = 0.0,
             np.ndarray[float, ndim = 1] sz, # = 0.05,
             np.ndarray[float, ndim = 1] sv, # = 0.1,
             np.ndarray[float, ndim = 1] st, # = 0.0,
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
    Simulate reaction times and choices from a full drift diffusion model with flexible boundaries.

    Args:
        v (np.ndarray): Drift rate for each trial.
        a (np.ndarray): Boundary separation for each trial.
        z (np.ndarray): Starting point (between 0 and 1) for each trial.
        t (np.ndarray): Non-decision time for each trial.
        sz (np.ndarray): Variability in starting point for each trial.
        sv (np.ndarray): Variability in drift rate for each trial.
        st (np.ndarray): Variability in non-decision time for each trial.
        deadline (np.ndarray): Maximum reaction time allowed for each trial.
        s (np.ndarray): Noise standard deviation for each trial.
        delta_t (float): Time step size for simulation (default: 0.001).
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
        ValueError: If return_option is neither 'full' nor 'minimal'.
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
    t_s = setup['t_s']
    cdef int num_draws = setup['num_draws']
    cdef float delta_t_sqrt = setup['delta_t_sqrt']
    cdef int num_steps = int((max_t / delta_t) + 1)
    cdef float c_max_t = max_t

    # Param views
    cdef float[:] v_view  = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] sz_view = sz
    cdef float[:] sv_view = sv
    cdef float[:] st_view = st
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s

    # Boundary storage
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef float y, t_particle, t_tmp, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0
    cdef float drift_increment = 0.0

    # Variables for parallel execution - per-thread RNG states
    cdef float[:, :] boundaries_all
    cdef RngState[MAX_THREADS] rng_states
    cdef uint64_t combined_seed
    cdef float z_k, v_k, t_k, s_k, sz_k, sv_k, st_k, sqrt_st_k, deadline_k
    cdef float bound_val, neg_bound_val, noise, y_disp, v_disp, t_disp
    cdef int choice
    cdef int tid
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
            boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
            compute_boundary(boundary,
                             t_s,
                             boundary_fun,
                             boundary_params_tmp)

            deadline_tmp = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])
            sqrt_st = delta_t_sqrt * s_view[k]
            # Loop over samples
            for n in range(n_samples):
                # initialize starting point
                y = ((-1) * boundary_view[0]) + (z_view[k] * 2.0 * (boundary_view[0]))  # reset starting position

                # get drift by random displacement of v
                drift_increment = (v_view[k] + sv_view[k] * gaussian_values[m]) * delta_t
                t_tmp = t_view[k] + (2 * (random_uniform() - 0.5) * st_view[k])

                # apply uniform displacement on y
                y += 2 * (random_uniform() - 0.5) * sz_view[k]

                # increment m appropriately
                m += 1
                if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                t_particle = 0.0 # reset time
                ix = 0 # reset boundary index

                if n == 0:
                    if k == 0:
                        traj_view[0, 0] = y

                # Random walker
                while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t_particle <= deadline_tmp:
                    y += drift_increment + (sqrt_st * gaussian_values[m])
                    t_particle += delta_t
                    ix += 1
                    m += 1

                    if n == 0:
                        if k == 0:
                            traj_view[ix, 0] = y
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                smooth_u = compute_smooth_unif(smooth_unif, t_particle, deadline_tmp, delta_t)

                rts_view[n, k, 0] = t_particle + t_tmp + smooth_u # Store rt
                choices_view[n, k, 0] = np.sign(y) # Store choice

                enforce_deadline(rts_view, deadline_view, n, k, 0)

    # =========================================================================
    # PARALLEL PATH (n_threads > 1): FLATTENED OpenMP parallelization with C RNG
    # Parallelizes over (n_trials × n_samples) for optimal efficiency
    # =========================================================================
    else:
        # Precompute ALL trial data outside nogil
        boundaries_all_np = np.zeros((n_trials, num_steps), dtype=DTYPE)

        for k in range(n_trials):
            boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
            compute_boundary(boundary, t_s, boundary_fun, boundary_params_tmp)
            boundaries_all_np[k, :] = boundary

        boundaries_all = boundaries_all_np

        # Total iterations = n_trials × n_samples
        total_iterations = <Py_ssize_t>n_trials * <Py_ssize_t>n_samples
        c_n_threads = n_threads

        # Allocate per-thread GSL RNGs BEFORE parallel block
        for i_thread in range(c_n_threads):
            rng_alloc(&rng_states[i_thread])

        with nogil, parallel(num_threads=n_threads):
            for flat_idx in prange(total_iterations, schedule='dynamic'):
                tid = threadid()

                # Compute (k, n) from flat index
                k = flat_idx // c_n_samples
                n = flat_idx % c_n_samples

                z_k = z_view[k]
                v_k = v_view[k]
                t_k = t_view[k]
                s_k = s_view[k]
                sz_k = sz_view[k]
                sv_k = sv_view[k]
                st_k = st_view[k]
                sqrt_st_k = delta_t_sqrt * s_k
                deadline_k = deadline_view[k]

                combined_seed = rng_mix_seed(seed, <uint64_t>k, <uint64_t>n)
                rng_seed(&rng_states[tid], combined_seed)

                # Generate variability using C RNG
                # sv uses normal distribution, sz and st use uniform
                v_disp = sv_k * rng_gaussian_f32(&rng_states[tid])  # Normal for drift variability

                # Uniform for starting point and t variability: 2 * (U - 0.5) = range [-1, 1]
                y_disp = 2.0 * (<float>ssms_uniform(&rng_states[tid]) - 0.5) * sz_k
                t_disp = 2.0 * (<float>ssms_uniform(&rng_states[tid]) - 0.5) * st_k

                bound_val = boundaries_all[k, 0]
                y = (-1.0) * bound_val + (z_k * 2.0 * bound_val) + y_disp

                drift_increment = (v_k + v_disp) * delta_t
                t_tmp = t_k + t_disp

                # Compute deadline
                deadline_tmp = fmin(c_max_t, deadline_k - t_tmp)
                if deadline_tmp < 0:
                    deadline_tmp = c_max_t

                t_particle = 0.0
                ix = 0

                while True:
                    bound_val = boundaries_all[k, ix]
                    neg_bound_val = -bound_val

                    if y < neg_bound_val or y > bound_val or t_particle > deadline_tmp:
                        break

                    noise = rng_gaussian_f32(&rng_states[tid])
                    y = y + drift_increment + sqrt_st_k * noise
                    t_particle = t_particle + delta_t
                    ix = ix + 1

                    if ix >= num_steps:
                        break

                rts_view[n, k, 0] = t_particle + t_tmp

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
        simulator_name='full_ddm',
        possible_choices=[-1, 1],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )

    if return_option == 'full':
        sim_config = {'delta_t': delta_t, 'max_t': max_t}
        params = {
            'v': v, 'a': a, 'z': z, 't': t,
            'sz': sz, 'sv': sv, 'st': st,
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

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES AND FLEXIBLE SLOPE (TRIAL VARIABLE) ---
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_sdv(np.ndarray[float, ndim = 1] v,
            np.ndarray[float, ndim = 1] a,
            np.ndarray[float, ndim = 1] z,
            np.ndarray[float, ndim = 1] t,
            np.ndarray[float, ndim = 1] deadline,
            np.ndarray[float, ndim = 1] s, # noise sigma  (NOTE: this is trial dependent !)
            float max_t = 20,
            float delta_t = 0.001,
            int n_samples = 20000,
            int n_trials = 1,
            boundary_fun = None,
            boundary_params = {},
            random_state = None,
            return_option = 'full',
            smooth_unif = False,
            int n_threads = 1,
            **kwargs,
            ):
    """
    Simulate reaction times and choices from a drift diffusion model with flexible boundaries.

    Args:
        v (np.ndarray): Drift rate for each trial.
        z (np.ndarray): Starting point bias for each trial (between 0 and 1).
        t (np.ndarray): Non-decision time for each trial.
        deadline (np.ndarray): Deadline for each trial.
        s (np.ndarray): TRIAL DEPENDENT noise sigma for each trial.
        max_t (float): Maximum time for simulation.
        delta_t (float): Time step for simulation.
        n_samples (int): Number of samples to simulate.
        n_trials (int): Number of trials to simulate.
        boundary_fun (callable): Boundary function (function of t and parameters).
        boundary_params (dict): Parameters for boundary function.
        random_state (int): Random state for simulation.
        return_option (str): 'full' or 'minimal' return options.
        smooth_unif (bool): Whether to apply uniform smoothing.
        **kwargs: Additional keyword arguments.

    Returns:
        dict: Dictionary with 'rts', 'choices', and 'metadata'.
    """

    # Check if parallel execution is requested and available
    n_threads = check_parallel_request(n_threads)

    # Get seed for reproducibility
    cdef uint64_t seed = random_state if random_state is not None else np.random.randint(0, 2**31)

    # Param views
    cdef float[:] v_view = v
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s

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
    cdef int num_steps = int((max_t / delta_t) + 1)
    cdef float c_max_t = max_t

    # Boundary storage for the upper bound
    boundary = np.zeros(t_s.shape, dtype=DTYPE)
    cdef float[:] boundary_view = boundary

    cdef float y, t_particle, sqrt_st, smooth_u
    cdef int n, ix, k
    cdef int m = 0
    cdef float deadline_tmp = 0.0

    # Variables for parallel execution - per-thread RNG states
    cdef float[:, :] boundaries_all
    cdef float[:] deadlines_tmp_all
    cdef float[:] sqrt_st_all
    cdef float[:] drift_inc_all
    cdef RngState[MAX_THREADS] rng_states
    cdef uint64_t combined_seed
    cdef float z_k, v_k, t_k, s_k, sqrt_st_k, deadline_k
    cdef float bound_val, neg_bound_val, noise, drift_inc
    cdef int choice
    cdef int tid
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

            for n in range(n_samples):
                y = (-1) * boundary_view[0] + (z_view[k] * 2 * (boundary_view[0]))  # reset starting position
                t_particle = 0.0 # reset time
                ix = 0 # reset boundary index

                # Random walker
                while (y >= (-1) * boundary_view[ix]) and (y <= boundary_view[ix]) and (t_particle <= deadline_tmp):
                    y += (v_view[k] * delta_t) + (sqrt_st * draw_gaussian(1)[0])
                    t_particle += delta_t
                    ix += 1
                    if ix >= num_draws:
                        ix = num_draws - 1

                # Note the if here (need to do store choice and rt in case where need to store trajectory)
                if (k == 0):
                    if (n == 0):
                        traj_view[:ix, 0] = y

                smooth_u = compute_smooth_unif(smooth_unif, t_particle, deadline_tmp, delta_t)

                rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u # Store rt
                choices_view[n, k, 0] = sign(y) # Store choice

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
        drift_inc_all_np = np.zeros(n_trials, dtype=DTYPE)

        for k in range(n_trials):
            boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
            compute_boundary(boundary, t_s, boundary_fun, boundary_params_tmp)
            boundaries_all_np[k, :] = boundary
            deadlines_tmp_all_np[k] = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])
            sqrt_st_all_np[k] = delta_t_sqrt * s_view[k]
            drift_inc_all_np[k] = v_view[k] * delta_t

        boundaries_all = boundaries_all_np
        deadlines_tmp_all = deadlines_tmp_all_np
        sqrt_st_all = sqrt_st_all_np
        drift_inc_all = drift_inc_all_np

        # Total iterations = n_trials × n_samples
        total_iterations = <Py_ssize_t>n_trials * <Py_ssize_t>n_samples
        c_n_threads = n_threads

        # Allocate per-thread GSL RNGs BEFORE parallel block
        for i_thread in range(c_n_threads):
            rng_alloc(&rng_states[i_thread])

        with nogil, parallel(num_threads=n_threads):
            for flat_idx in prange(total_iterations, schedule='dynamic'):
                tid = threadid()

                # Compute (k, n) from flat index
                k = flat_idx // c_n_samples
                n = flat_idx % c_n_samples

                z_k = z_view[k]
                t_k = t_view[k]
                sqrt_st_k = sqrt_st_all[k]
                deadline_k = deadline_view[k]
                drift_inc = drift_inc_all[k]
                deadline_tmp = deadlines_tmp_all[k]

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
                    y = y + drift_inc + sqrt_st_k * noise
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
        simulator_name='ddm_sdv',
        possible_choices=[-1, 1],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )

    if return_option == 'full':
        sim_config = {'delta_t': delta_t, 'max_t': max_t}
        params = {
            'v': v, 'a': a, 'z': z, 't': t,
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

# -------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: Full DDM with flexible bounds --------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_sdv(np.ndarray[float, ndim = 1] v,
            np.ndarray[float, ndim = 1] a,
            np.ndarray[float, ndim = 1] z,
            np.ndarray[float, ndim = 1] t,
            np.ndarray[float, ndim = 1] sv,
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
    Simulate reaction times and choices from a drift diffusion model with flexible boundaries and inter-trial variability in drift rate.

    Args:
        v (np.ndarray): Drift rate for each trial.
        a (np.ndarray): Boundary separation for each trial.
        z (np.ndarray): Starting point (between 0 and 1) for each trial.
        t (np.ndarray): Non-decision time for each trial.
        sv (np.ndarray): Standard deviation of drift rate for each trial.
        deadline (np.ndarray): Maximum reaction time allowed for each trial.
        s (np.ndarray): Noise standard deviation for each trial.
        delta_t (float): Time step size for simulation (default: 0.001).
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
        ValueError: If return_option is neither 'full' nor 'minimal'.
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
    t_s = setup['t_s']
    cdef int num_draws = setup['num_draws']
    cdef float delta_t_sqrt = setup['delta_t_sqrt']
    cdef int num_steps = int((max_t / delta_t) + 1)
    cdef float c_max_t = max_t

    # Param views
    cdef float[:] v_view  = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] sv_view = sv
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s

    # Boundary storage
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef float y, t_particle, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0
    cdef float drift_increment = 0.0

    # Variables for parallel execution - per-thread RNG states
    cdef float[:, :] boundaries_all
    cdef float[:] deadlines_tmp_all
    cdef float[:] sqrt_st_all
    cdef RngState[MAX_THREADS] rng_states
    cdef uint64_t combined_seed
    cdef float z_k, v_k, t_k, s_k, sv_k, sqrt_st_k, deadline_k
    cdef float bound_val, neg_bound_val, noise, v_disp
    cdef int choice
    cdef int tid
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
            compute_boundary(boundary,
                             t_s,
                             boundary_fun,
                             boundary_params_tmp)

            deadline_tmp = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])
            sqrt_st = delta_t_sqrt * s_view[k]
            # Loop over samples
            for n in range(n_samples):
                # initialize starting point
                y = ((-1) * boundary_view[0]) + (z_view[k] * 2.0 * (boundary_view[0]))  # reset starting position

                # get drift by random displacement of v
                drift_increment = (v_view[k] + sv_view[k] * gaussian_values[m]) * delta_t

                # increment m appropriately
                m += 1
                if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                t_particle = 0.0 # reset time
                ix = 0 # reset boundary index

                if n == 0:
                    if k == 0:
                        traj_view[0, 0] = y

                # Random walker
                while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t_particle <= deadline_tmp:
                    y += drift_increment + (sqrt_st * gaussian_values[m])
                    t_particle += delta_t
                    ix += 1
                    m += 1

                    if n == 0:
                        if k == 0:
                            traj_view[ix, 0] = y

                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                smooth_u = compute_smooth_unif(smooth_unif, t_particle, deadline_tmp, delta_t)

                rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u # Store rt
                choices_view[n, k, 0] = np.sign(y) # Store choice

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
                tid = threadid()

                # Compute (k, n) from flat index
                k = flat_idx // c_n_samples
                n = flat_idx % c_n_samples

                z_k = z_view[k]
                v_k = v_view[k]
                t_k = t_view[k]
                sv_k = sv_view[k]
                sqrt_st_k = sqrt_st_all[k]
                deadline_k = deadline_view[k]
                deadline_tmp = deadlines_tmp_all[k]

                combined_seed = rng_mix_seed(seed, <uint64_t>k, <uint64_t>n)
                rng_seed(&rng_states[tid], combined_seed)

                # Generate drift variability
                v_disp = sv_k * rng_gaussian_f32(&rng_states[tid])
                drift_increment = (v_k + v_disp) * delta_t

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
                    y = y + drift_increment + sqrt_st_k * noise
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
        simulator_name='ddm_sdv',
        possible_choices=[-1, 1],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )

    if return_option == 'full':
        sim_config = {'delta_t': delta_t, 'max_t': max_t}
        params = {
            'v': v, 'a': a, 'z': z, 't': t,
            'sv': sv, 'deadline': deadline, 's': s
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

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES, FLEXIBLE SLOPE, AND DECAY ----------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flexbound_tradeoff(np.ndarray[float, ndim = 1] vh,
                           np.ndarray[float, ndim = 1] vl1,
                           np.ndarray[float, ndim = 1] vl2,
                           np.ndarray[float, ndim = 1] a,
                           np.ndarray[float, ndim = 1] zh,
                           np.ndarray[float, ndim = 1] zl1,
                           np.ndarray[float, ndim = 1] zl2,
                           np.ndarray[float, ndim = 1] d, # d for 'dampen' effect on drift parameter
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
    Simulate a Drift Diffusion Model (DDM) with flexible boundaries for a tradeoff scenario.

    This function simulates a two-stage decision process where the first stage (high-dimensional)
    influences the second stage (low-dimensional) through a bias trace.

    Parameters:
    -----------
    vh, vl1, vl2 : np.ndarray[float, ndim=1]
        Drift rates for high-dimensional and two low-dimensional choices.
    a : np.ndarray[float, ndim=1]
        Initial boundary separation.
    zh, zl1, zl2 : np.ndarray[float, ndim=1]
        Starting points for high-dimensional and two low-dimensional choices.
    d : np.ndarray[float, ndim=1]
        Damping factor for drift rate.
    t : np.ndarray[float, ndim=1]
        Non-decision time.
    deadline : np.ndarray[float, ndim=1]
        Time limit for each trial.
    s : np.ndarray[float, ndim=1]
        Noise standard deviation.
    delta_t : float, optional
        Size of time steps (default: 0.001).
    max_t : float, optional
        Maximum time for a trial (default: 20).
    n_samples : int, optional
        Number of samples to simulate (default: 20000).
    n_trials : int, optional
        Number of trials to simulate (default: 1).
    print_info : bool, optional
        Whether to print simulation information (default: True).
    boundary_fun : callable, optional
        Function defining the decision boundary over time.
    boundary_params : dict, optional
        Parameters for the boundary function.
    random_state : int or None, optional
        Seed for random number generation (default: None).
    return_option : str, optional
        Determines the format of returned data ('full' or 'minimal', default: 'full').
    smooth_unif : bool, optional
        Whether to use smooth uniform distribution for small time increments (default: False).

    Returns:
    --------
    dict
        A dictionary containing simulated reaction times, choices, and metadata.
        The exact contents depend on the 'return_option' parameter.

    Raises:
    -------
    ValueError
        If an invalid 'return_option' is provided.

    Notes:
    ------
    This function implements a complex DDM with flexible boundaries and a two-stage
    decision process, suitable for modeling tradeoff scenarios in decision-making.
    """

    # Note: This complex two-stage model does not support parallel execution
    # due to sequential dependencies (bias_trace). n_threads parameter is accepted
    # for API consistency but only n_threads=1 is supported.
    if n_threads > 1:
        import warnings
        warnings.warn(
            "ddm_flexbound_tradeoff does not support parallel execution. "
            "Running with n_threads=1.",
            UserWarning
        )

    setup = setup_simulation(n_samples, n_trials, max_t, delta_t, random_state)

    # Extract setup (no traj for this model)
    rts = setup['rts']
    choices = setup['choices']
    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices
    cdef float[:] gaussian_values = setup['gaussian_values']
    t_s = setup['t_s']
    cdef int num_draws = setup['num_draws']
    cdef float delta_t_sqrt = setup['delta_t_sqrt']

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
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s

    # Boundary storage
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    # Y particle trace (specific to this model)
    bias_trace = np.zeros(num_draws, dtype = DTYPE)
    cdef float[:] bias_trace_view = bias_trace

    cdef float y_h, y_l, v_l, t_h, t_l, tmp_pos_dep, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n, ix, ix_tmp, k
    cdef Py_ssize_t m = 0

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
        compute_boundary(boundary,
                         t_s,
                         boundary_fun,
                         boundary_params_tmp)

        deadline_tmp = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])
        sqrt_st = delta_t_sqrt * s_view[k]
        # Loop over samples
        for n in range(n_samples):
            choices_view[n, k, 0] = 0 # reset choice
            t_h = 0 # reset time high dimension
            t_l = 0 # reset time low dimension
            ix = 0 # reset boundary index

            # Initialize walkers
            y_h = (-1) * boundary_view[0] + (zh_view[k] * 2 * (boundary_view[0]))
            bias_trace_view[0] = ((y_h + boundary_view[0]) / (2 * boundary_view[0]))

            # Random walks until y_h hits bound
            while (y_h >= ((-1) * boundary_view[ix])) and ((y_h <= boundary_view[ix])) and (t_h <= deadline_tmp):
                y_h += (vh_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                bias_trace_view[ix] = ((y_h + boundary_view[ix]) / (2 * boundary_view[ix]))
                t_h += delta_t
                ix += 1
                m += 1
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add 2 deterministically
            # y at lower bound --> choice_view[n, k, 0] stay the same deterministically
            if random_uniform() <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 2

            if choices_view[n, k, 0] == 2:
                y_l = (- 1) * boundary_view[0] + (zl2_view[k] * 2 * (boundary_view[0]))
                v_l = vl2_view[k]

                # Fill bias trace until max_rt reached
                ix_tmp = ix + 1
                while ix_tmp < num_draws:
                    bias_trace_view[ix_tmp] = 1.0
                    ix_tmp += 1

            else: # Store intermediate choice
                y_l = (- 1) * boundary_view[0] + (zl1_view[k] * 2 * (boundary_view[0]))
                v_l = vl1_view[k]

                # Fill bias trace until max_rt reached
                ix_tmp = ix + 1
                while ix_tmp < num_draws:
                    bias_trace_view[ix_tmp] = 0.0
                    ix_tmp += 1

                #We need to reverse the bias_trace if we took the lower choice
                ix_tmp = 0
                while ix_tmp < num_draws:
                    bias_trace_view[ix_tmp] = 1.0 - bias_trace_view[ix_tmp]
                    ix_tmp += 1

            # Random walks until the y_l corresponding to y_h hits bound
            ix = 0
            while (y_l >= ((-1) * boundary_view[ix])) and (y_l <= boundary_view[ix]) and (t_l <= deadline_tmp):
                # Compute local position dependence
                # AF-todo: can't understand what the idea here is anymore
                # especially why bias_trace_view is flipped (-1) here
                tmp_pos_dep = (1 + (d_view[k] * (bias_trace_view[ix] - 1.0))) / (2 - d_view[k])

                if (bias_trace_view[ix] < 1) and (bias_trace_view[ix] > 0):
                    # Before high-dim choice is taken
                    y_l += tmp_pos_dep * (v_l * delta_t) # Add drift
                    y_l += tmp_pos_dep * sqrt_st * gaussian_values[m] # Add noise
                else:
                    # After high-dim choice is taken
                    y_l += (v_l * delta_t) # Add drift
                    y_l += sqrt_st * gaussian_values[m] # Add noise

                t_l += delta_t # update time for low_dim choice
                ix += 1 # update time index
                m += 1 # update rv couter

                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            smooth_u = compute_smooth_unif(smooth_unif, fmax(t_h, t_l), deadline_tmp, delta_t)

            rts_view[n, k, 0] = fmax(t_h, t_l) + t_view[k]

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add one deterministically
            # y at lower bound --> choice_view[n, k, 0] stays the same deterministically
            if random_uniform() <= ((y_l + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 1

            enforce_deadline(rts_view, deadline_view, n, k, 0)

    # Build minimal metadata first
    minimal_meta = build_minimal_metadata(
        simulator_name='ddm_flexbound_mic2_adj',
        possible_choices=[0, 1, 2, 3],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )

    if return_option == 'full':
        sim_config = {'delta_t': delta_t, 'max_t': max_t}
        params = {
            'vh': vh, 'vl1': vl1, 'vl2': vl2,
            'a': a, 'zh': zh, 'zl1': zl1, 'zl2': zl2,
            'd': d, 't': t, 'deadline': deadline, 's': s
        }
        # Add special trajectory note for this model
        extra_params_dict = {
            'trajectory': 'This simulator does not yet allow for trajectory simulation'
        }
        full_meta = build_full_metadata(
            minimal_metadata=minimal_meta,
            params=params,
            sim_config=sim_config,
            boundary_fun=boundary_fun,
            boundary=boundary,
            boundary_params=boundary_params,
            extra_params=extra_params_dict
        )
        return build_return_dict(rts, choices, full_meta)

    elif return_option == 'minimal':
        return build_return_dict(rts, choices, minimal_meta)

    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# -----------------------------------------------------------------------------------------------


# ================================================================================================
# PROOF OF CONCEPT: FLATTENED PARALLELIZATION
# ================================================================================================
# This function demonstrates parallelization over the flattened (n_trials × n_samples) space,
# which works optimally regardless of whether n_trials >> n_samples or n_samples >> n_trials.
# ================================================================================================

def ddm_flexbound_flat(np.ndarray[float, ndim = 1] v,
                       np.ndarray[float, ndim = 1] a,
                       np.ndarray[float, ndim = 1] z,
                       np.ndarray[float, ndim = 1] t,
                       np.ndarray[float, ndim = 1] deadline,
                       np.ndarray[float, ndim = 1] s,
                       float max_t = 20,
                       float delta_t = 0.001,
                       int n_samples = 20000,
                       int n_trials = 1,
                       boundary_fun = None,
                       boundary_params = {},
                       random_state = None,
                       return_option = 'full',
                       smooth_unif = False,
                       int n_threads = 1,
                       **kwargs,
                       ):
    """
    DDM with flexible boundaries using FLATTENED parallelization.

    This is a proof-of-concept implementation that parallelizes over the combined
    (n_trials × n_samples) space rather than just n_trials. This ensures good
    parallel efficiency regardless of whether n_trials >> n_samples or vice versa.

    Key differences from ddm_flexbound:
    - Parallel loop is over total_iterations = n_trials × n_samples
    - Each thread computes its own (k, n) indices from flat_idx
    - Works efficiently even when n_trials=1 and n_samples=100000

    Args:
        v, a, z, t, deadline, s: Parameter arrays (one value per trial)
        max_t, delta_t: Simulation time parameters
        n_samples: Number of samples per trial
        n_trials: Number of trials
        boundary_fun: Boundary function
        boundary_params: Boundary parameters
        random_state: Random seed
        return_option: 'full' or 'minimal'
        smooth_unif: Whether to apply uniform smoothing
        n_threads: Number of threads (1 = sequential, >1 = parallel)

    Returns:
        dict: Simulated reaction times, choices, and metadata
    """
    # Check if parallel execution is requested and available
    n_threads = check_parallel_request(n_threads)

    # Get seed for reproducibility
    cdef uint64_t seed = random_state if random_state is not None else np.random.randint(0, 2**31)

    setup = setup_simulation(n_samples, n_trials, max_t, delta_t, random_state)

    # Extract arrays and create memory views
    traj = setup['traj']
    rts = setup['rts']
    choices = setup['choices']
    cdef float[:, :] traj_view = traj
    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices
    cdef float[:] gaussian_values = setup['gaussian_values']
    t_s = setup['t_s']
    cdef int num_draws = setup['num_draws']
    cdef float delta_t_sqrt = setup['delta_t_sqrt']

    # Param views
    cdef float[:] v_view = v
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s

    # Boundary storage
    boundary = np.zeros(t_s.shape, dtype=DTYPE)
    cdef float[:] boundary_view = boundary

    # Number of timesteps
    cdef int num_steps = int((max_t / delta_t) + 1)

    cdef float y, t_particle, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0

    # Variables for parallel execution - per-thread RNG states
    cdef float[:, :] boundaries_all       # 2D: [n_trials, num_steps]
    cdef float[:] deadlines_tmp           # 1D: [n_trials]
    cdef float[:] sqrt_st_all             # 1D: [n_trials] - precomputed sqrt_st per trial
    cdef float[:] drift_inc_all           # 1D: [n_trials] - precomputed drift increment per trial
    cdef RngState[MAX_THREADS] rng_states
    cdef uint64_t combined_seed
    cdef float v_k, z_k, t_k, s_k, sqrt_st_k, deadline_tmp_k
    cdef float drift_inc, noise, bound_val, neg_bound_val
    cdef int choice
    cdef int tid
    cdef int i_thread
    cdef int c_n_threads

    # Flattened parallelization variables
    cdef Py_ssize_t flat_idx
    cdef Py_ssize_t total_iterations
    cdef int c_n_samples = n_samples  # C-typed for nogil division

    # =========================================================================
    # SEQUENTIAL PATH (n_threads == 1): Original algorithm
    # =========================================================================
    if n_threads == 1:
        for k in range(n_trials):
            boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
            compute_boundary(boundary, t_s, boundary_fun, boundary_params_tmp)

            deadline_tmp = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])
            sqrt_st = delta_t_sqrt * s_view[k]

            for n in range(n_samples):
                y = (-1) * boundary_view[0] + (z_view[k] * 2 * (boundary_view[0]))
                t_particle = 0.0
                ix = 0

                if n == 0 and k == 0:
                    traj_view[0, 0] = y

                while (y >= (-1) * boundary_view[ix]) and (y <= boundary_view[ix]) and (t_particle <= deadline_tmp):
                    y += (v_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                    t_particle += delta_t
                    ix += 1
                    m += 1

                    if n == 0 and k == 0:
                        traj_view[ix, 0] = y

                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                smooth_u = compute_smooth_unif(smooth_unif, t_particle, deadline_tmp, delta_t)
                rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u
                choices_view[n, k, 0] = sign(y)
                enforce_deadline(rts_view, deadline_view, n, k, 0)

    # =========================================================================
    # PARALLEL PATH (n_threads > 1): FLATTENED OpenMP parallelization
    # =========================================================================
    else:
        # Pre-compute ALL trial-specific data outside nogil
        boundaries_all_np = np.zeros((n_trials, num_steps), dtype=DTYPE)
        deadlines_tmp_np = np.zeros(n_trials, dtype=DTYPE)
        sqrt_st_all_np = np.zeros(n_trials, dtype=DTYPE)
        drift_inc_all_np = np.zeros(n_trials, dtype=DTYPE)

        for k in range(n_trials):
            boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
            compute_boundary(boundary, t_s, boundary_fun, boundary_params_tmp)
            boundaries_all_np[k, :] = boundary
            deadlines_tmp_np[k] = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])
            sqrt_st_all_np[k] = delta_t_sqrt * s_view[k]
            drift_inc_all_np[k] = v_view[k] * delta_t

        boundaries_all = boundaries_all_np
        deadlines_tmp = deadlines_tmp_np
        sqrt_st_all = sqrt_st_all_np
        drift_inc_all = drift_inc_all_np

        # Total iterations = n_trials × n_samples
        total_iterations = <Py_ssize_t>n_trials * <Py_ssize_t>n_samples
        c_n_threads = n_threads

        # Allocate per-thread GSL RNGs BEFORE parallel block
        for i_thread in range(c_n_threads):
            rng_alloc(&rng_states[i_thread])

        # Parallel execution over FLATTENED iteration space
        with nogil, parallel(num_threads=n_threads):
            for flat_idx in prange(total_iterations, schedule='dynamic'):
                tid = threadid()

                # Compute (k, n) from flat index
                k = flat_idx // c_n_samples   # trial index
                n = flat_idx % c_n_samples    # sample index

                # Access pre-computed trial parameters
                z_k = z_view[k]
                t_k = t_view[k]
                sqrt_st_k = sqrt_st_all[k]
                deadline_tmp_k = deadlines_tmp[k]
                drift_inc = drift_inc_all[k]

                # Re-seed per-thread RNG with unique seed for this (trial, sample)
                combined_seed = rng_mix_seed(seed, <uint64_t>k, <uint64_t>n)
                rng_seed(&rng_states[tid], combined_seed)

                # Initialize particle position
                bound_val = boundaries_all[k, 0]
                y = (-1.0) * bound_val + (z_k * 2.0 * bound_val)
                t_particle = 0.0
                ix = 0

                # Random walk
                while True:
                    bound_val = boundaries_all[k, ix]
                    neg_bound_val = -bound_val

                    if y < neg_bound_val or y > bound_val or t_particle > deadline_tmp_k:
                        break

                    noise = rng_gaussian_f32(&rng_states[tid])
                    y = y + drift_inc + sqrt_st_k * noise
                    t_particle = t_particle + delta_t
                    ix = ix + 1

                    if ix >= num_steps:
                        break

                # Store results
                rts_view[n, k, 0] = t_particle + t_k

                # Choice based on sign of y (same as sequential path)
                if y > 0.0:
                    choice = 1
                elif y < 0.0:
                    choice = -1
                else:
                    choice = 0
                choices_view[n, k, 0] = choice

                # Deadline enforcement
                if rts_view[n, k, 0] >= deadline_view[k] or deadline_view[k] <= 0:
                    rts_view[n, k, 0] = -999.0

        # Free per-thread GSL RNGs AFTER parallel block
        for i_thread in range(c_n_threads):
            rng_free(&rng_states[i_thread])

    # Build metadata
    minimal_meta = build_minimal_metadata(
        simulator_name='ddm_flexbound_flat',
        possible_choices=[-1, 1],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__ if boundary_fun else 'constant'
    )

    if return_option == 'full':
        sim_config = {'delta_t': delta_t, 'max_t': max_t}
        params = {
            'v': v, 'a': a, 'z': z, 't': t,
            'deadline': deadline, 's': s
        }
        full_meta = build_full_metadata(
            minimal_metadata=minimal_meta,
            params=params,
            sim_config=sim_config,
            boundary_fun=boundary_fun,
            boundary=boundary,
            boundary_params=boundary_params,
            traj=traj
        )
        return build_return_dict(rts, choices, full_meta)

    elif return_option == 'minimal':
        return build_return_dict(rts, choices, minimal_meta)

    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# -----------------------------------------------------------------------------------------------
