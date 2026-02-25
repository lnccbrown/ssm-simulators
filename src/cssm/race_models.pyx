# Global settings for cython
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: initializedcheck=False

"""
Race Models

This module contains simulator functions for race models, where multiple
accumulators race independently toward their own decision boundaries.
Unlike DDM, race models have separate evidence accumulators for each choice.
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
    csum,
    compute_boundary,
    compute_smooth_unif,
    enforce_deadline,
    compute_deadline_tmp,
    build_param_dict_from_2d_array,
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

# Maximum particles/accumulators for stack-allocated arrays
DEF MAX_PARTICLES = 16

# Include shared constants (MAX_THREADS, etc.)
include "_constants.pxi"

# C-level helper functions for nogil operation ------------------------------------

cdef inline bint check_finished(float[:] particles, float boundary, int n) noexcept:
    """
    Check if any particle has crossed the boundary.

    Args:
        particles (float[:]): Array of particle positions.
        boundary (float): Boundary value to check against.
        n (int): Number of particles.

    Returns:
        bool: True if any particle has crossed the boundary, False otherwise.
    """
    cdef int i
    for i in range(n):
        if particles[i] > boundary:
            return True
    return False


cdef inline bint check_finished_stack(float* particles, float boundary, int n) noexcept nogil:
    """
    Check if any particle has crossed the boundary (stack-allocated version).

    Args:
        particles: Pointer to stack-allocated particle array.
        boundary: Boundary value to check against.
        n: Number of particles.

    Returns:
        True if any particle has crossed the boundary, False otherwise.
    """
    cdef int i
    for i in range(n):
        if particles[i] > boundary:
            return True
    return False


cdef inline int argmax_stack(float* arr, int n) noexcept nogil:
    """
    Find the index of the maximum element (stack-allocated version).

    Args:
        arr: Pointer to float array.
        n: Number of elements.

    Returns:
        Index of the maximum element.
    """
    cdef int best_idx = 0
    cdef float best_val = arr[0]
    cdef int i
    for i in range(1, n):
        if arr[i] > best_val:
            best_val = arr[i]
            best_idx = i
    return best_idx


cdef inline float csum_stack(float* arr, int n) noexcept nogil:
    """
    Compute sum of array elements (stack-allocated version).

    Args:
        arr: Pointer to float array.
        n: Number of elements.

    Returns:
        Sum of all elements.
    """
    cdef float total = 0.0
    cdef int i
    for i in range(n):
        total += arr[i]
    return total


# Race Model ------------------------------------

def race_model(np.ndarray[float, ndim = 2] v,  # np.array expected, one column of floats
               np.ndarray[float, ndim = 2] z, # np.array expected, one column of floats
               np.ndarray[float, ndim = 2] t, # for now we we don't allow t by choice
               np.ndarray[float, ndim = 2] s, # np.array expected, one column of floats
               np.ndarray[float, ndim = 1] deadline,
               float delta_t = 0.001, # time increment step
               float max_t = 20, # maximum rt allowed
               int n_samples = 2000,
               int n_trials = 1,
               boundary_fun = None,
               boundary_params = {},
               random_state = None,
               return_option = 'full',
               smooth_unif = False,
               int n_threads = 1,
               **kwargs):
    """
    Simulate reaction times and choices from a race model with N samples.

    Args:
        v (np.ndarray): Drift rates for each accumulator and trial.
        z (np.ndarray): Starting points for each accumulator and trial.
        t (np.ndarray): Non-decision time for each trial.
        s (np.ndarray): Noise standard deviation for each accumulator and trial.
        deadline (np.ndarray): Maximum reaction time allowed for each trial.
        delta_t (float): Time increment step for simulation (default: 0.001).
        max_t (float): Maximum time for simulation (default: 20).
        n_samples (int): Number of samples to simulate per trial (default: 2000).
        n_trials (int): Number of trials to simulate (default: 1).
        boundary_fun (callable): Function defining the shape of the boundary over time.
        boundary_params (dict): Parameters for the boundary function.
        random_state (int or None): Seed for random number generator (default: None).
        return_option (str): 'full' for complete output, 'minimal' for basic output (default: 'full').
        smooth_unif (bool): Whether to apply uniform smoothing to reaction times (default: False).
        n_threads (int): Number of threads for parallel execution (default: 1).
        **kwargs: Additional keyword arguments.

    Returns:
        dict: A dictionary containing simulated reaction times, choices, and metadata.
              The exact contents depend on the return_option.

    Raises:
        ValueError: If return_option is not 'full' or 'minimal'.
        ValueError: If n_particles > MAX_PARTICLES (16) for parallel execution.
    """
    cdef int n_particles = v.shape[1]

    # Check particle count limit for parallel execution
    if n_threads > 1 and n_particles > MAX_PARTICLES:
        raise ValueError(
            f"race_model parallel execution requires n_particles <= {MAX_PARTICLES}, "
            f"got {n_particles}. Use n_threads=1 for larger particle counts."
        )

    # Check OpenMP availability for parallel execution
    if n_threads > 1:
        from cssm._openmp_status import check_parallel_request
        n_threads = check_parallel_request(n_threads)

    # Sequential path (n_threads=1)
    if n_threads == 1:
        return _race_model_sequential(
            v, z, t, s, deadline, delta_t, max_t, n_samples, n_trials,
            boundary_fun, boundary_params, random_state, return_option, smooth_unif
        )

    # Parallel path
    # Param views
    cdef float[:, :] v_view = v
    cdef float[:, :] z_view = z
    cdef float[:, :] t_view = t
    cdef float[:, :] s_view = s
    cdef float[:] deadline_view = deadline

    cdef float delta_t_sqrt = sqrt(delta_t)
    sqrt_st = delta_t_sqrt * s
    cdef float[:, :] sqrt_st_view = sqrt_st

    rts = np.zeros((n_samples, n_trials, 1), dtype=DTYPE)
    cdef float[:, :, :] rts_view = rts
    choices = np.zeros((n_samples, n_trials, 1), dtype=np.intc)
    cdef int[:, :, :] choices_view = choices

    # Trajectory storage - disabled in parallel mode
    traj = np.zeros((int(max_t / delta_t) + 1, n_particles), dtype=DTYPE)
    traj[:, :] = -999
    cdef float[:, :] traj_view = traj

    cdef int num_steps = int((max_t / delta_t) + 1)
    cdef float c_max_t = max_t
    cdef int c_n_samples = n_samples
    cdef int c_n_particles = n_particles

    # Pre-compute boundaries for all trials (outside nogil)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundaries_all_np = np.zeros((n_trials, len(t_s)), dtype=DTYPE)
    deadlines_tmp = np.zeros(n_trials, dtype=DTYPE)

    cdef Py_ssize_t k_precomp
    for k_precomp in range(n_trials):
        boundary_params_tmp = {key: boundary_params[key][k_precomp] for key in boundary_params.keys()}
        boundary_tmp = np.zeros(t_s.shape, dtype=DTYPE)
        compute_boundary(boundary_tmp, t_s, boundary_fun, boundary_params_tmp)
        boundaries_all_np[k_precomp, :] = boundary_tmp
        deadlines_tmp[k_precomp] = compute_deadline_tmp(max_t, deadline_view[k_precomp], t_view[k_precomp, 0])

    cdef float[:, :] boundaries_view = boundaries_all_np
    cdef float[:] deadlines_view = deadlines_tmp

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
    cdef int ix, j
    cdef float t_particle, deadline_tmp_k, bound_val, noise
    # Per-thread particle arrays to avoid race conditions
    cdef float particles[MAX_THREADS][MAX_PARTICLES]

    # Allocate per-thread GSL RNGs BEFORE parallel block
    for i_thread in range(c_n_threads):
        rng_alloc(&rng_states[i_thread])

    # Parallel execution over FLATTENED iteration space
    with nogil, parallel(num_threads=n_threads):
        for flat_idx in prange(total_iterations, schedule='dynamic'):
            # Get thread ID for per-thread RNG and particle array
            tid = threadid()

            k = flat_idx // c_n_samples  # trial index
            n = flat_idx % c_n_samples   # sample index

            # Re-seed per-thread RNG with unique seed for this (trial, sample)
            combined_seed = rng_mix_seed(base_seed, <uint64_t>k, <uint64_t>n)
            rng_seed(&rng_states[tid], combined_seed)

            deadline_tmp_k = deadlines_view[k]
            bound_val = boundaries_view[k, 0]

            # Initialize particle positions (using thread-local array)
            for j in range(c_n_particles):
                particles[tid][j] = z_view[k, j] * bound_val

            t_particle = 0.0
            ix = 0

            # Race simulation
            while True:
                bound_val = boundaries_view[k, ix]
                if check_finished_stack(&particles[tid][0], bound_val, c_n_particles) or t_particle > deadline_tmp_k:
                    break

                for j in range(c_n_particles):
                    noise = ssms_gaussian_f32(&rng_states[tid])
                    particles[tid][j] = particles[tid][j] + (v_view[k, j] * delta_t) + sqrt_st_view[k, j] * noise
                    particles[tid][j] = fmax(0.0, particles[tid][j])  # Cut off at 0

                t_particle = t_particle + delta_t
                ix = ix + 1
                if ix >= num_steps:
                    break

            rts_view[n, k, 0] = t_particle + t_view[k, 0]
            choices_view[n, k, 0] = argmax_stack(&particles[tid][0], c_n_particles)

            # Enforce deadline
            if rts_view[n, k, 0] > deadline_view[k]:
                rts_view[n, k, 0] = -999.0

    # Free per-thread GSL RNGs AFTER parallel block
    for i_thread in range(c_n_threads):
        rng_free(&rng_states[i_thread])

    # Build minimal metadata first
    minimal_meta = build_minimal_metadata(
        simulator_name='race_model',
        possible_choices=[-1, 1],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )

    if return_option == 'full':
        # Build v_dict and z_dict dynamically
        v_dict = build_param_dict_from_2d_array(v, 'v', n_particles)
        z_dict = build_param_dict_from_2d_array(z, 'z', n_particles)

        # Update possible_choices for full (n_particles-specific)
        minimal_meta['possible_choices'] = list(np.arange(0, n_particles, 1))

        sim_config = {'delta_t': delta_t, 'max_t': max_t, 'n_threads': n_threads}
        params = {'v': v, 'z': z, 't': t, 'deadline': deadline, 's': s}
        full_meta = build_full_metadata(
            minimal_metadata=minimal_meta,
            params=params,
            sim_config=sim_config,
            boundary_fun=boundary_fun,
            boundary=boundaries_all_np[0] if n_trials > 0 else np.array([]),
            traj=traj,
            boundary_params=boundary_params,
            extra_params={**v_dict, **z_dict}
        )
        return build_return_dict(rts, choices, full_meta)

    elif return_option == 'minimal':
        return build_return_dict(rts, choices, minimal_meta)

    else:
        raise ValueError('return_option must be either "full" or "minimal"')


def _race_model_sequential(
    np.ndarray[float, ndim = 2] v,
    np.ndarray[float, ndim = 2] z,
    np.ndarray[float, ndim = 2] t,
    np.ndarray[float, ndim = 2] s,
    np.ndarray[float, ndim = 1] deadline,
    float delta_t,
    float max_t,
    int n_samples,
    int n_trials,
    boundary_fun,
    boundary_params,
    random_state,
    return_option,
    smooth_unif
):
    """Sequential implementation of race_model (original code path)."""

    set_seed(random_state)
    # Param views
    cdef float[:, :] v_view = v
    cdef float[:, :] z_view = z
    cdef float[:, :] t_view = t
    cdef float[:, :] s_view = s
    cdef float[:] deadline_view = deadline

    cdef float delta_t_sqrt = sqrt(delta_t)
    sqrt_st = delta_t_sqrt * s
    cdef float[:, :] sqrt_st_view = sqrt_st

    cdef int n_particles = v.shape[1]
    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    cdef float[:, :, :] rts_view = rts
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    cdef int[:, :, :] choices_view = choices

    particles = np.zeros((n_particles), dtype = DTYPE)
    cdef float [:] particles_view = particles

    # TD: Add Trajectory
    traj = np.zeros((int(max_t / delta_t) + 1, n_particles), dtype = DTYPE)
    traj[:, :] = -999
    cdef float[:, :] traj_view = traj

    # Boundary storage
    cdef int num_steps = int((max_t / delta_t) + 1)

    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    # Initialize variables needed for for loop
    cdef float t_particle, smooth_u, deadline_tmp
    cdef Py_ssize_t n, ix, j, k
    cdef Py_ssize_t m = 0

    cdef int num_draws = num_steps * n_particles
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    cdef Py_ssize_t mu = 0
    cdef float[:] uniform_values = draw_uniform(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        compute_boundary(boundary, t_s, boundary_fun,
                        boundary_params_tmp)

        deadline_tmp = compute_deadline_tmp(max_t, deadline_view[k], t_view[k, 0])
        # Loop over samples
        for n in range(n_samples):
            for j in range(n_particles):
                particles_view[j] = z_view[k, j] * boundary_view[0] # Reset particle starting points

            t_particle = 0.0 # reset time
            ix = 0

            if n == 0:
                if k == 0:
                    for j in range(n_particles):
                        traj_view[0, j] = particles[j]

            # Random walker
            while not check_finished(particles_view, boundary_view[ix], n_particles) and t_particle <= deadline_tmp:
                for j in range(n_particles):
                    particles_view[j] += (v_view[k, j] * delta_t) + sqrt_st_view[k, j] * gaussian_values[m]
                    particles_view[j] = fmax(0.0, particles_view[j]) # Cut off particles at 0
                    m += 1
                    if m == num_draws:
                        m = 0
                        gaussian_values = draw_gaussian(num_draws)
                t_particle += delta_t
                ix += 1
                if n == 0:
                    if k == 0:
                        for j in range(n_particles):
                            traj_view[ix, j] = particles[j]

            smooth_u = compute_smooth_unif(smooth_unif, t_particle, deadline_tmp, delta_t, uniform_values[mu])
            mu += 1
            if mu == num_draws:
                uniform_values = draw_uniform(num_draws)
                mu = 0

            rts_view[n , k, 0] = t_particle + t[k, 0] + smooth_u # for now no t per choice option
            choices_view[n, k, 0] = np.argmax(particles)
            #rts_view[n, 0] = t + t[choices_view[n, 0]]

            enforce_deadline(rts_view, deadline_view, n, k, 0)


    # Build minimal metadata first
    minimal_meta = build_minimal_metadata(
        simulator_name='race_model',
        possible_choices=[-1, 1],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )

    if return_option == 'full':
        # Build v_dict and z_dict dynamically
        v_dict = build_param_dict_from_2d_array(v, 'v', n_particles)
        z_dict = build_param_dict_from_2d_array(z, 'z', n_particles)

        # Update possible_choices for full (n_particles-specific)
        minimal_meta['possible_choices'] = list(np.arange(0, n_particles, 1))

        sim_config = {'delta_t': delta_t, 'max_t': max_t}
        params = {'v': v, 'z': z, 't': t, 'deadline': deadline, 's': s}
        full_meta = build_full_metadata(
            minimal_metadata=minimal_meta,
            params=params,
            sim_config=sim_config,
            boundary_fun=boundary_fun,
            boundary=boundary,
            traj=traj,
            boundary_params=boundary_params,
            extra_params={**v_dict, **z_dict}
        )
        return build_return_dict(rts, choices, full_meta)

    elif return_option == 'minimal':
        return build_return_dict(rts, choices, minimal_meta)

    else:
        raise ValueError('return_option must be either "full" or "minimal"')

# Leaky Competing Accumulator Model ------------------------------------
def lca(np.ndarray[float, ndim = 2] v, # drift parameters (np.array expect: one column of floats)
        np.ndarray[float, ndim = 2] z, # initial bias parameters (np.array expect: one column of floats)
        np.ndarray[float, ndim = 2] g, # decay parameter
        np.ndarray[float, ndim = 2] b, # inhibition parameter
        np.ndarray[float, ndim = 2] t,
        np.ndarray[float, ndim = 2] s, # variance (can be one value or np.array of size as v and w)
        np.ndarray[float, ndim = 1] deadline,
        float delta_t = 0.001, # time-step size in simulator
        float max_t = 20, # maximal time
        int n_samples = 2000, # number of samples to produce
        int n_trials = 1,
        boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
        boundary_params = {},
        random_state = None,
        return_option = 'full',
        smooth_unif = False,
        int n_threads = 1,
        **kwargs):
    """
    Simulate reaction times and choices from a Leaky Competing Accumulator (LCA) model.

    Parameters:
    -----------
    v : np.ndarray, shape (n_trials, n_particles)
        Drift rate parameters for each particle.
    z : np.ndarray, shape (n_trials, n_particles)
        Initial bias parameters for each particle.
    g : np.ndarray, shape (n_trials, 1)
        Decay parameter.
    b : np.ndarray, shape (n_trials, 1)
        Inhibition parameter.
    t : np.ndarray, shape (n_trials, 1)
        Non-decision time.
    s : np.ndarray, shape (n_trials, n_particles)
        Standard deviation of the diffusion process.
    deadline : np.ndarray, shape (n_trials,)
        Deadline for each trial.
    delta_t : float, optional
        Time step size for the simulation (default: 0.001).
    max_t : float, optional
        Maximum time for the simulation (default: 20).
    n_samples : int, optional
        Number of samples to simulate (default: 2000).
    n_trials : int, optional
        Number of trials to simulate (default: 1).
    boundary_fun : callable, optional
        Boundary function that takes time as input (default: None).
    boundary_params : dict, optional
        Parameters for the boundary function (default: {}).
    random_state : int or None, optional
        Seed for random number generation (default: None).
    return_option : str, optional
        Determines the amount of data returned. Can be 'full' or 'minimal' (default: 'full').
    smooth_unif : bool, optional
        If True, applies uniform smoothing to reaction times (default: False).
    n_threads : int, optional
        Number of threads for parallel execution (default: 1).

    Returns:
    --------
    dict
        A dictionary containing simulated reaction times, choices, and metadata.
        The exact contents depend on the 'return_option' parameter.

    Raises:
        ValueError: If n_particles > MAX_PARTICLES (16) for parallel execution.
    """
    cdef int n_particles = v.shape[1]

    # Check particle count limit for parallel execution
    if n_threads > 1 and n_particles > MAX_PARTICLES:
        raise ValueError(
            f"lca parallel execution requires n_particles <= {MAX_PARTICLES}, "
            f"got {n_particles}. Use n_threads=1 for larger particle counts."
        )

    # Check OpenMP availability for parallel execution
    if n_threads > 1:
        from cssm._openmp_status import check_parallel_request
        n_threads = check_parallel_request(n_threads)

    # Sequential path (n_threads=1)
    if n_threads == 1:
        return _lca_sequential(
            v, z, g, b, t, s, deadline, delta_t, max_t, n_samples, n_trials,
            boundary_fun, boundary_params, random_state, return_option, smooth_unif
        )

    # Parallel path
    # Param views
    cdef float[:, :] v_view = v
    cdef float[:, :] z_view = z
    cdef float[:, :] g_view = g
    cdef float[:, :] b_view = b
    cdef float[:, :] t_view = t
    cdef float[:, :] s_view = s
    cdef float[:] deadline_view = deadline

    # Trajectory storage - disabled in parallel mode
    traj = np.zeros((int(max_t / delta_t) + 1, n_particles), dtype=DTYPE)
    traj[:, :] = -999
    cdef float[:, :] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype=DTYPE)
    cdef float[:, :, :] rts_view = rts

    choices = np.zeros((n_samples, n_trials, 1), dtype=np.intc)
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t)
    sqrt_st = s * delta_t_sqrt
    cdef float[:, :] sqrt_st_view = sqrt_st

    cdef int num_steps = int((max_t / delta_t) + 2)
    cdef float c_max_t = max_t
    cdef int c_n_samples = n_samples
    cdef int c_n_particles = n_particles

    # Pre-compute boundaries for all trials (outside nogil)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundaries_all_np = np.zeros((n_trials, len(t_s)), dtype=DTYPE)
    deadlines_tmp = np.zeros(n_trials, dtype=DTYPE)

    cdef Py_ssize_t k_precomp
    for k_precomp in range(n_trials):
        boundary_params_tmp = {key: boundary_params[key][k_precomp] for key in boundary_params.keys()}
        boundary_tmp = np.zeros(t_s.shape, dtype=DTYPE)
        compute_boundary(boundary_tmp, t_s, boundary_fun, boundary_params_tmp)
        boundaries_all_np[k_precomp, :] = boundary_tmp
        deadlines_tmp[k_precomp] = compute_deadline_tmp(max_t, deadline_view[k_precomp], t_view[k_precomp, 0])

    cdef float[:, :] boundaries_view = boundaries_all_np
    cdef float[:] deadlines_view = deadlines_tmp

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
    cdef int ix, i
    cdef float t_particle, deadline_tmp_k, particles_sum, bound_val, noise
    # Per-thread particle arrays to avoid race conditions
    cdef float particles[MAX_THREADS][MAX_PARTICLES]
    cdef float particles_reduced_sum[MAX_THREADS][MAX_PARTICLES]

    # Allocate per-thread GSL RNGs BEFORE parallel block
    for i_thread in range(c_n_threads):
        rng_alloc(&rng_states[i_thread])

    # Parallel execution over FLATTENED iteration space
    with nogil, parallel(num_threads=n_threads):
        for flat_idx in prange(total_iterations, schedule='dynamic'):
            # Get thread ID for per-thread RNG and particle arrays
            tid = threadid()

            k = flat_idx // c_n_samples  # trial index
            n = flat_idx % c_n_samples   # sample index

            # Re-seed per-thread RNG with unique seed for this (trial, sample)
            combined_seed = rng_mix_seed(base_seed, <uint64_t>k, <uint64_t>n)
            rng_seed(&rng_states[tid], combined_seed)

            deadline_tmp_k = deadlines_view[k]
            bound_val = boundaries_view[k, 0]

            # Initialize particle positions (using thread-local array)
            for i in range(c_n_particles):
                particles[tid][i] = z_view[k, i] * bound_val

            t_particle = 0.0
            ix = 0

            # LCA simulation with lateral inhibition
            while True:
                bound_val = boundaries_view[k, ix]
                if check_finished_stack(&particles[tid][0], bound_val, c_n_particles) or t_particle > deadline_tmp_k:
                    break

                # Calculate current sum over particle positions
                particles_sum = csum_stack(&particles[tid][0], c_n_particles)

                # Update particle positions with decay and inhibition
                for i in range(c_n_particles):
                    particles_reduced_sum[tid][i] = (-1.0) * particles[tid][i] + particles_sum
                    noise = ssms_gaussian_f32(&rng_states[tid])
                    particles[tid][i] = particles[tid][i] + ((v_view[k, i] - (g_view[k, 0] * particles[tid][i]) - \
                            (b_view[k, 0] * particles_reduced_sum[tid][i])) * delta_t) + \
                            (sqrt_st_view[k, i] * noise)
                    particles[tid][i] = fmax(0.0, particles[tid][i])

                t_particle = t_particle + delta_t
                ix = ix + 1
                if ix >= num_steps:
                    break

            rts_view[n, k, 0] = t_particle + t_view[k, 0]
            choices_view[n, k, 0] = argmax_stack(&particles[tid][0], c_n_particles)

            # Enforce deadline
            if rts_view[n, k, 0] > deadline_view[k]:
                rts_view[n, k, 0] = -999.0

    # Free per-thread GSL RNGs AFTER parallel block
    for i_thread in range(c_n_threads):
        rng_free(&rng_states[i_thread])

    # Build minimal metadata first
    minimal_meta = build_minimal_metadata(
        simulator_name='lca',
        possible_choices=[-1, 1],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )

    if return_option == 'full':
        # Build v_dict and z_dict dynamically
        v_dict = build_param_dict_from_2d_array(v, 'v', n_particles)
        z_dict = build_param_dict_from_2d_array(z, 'z', n_particles)

        # Update possible_choices for full (n_particles-specific)
        minimal_meta['possible_choices'] = list(np.arange(0, n_particles, 1))

        sim_config = {'delta_t': delta_t, 'max_t': max_t, 'n_threads': n_threads}
        params = {'v': v, 'z': z, 'g': g, 'b': b, 't': t, 'deadline': deadline, 's': s}
        full_meta = build_full_metadata(
            minimal_metadata=minimal_meta,
            params=params,
            sim_config=sim_config,
            boundary_fun=boundary_fun,
            boundary=boundaries_all_np[0] if n_trials > 0 else np.array([]),
            traj=traj,
            boundary_params=boundary_params,
            extra_params={**v_dict, **z_dict}
        )
        return build_return_dict(rts, choices, full_meta)

    elif return_option == 'minimal':
        return build_return_dict(rts, choices, minimal_meta)

    else:
        raise ValueError('return_option must be either "full" or "minimal"')


def _lca_sequential(
    np.ndarray[float, ndim = 2] v,
    np.ndarray[float, ndim = 2] z,
    np.ndarray[float, ndim = 2] g,
    np.ndarray[float, ndim = 2] b,
    np.ndarray[float, ndim = 2] t,
    np.ndarray[float, ndim = 2] s,
    np.ndarray[float, ndim = 1] deadline,
    float delta_t,
    float max_t,
    int n_samples,
    int n_trials,
    boundary_fun,
    boundary_params,
    random_state,
    return_option,
    smooth_unif
):
    """Sequential implementation of lca (original code path)."""

    set_seed(random_state)
    # Param views
    cdef float[:, :] v_view = v
    cdef float[:, :] z_view = z
    cdef float[:, :] g_view = g
    cdef float[:, :] b_view = b
    cdef float[:, :] t_view = t
    cdef float[:, :] s_view = s
    cdef float[:] deadline_view = deadline

    # Trajectory
    cdef int n_particles = v.shape[1]
    traj = np.zeros((int(max_t / delta_t) + 1, n_particles), dtype = DTYPE)
    traj[:, :] = -999
    cdef float[:, :] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    cdef float[:, :, :] rts_view = rts

    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    cdef int[:, :, :] choices_view = choices

    particles = np.zeros(n_particles, dtype = DTYPE)
    cdef float[:] particles_view = particles

    particles_reduced_sum = np.zeros(n_particles, dtype = DTYPE)
    cdef float[:] particles_reduced_sum_view = particles_reduced_sum

    cdef float delta_t_sqrt = sqrt(delta_t)
    sqrt_st = s * delta_t_sqrt
    cdef float[:, :] sqrt_st_view = sqrt_st

    cdef Py_ssize_t n, i, ix, k
    cdef Py_ssize_t m = 0
    cdef float t_par, particles_sum, smooth_u, deadline_tmp

    # Boundary storage
    cdef int num_steps = int((max_t / delta_t) + 2)

    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef int num_draws = num_steps * n_particles
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    cdef Py_ssize_t mu = 0
    cdef float[:] uniform_values = draw_uniform(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        compute_boundary(boundary, t_s, boundary_fun, boundary_params_tmp)

        deadline_tmp = compute_deadline_tmp(max_t, deadline_view[k], t_view[k, 0])
        for n in range(n_samples):
            # Reset particle starting points
            for i in range(n_particles):
                particles_view[i] = z_view[k, i] * boundary_view[0]

            t_particle = 0.0 # reset time
            ix = 0 # reset boundary index

            if n == 0:
                if k == 0:
                    for i in range(n_particles):
                        traj_view[0, i] = particles[i]

            while not check_finished(particles_view, boundary_view[ix], n_particles) and t_particle <= deadline_tmp:
                # calculate current sum over particle positions
                particles_sum = csum(particles_view)

                # update particle positions
                for i in range(n_particles):
                    particles_reduced_sum_view[i] = (- 1) * particles_view[i] + particles_sum
                    particles_view[i] += ((v_view[k, i] - (g_view[k, 0] * particles_view[i]) - \
                            (b_view[k, 0] * particles_reduced_sum_view[i])) * delta_t) + (sqrt_st_view[k, i] * gaussian_values[m])
                    particles_view[i] = fmax(0.0, particles_view[i])
                    m += 1

                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                t_particle += delta_t # increment time
                ix += 1 # increment boundary index

                if n == 0:
                    if k == 0:
                        for i in range(n_particles):
                            traj_view[ix, i] = particles[i]

            smooth_u = compute_smooth_unif(smooth_unif, t_particle, deadline_tmp, delta_t, uniform_values[mu])
            mu += 1
            if mu == num_draws:
                uniform_values = draw_uniform(num_draws)
                mu = 0

            choices_view[n, k, 0] = np.argmax(particles) # store choices for sample n
            rts_view[n, k, 0] = t_particle + t_view[k, 0] + smooth_u # t[choices_view[n, 0]] # store reaction time for sample n

            enforce_deadline(rts_view, deadline_view, n, k, 0)

    # Build minimal metadata first
    minimal_meta = build_minimal_metadata(
        simulator_name='lca',
        possible_choices=[-1, 1],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )

    if return_option == 'full':
        # Build v_dict and z_dict dynamically
        v_dict = build_param_dict_from_2d_array(v, 'v', n_particles)
        z_dict = build_param_dict_from_2d_array(z, 'z', n_particles)

        # Update possible_choices for full (n_particles-specific)
        minimal_meta['possible_choices'] = list(np.arange(0, n_particles, 1))

        sim_config = {'delta_t': delta_t, 'max_t': max_t}
        params = {'v': v, 'z': z, 'g': g, 'b': b, 't': t, 'deadline': deadline, 's': s}
        full_meta = build_full_metadata(
            minimal_metadata=minimal_meta,
            params=params,
            sim_config=sim_config,
            boundary_fun=boundary_fun,
            boundary=boundary,
            traj=traj,
            boundary_params=boundary_params,
            extra_params={**v_dict, **z_dict}
        )
        return build_return_dict(rts, choices, full_meta)

    elif return_option == 'minimal':
        return build_return_dict(rts, choices, minimal_meta)

    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# -----------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: Racing Diffusion Model ----------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)

def racing_diffusion_model(np.ndarray[float, ndim = 2] v,  # mean drift rates
                  np.ndarray[float, ndim = 2] b,  # response boundaries (thresholds)
                  np.ndarray[float, ndim = 2] A,  # between-trial variability in starting point (U[0, A])
                  np.ndarray[float, ndim = 2] t,  # non-decision times
                  np.ndarray[float, ndim = 2] s,  # diffusion coefficients (within-trial noise)
                  np.ndarray[float, ndim = 1] deadline,
                  float delta_t = 0.001, # time increment step
                  float max_t = 20, # maximum rt allowed
                  int n_samples = 2000,
                  int n_trials = 1,
                  random_state = None,
                  return_option = 'full',
                  smooth_unif = False,
                  **kwargs):
    """
    Simulate reaction times and choices from the Racing Diffusion Model (RDM)
    based on the generative process described in Tillman et al. (2020).

    This model implements a "first-past-the-post" race of N independent
    Wiener diffusion processes with no reflecting lower boundary.

    Parameters:
    -----------
    v : np.ndarray
        Mean drift rates. Shape (n_trials, n_particles).
    b : np.ndarray
        Response boundaries (thresholds), shared across particles within a trial. Shape (n_trials, 1).
    A : np.ndarray
        Upper bound of the uniform starting point distribution (U[0, A]) shared across particles within a trial. Shape (n_trials, 1).
    t : np.ndarray
        Non-decision times. Shape (n_trials, 1).
    s : np.ndarray
        Diffusion coefficients (within-trial noise), shared across all particles in a trial. Shape (n_trials, 1).
    deadline : np.ndarray
        Maximum reaction time allowed for each trial. Shape (n_trials,).
    delta_t : float, optional
        Time increment step for simulation (default: 0.001).
    max_t : float, optional
        Maximum time for simulation (default: 20).
    n_samples : int, optional
        Number of samples to simulate per trial (default: 2000).
    n_trials : int, optional
        Number of trials to simulate (default: 1).
    random_state : int or None, optional
        Seed for random number generator (default: None).
    return_option : str, optional
        'full' for complete output, 'minimal' for basic output (default: 'full').
    smooth_unif : bool, optional
        Whether to apply uniform smoothing to reaction times (default: False).
    **kwargs : dict
        Additional keyword arguments.

    Returns:
    --------
    dict
        A dictionary containing simulated reaction times, choices, and metadata.
        The exact contents depend on the 'return_option' parameter.
    """

    set_seed(random_state)
    # Param views
    cdef float[:, :] v_view = v
    cdef float[:, :] b_view = b
    cdef float[:, :] A_view = A
    cdef float[:, :] t_view = t
    cdef float[:, :] s_view = s
    cdef float[:] deadline_view = deadline

    cdef float delta_t_sqrt = sqrt(delta_t)
    sqrt_st = delta_t_sqrt * s
    cdef float[:, :] sqrt_st_view = sqrt_st

    cdef int n_particles = v.shape[1]
    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    cdef float[:, :, :] rts_view = rts
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    cdef int[:, :, :] choices_view = choices

    particles = np.zeros((n_particles), dtype = DTYPE)
    cdef float [:] particles_view = particles

    # Trajectory saving (for first trial, first sample)
    traj = np.zeros((int(max_t / delta_t) + 1, n_particles), dtype = DTYPE)
    traj[:, :] = -999
    cdef float[:, :] traj_view = traj

    # Initialize variables needed for for loop
    cdef float t_particle, smooth_u, deadline_tmp
    cdef Py_ssize_t n, ix, j, k
    cdef Py_ssize_t m = 0
    cdef int winner = -1
    cdef int winner_found = 0 # <-- FIX: Use 'int' (0=False, 1=True) instead of 'bool'

    cdef int num_steps = int((max_t / delta_t) + 1)
    cdef int num_draws = num_steps * n_particles
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    for k in range(n_trials):

        deadline_tmp = min(max_t, deadline_view[k] - t_view[k, 0])

        # Loop over samples
        for n in range(n_samples):

            for j in range(n_particles):
                particles_view[j] = random_uniform() * A_view[k, 0]

            t_particle = 0.0 # reset time
            ix = 0
            winner = -1         # Reset winner for this sample
            winner_found = 0    # <-- FIX: Reset to 0 (False)

            # Save initial trajectory
            if n == 0:
                if k == 0:
                    for j in range(n_particles):
                        traj_view[0, j] = particles[j]

            # Random walker
            while not winner_found and t_particle <= deadline_tmp: # <-- 'not 0' is True
                for j in range(n_particles):
                    # Standard Wiener diffusion process update
                    particles_view[j] += (v_view[k, j] * delta_t) + sqrt_st_view[k, 0] * gaussian_values[m]

                    # No reflecting boundary for RDM
                    # The line `particles_view[j] = fmax(0.0, particles_view[j])` is REMOVED.

                    m += 1
                    if m == num_draws: # Resample random numbers if needed
                        m = 0
                        gaussian_values = draw_gaussian(num_draws)

                    # Check for a winner (first-past-the-post)
                    if particles_view[j] >= b_view[k, 0]:
                        winner_found = 1 # <-- FIX: Set to 1 (True)
                        winner = j
                        break # Stop checking, we have a winner

                if winner_found: # <-- `if 1` is True
                    break # Stop the while loop, a decision is made

                t_particle += delta_t
                ix += 1

                # Save running trajectory
                if n == 0:
                    if k == 0:
                        for j in range(n_particles):
                            traj_view[ix, j] = particles[j]

            # --- End of while loop ---

            # Apply smoothing if specified (using shared utility for consistency)
            smooth_u = compute_smooth_unif(t_particle, deadline_tmp, delta_t, smooth_unif)

            # Store RT and choice
            rts_view[n , k, 0] = t_particle + t[k, 0] + smooth_u
            choices_view[n, k, 0] = winner

            # Handle non-responses (deadline hit or no decision)
            enforce_deadline(rts_view, deadline_view, n, k, 0)
            if rts_view[n, k, 0] == -999 or (not winner_found):
                rts_view[n, k, 0] = -999
                choices_view[n, k, 0] = -1  # Ensure choice is also -1

    # Build minimal metadata first
    minimal_meta = build_minimal_metadata(
        simulator_name='rdm_simulator',
        possible_choices=list(np.arange(0, n_particles, 1)),
        n_samples=n_samples,
        n_trials=n_trials,
    )

    if return_option == 'full':
        v_dict = build_param_dict_from_2d_array(v, 'v', n_particles)

        sim_config = {'delta_t': delta_t, 'max_t': max_t}
        params = {'v': v, 'b': b, 'A': A, 't': t, 'deadline': deadline, 's': s}
        full_meta = build_full_metadata(
            minimal_metadata=minimal_meta,
            params=params,
            sim_config=sim_config,
            traj=traj,
            extra_params=v_dict
        )
        return build_return_dict(rts, choices, full_meta)

    elif return_option == 'minimal':
        return build_return_dict(rts, choices, minimal_meta)

    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# -----------------------------------------------------------------------------------------------
