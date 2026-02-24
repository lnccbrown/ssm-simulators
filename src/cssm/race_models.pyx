# Globaly settings for cython
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

import numpy as np
cimport numpy as np

# Import utility functions from the _utils module
from cssm._utils import (
    set_seed,
    random_uniform,
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

DTYPE = np.float32

# Helper function for race models ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)

cdef bint check_finished(float[:] particles, float boundary, int n):
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

#def test_check():
#    # Quick sanity check for the check_finished function
#    temp = np.random.normal(0,1, 10).astype(DTYPE)
#    cdef float[:] temp_view = temp
#    start = time()
#    [check_finished(temp_view, 3) for _ in range(1000000)]
#    print(check_finished(temp_view, 3))
#    end = time()
#    print("cython check: {}".format(start - end))
#    start = time()
#    [(temp > 3).any() for _ in range(1000000)]
#    end = time()
#    print("numpy check: {}".format(start - end))

# @cythonboundscheck(False)
# @cythonwraparound(False)
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
               **kwargs):
    """
    Simulate reaction times and choices from a race model with N samples.

    Args:
        v (np.ndarray): Drift rates for each accumulator and trial.
        a (np.ndarray): Initial boundary separation for each trial.
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
        **kwargs: Additional keyword arguments.

    Returns:
        dict: A dictionary containing simulated reaction times, choices, and metadata.
              The exact contents depend on the return_option.

    Raises:
        ValueError: If return_option is not 'full' or 'minimal'.
    """

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

            smooth_u = compute_smooth_unif(smooth_unif, t_particle, deadline_tmp, delta_t)

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
    # -------------------------------------------------------------------------------------------------

# @cythonboundscheck(False)
# @cythonwraparound(False)

# Simulate (rt, choice) tuples from: Leaky Competing Accumulator Model -----------------------------
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
        **kwargs):
    """
    Simulate reaction times and choices from a Leaky Competing Accumulator (LCA) model.

    Parameters:
    -----------
    v : np.ndarray, shape (n_trials, n_particles)
        Drift rate parameters for each particle.
    a : np.ndarray, shape (n_trials, 1)
        Criterion height (decision threshold).
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

    Returns:
    --------
    dict
        A dictionary containing simulated reaction times, choices, and metadata.
        The exact contents depend on the 'return_option' parameter.
    """

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

            smooth_u = compute_smooth_unif(smooth_unif, t_particle, deadline_tmp, delta_t)

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
        Response boundaries (thresholds). Shape (n_trials, n_particles).
    A : np.ndarray
        Upper bound of the uniform starting point distribution (U[0, A]) shared across particles within a trial. Shape (n_trials, 1).
    t : np.ndarray
        Non-decision times. Shape (n_trials, 1).
    s : np.ndarray
        Diffusion coefficients (within-trial noise). Shape (n_trials, n_particles).
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

            # Apply smoothing if specified
            if smooth_unif:
                if t_particle == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif t_particle < deadline_tmp: # Only smooth if not a deadline response
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            # Store RT and choice
            rts_view[n , k, 0] = t_particle + t[k, 0] + smooth_u
            choices_view[n, k, 0] = winner

            # Handle non-responses (deadline hit or no decision)
            if (rts_view[n, k, 0] >= deadline_view[k]) | (not winner_found): # <-- `not 0` is True
                rts_view[n, k, 0] = -999
                choices_view[n, k, 0] = -1 # Ensure choice is also -1


        # Create parameter dictionaries for metadata
        v_dict = {}
        for i in range(n_particles):
            v_dict['v' + str(i)] = v[:, i]

    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'metadata': {**v_dict,
                                                            'b': b,
                                                            'A': A,
                                                            't': t,
                                                            'deadline': deadline,
                                                            's': s,
                                                            'delta_t': delta_t,
                                                            'max_t': max_t,
                                                            'n_samples': n_samples,
                                                            'n_trials': n_trials,
                                                            'simulator': 'rdm_simulator',
                                                            'possible_choices': list(np.arange(0, n_particles, 1)),
                                                            'trajectory': traj}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices, 'metadata': {'simulator': 'rdm_simulator',
                                                             'possible_choices': list(np.arange(0, n_particles, 1)),
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             }}

    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# -----------------------------------------------------------------------------------------------
