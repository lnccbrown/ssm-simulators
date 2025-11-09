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
from cssm._utils import set_seed, random_uniform, draw_gaussian, sign, csum

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
               np.ndarray[float, ndim = 2] a, # initial boundary separation
               np.ndarray[float, ndim = 2] z, # np.array expected, one column of floats
               np.ndarray[float, ndim = 2] t, # for now we we don't allow t by choice
               np.ndarray[float, ndim = 2] s, # np.array expected, one column of floats
               np.ndarray[float, ndim = 1] deadline,
               float delta_t = 0.001, # time increment step
               float max_t = 20, # maximum rt allowed
               int n_samples = 2000, 
               int n_trials = 1,
               boundary_fun = None,
               boundary_multiplicative = True,
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
        boundary_multiplicative (bool): If True, boundary function is multiplicative; if False, additive.
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
    cdef float[:, :] a_view = a
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

        # Precompute boundary evaluations
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k, 0], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k, 0], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
    
        deadline_tmp = min(max_t, deadline_view[k] - t_view[k, 0])
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

            if smooth_unif:
                if t_particle == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif t_particle < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n , k, 0] = t_particle + t[k, 0] + smooth_u # for now no t per choice option
            choices_view[n, k, 0] = np.argmax(particles)
            #rts_view[n, 0] = t + t[choices_view[n, 0]]

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999
            

        # Create some dics
        v_dict = {}
        z_dict = {}
        #t_dict = {}
        for i in range(n_particles):
            v_dict['v' + str(i)] = v[:, i]
            z_dict['z' + str(i)] = z[:, i]
            #t_dict['t_' + str(i)] = t[i] # for now no t by choice

    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'metadata': {**v_dict,
                                                            'a': a, 
                                                            **z_dict,
                                                            't': t,
                                                            'deadline': deadline,
                                                            # **t_dict, # for now no t by choice
                                                            's': s,
                                                            **boundary_params,
                                                            'delta_t': delta_t,
                                                            'max_t': max_t,
                                                            'n_samples': n_samples,
                                                            'n_trials': n_trials,
                                                            'simulator': 'race_model',
                                                            'boundary_fun_type': boundary_fun.__name__,
                                                            'possible_choices': list(np.arange(0, n_particles, 1)),
                                                            'trajectory': traj,
                                                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices, 'metadata': {'simulator': 'race_model', 
                                                             'possible_choices': [-1, 1],
                                                             'boundary_fun_type': boundary_fun.__name__,
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             }}

    else:
        raise ValueError('return_option must be either "full" or "minimal"')
    # -------------------------------------------------------------------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)

# Simulate (rt, choice) tuples from: Leaky Competing Accumulator Model -----------------------------
def lca(np.ndarray[float, ndim = 2] v, # drift parameters (np.array expect: one column of floats)
        np.ndarray[float, ndim = 2] a, # criterion height
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
        boundary_multiplicative = True,
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
    boundary_multiplicative : bool, optional
        If True, the boundary function is multiplicative; if False, it's additive (default: True).
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
    cdef float[:, :] a_view = a
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
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k, 0], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k, 0], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)

        deadline_tmp = min(max_t, deadline_view[k] - t_view[k, 0])
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

            if smooth_unif:
                if t_particle == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif t_particle < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0
        
            choices_view[n, k, 0] = np.argmax(particles) # store choices for sample n
            rts_view[n, k, 0] = t_particle + t_view[k, 0] + smooth_u # t[choices_view[n, 0]] # store reaction time for sample n

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999
        
    # Create some dics
    v_dict = {}
    z_dict = {}
    #t_dict = {}
    
    for i in range(n_particles):
        v_dict['v' + str(i)] = v[:, i]
        z_dict['z' + str(i)] = z[:, i]

    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'metadata': {**v_dict,
                                                            'a': a,
                                                            **z_dict,
                                                            'g': g,
                                                            'b': b,
                                                            't': t,
                                                            'deadline': deadline,
                                                            's': s,
                                                            **boundary_params,
                                                            'delta_t': delta_t,
                                                            'max_t': max_t,
                                                            'n_samples': n_samples,
                                                            'n_trials': n_trials,
                                                            'simulator' : 'lca',
                                                            'boundary_fun_type': boundary_fun.__name__,
                                                            'possible_choices': list(np.arange(0, n_particles, 1)),
                                                            'trajectory': traj,
                                                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices, 'metadata': {'simulator': 'lca', 
                                                             'possible_choices': [-1, 1],
                                                             'boundary_fun_type': boundary_fun.__name__,
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             }}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# -----------------------------------------------------------------------------------------------
