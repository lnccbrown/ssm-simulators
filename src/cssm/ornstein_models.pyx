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
from libc.math cimport sqrt

import numpy as np
cimport numpy as np

# Import utility functions from the _utils module
from cssm._utils import set_seed, random_uniform, draw_gaussian, sign

DTYPE = np.float32

def ornstein_uhlenbeck(np.ndarray[float, ndim = 1] v, # drift parameter
                       np.ndarray[float, ndim = 1] a, # initial boundary separation
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
                       boundary_multiplicative = True,
                       boundary_params = {},
                       random_state = None,
                       return_option = 'full',
                       smooth_unif = False,
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
    # Data-structs for trajectory storage
    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:,:] traj_view = traj

    # Param views
    cdef float[:] v_view  = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] g_view = g
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s

    # Initializations
    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE) # rt storage
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc) # choice storage

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    #cdef float sqrt_st = s * delta_t_sqrt

    # Boundary Storage
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef float y, t_particle, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
    
        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
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

            if smooth_unif:
                if t_particle == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif t_particle < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u
            choices_view[n, k, 0] = sign(y)

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999

    if return_option == 'full':
        return { 'rts': rts, 'choices': choices, 'metadata': {'v': v,
                                                            'a': a,
                                                            'z': z,
                                                            'g': g,
                                                            't': t,
                                                            'deadline': deadline,
                                                            's': s,
                                                            **boundary_params,
                                                            'delta_t': delta_t,
                                                            'max_t': max_t,
                                                            'n_samples': n_samples,
                                                            'n_trials': n_trials,
                                                            'simulator': 'ornstein_uhlenbeck',
                                                            'boundary_fun_type': boundary_fun.__name__,
                                                            'possible_choices': [-1, 1],
                                                            'trajectory': traj,
                                                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices,  'metadata': {'simulator': 'ornstein_uhlenbeck', 
                                                             'possible_choices': [-1, 1],
                                                             'boundary_fun_type': boundary_fun.__name__,
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             }}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')

