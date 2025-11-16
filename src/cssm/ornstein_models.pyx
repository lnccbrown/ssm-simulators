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
    cdef float[:] v_view  = v
    cdef float[:] a_view = a
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

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
        compute_boundary(boundary, t_s, a_view[k], boundary_fun, 
                        boundary_params_tmp, boundary_multiplicative)
    
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

            smooth_u = compute_smooth_unif(smooth_unif, t_particle, deadline_tmp, delta_t)

            rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u
            choices_view[n, k, 0] = sign(y)

            enforce_deadline(rts_view, deadline_view, n, k, 0)

    # Build minimal metadata first
    minimal_meta = build_minimal_metadata(
        simulator_name='ornstein_uhlenbeck',
        possible_choices=[-1, 1],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )
    
    if return_option == 'full':
        sim_config = {'delta_t': delta_t, 'max_t': max_t}
        params = {
            'v': v, 'a': a, 'z': z, 'g': g, 't': t,
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

