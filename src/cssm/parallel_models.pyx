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

# Parallel Models ------------------------------------

def ddm_flexbound_par2(np.ndarray[float, ndim = 1] vh, 
                       np.ndarray[float, ndim = 1] vl1,
                       np.ndarray[float, ndim = 1] vl2,
                       np.ndarray[float, ndim = 1] a,
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
                       boundary_multiplicative = True,
                       boundary_params = {},
                       random_state = None,
                       return_option = 'full',
                       smooth_unif = False,
                       **kwargs):
    """
    Simulate a parallel diffusion decision model with flexible boundaries.

    This function simulates a two-stage decision process where a high-dimensional choice
    is made first, followed by a low-dimensional choice. The process uses a flexible
    boundary that can change over time.

    Parameters:
    -----------
    vh, vl1, vl2 : np.ndarray
        Drift rates for high-dimensional and two low-dimensional choices.
    a : np.ndarray
        Initial boundary separation.
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
    boundary_multiplicative : bool, optional
        If True, boundary function is multiplied by 'a'. If False, it's added. Default is True.
    boundary_params : dict, optional
        Additional parameters for the boundary function.
    random_state : int or None, optional
        Seed for random number generator. Default is None.
    return_option : str, optional
        Determines the content of the returned dictionary. Can be 'full' or 'minimal'. Default is 'full'.
    smooth_unif : bool, optional
        If True, adds uniform noise to simulate continuous time. Default is False.

    Returns:
    --------
    dict
        A dictionary containing simulation results. The exact contents depend on the return_option.
        'full' returns all simulation data and parameters, while 'minimal' returns only essential outputs.
    """

    set_seed(random_state)
    # Param views
    cdef float[:] vh_view = vh
    cdef float[:] vl1_view = vl1
    cdef float[:] vl2_view = vl2
    cdef float[:] a_view = a
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

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
        compute_boundary(boundary, t_s, a_view[k], boundary_fun, 
                        boundary_params_tmp, boundary_multiplicative)
        
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
                if random_uniform() <= 0.5:
                    choices_view[n, k, 0] += 2
            # Otherwise apply rule from above
            elif random_uniform() <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 2

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
            smooth_u = compute_smooth_unif(smooth_unif, t_particle, deadline_tmp, delta_t)

            rts_view[n, k, 0] = fmax(t_h, t_l) + t_view[k] + smooth_u
            rts_high_view[n, k, 0] = t_h + t_view[k]
            rts_low_view[n, k, 0] = t_l + t_view[k]

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add one deterministically
            # y at lower bound --> choice_view[n, k, 0] stays the same deterministically
            
            # If boundary is negative (or 0) already, we flip a coin
            if boundary_view[ix] <= 0:
                if random_uniform() <= 0.5:
                    choices_view[n, k, 0] += 1
            # Otherwise apply rule from above
            elif random_uniform() <= ((y_l + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 1

            enforce_deadline(rts_view, deadline_view, n, k, 0)

    # Build minimal metadata first
    minimal_meta = build_minimal_metadata(
        simulator_name='ddm_flexbound',
        possible_choices=[0, 1, 2, 3],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )
    
    # Extra arrays for this model
    extra_arrays_dict = {'rts_low': rts_low, 'rts_high': rts_high}
    
    if return_option == 'full':
        sim_config = {'delta_t': delta_t, 'max_t': max_t}
        params = {
            'vh': vh, 'vl1': vl1, 'vl2': vl2,
            'a': a, 'zh': zh, 'zl1': zl1, 'zl2': zl2,
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
