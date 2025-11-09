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
from libc.math cimport sqrt, log, fmax

import numpy as np
cimport numpy as np

# Import utility functions from the _utils module
from cssm._utils import set_seed, random_uniform, draw_gaussian, sign

DTYPE = np.float32

# Sequential Models ------------------------------------

def ddm_flexbound_seq2(np.ndarray[float, ndim = 1] vh,
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
    boundary_multiplicative : bool, optional
        If True, the boundary function is multiplicative; if False, it's additive (default: True).
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
    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices
    cdef int decision_taken = 0

    # TD: Add Trajectory
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

    cdef float y_h, t_particle, t_particle1, t_particle2, y_l, y_l1, y_l2, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n, ix, ix1, ix2, k
    cdef Py_ssize_t m = 0
    #cdef Py_ssize_t traj_id
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
                    if random_uniform() <= 0.5:
                        choices_view[n, k, 0] += 2
                elif random_uniform() <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
                        choices_view[n, k, 0] += 2

                # Low dim choice random (didn't even get to process it if rt is at max after first choice)
                # so we just apply a priori bias
                if choices_view[n, k, 0] == 0:
                    if random_uniform() <= zl1_view[k]:
                        choices_view[n, k, 0] += 1
                else:
                    if random_uniform() <= zl2_view[k]:
                        choices_view[n, k, 0] += 1
                rts_view[n, k, 0] = t_particle
                decision_taken = 1
            else:
                # If boundary is negative (or 0) already, we flip a coin
                if boundary_view[ix] <= 0:
                    if random_uniform() <= 0.5:
                        choices_view[n, k, 0] += 2
                # Otherwise apply rule from above
                elif random_uniform() <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
                    choices_view[n, k, 0] += 2

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
                        if random_uniform() < zl1_view[k]:
                            choices_view[n, k, 0] += 1
                        decision_taken = 1
                    
                    if n == 0:
                        if k == 0:
                            traj_view[ix, 1] = y_l1
                else:
                    # In case boundary is negative already, we flip a coin with bias determined by w_l_ parameter
                    if (y_l2 >= boundary_view[ix]) or (y_l2 <= ((-1) * boundary_view[ix])):
                        if random_uniform() < zl2_view[k]:
                            choices_view[n, k, 0] += 1
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

            if smooth_unif:
                if t_particle == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif t_particle < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            # Add nondecision time and smoothing of rt
            rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u

            # Take account of deadline
            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                    rts_view[n, k, 0] = -999

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add one deterministically
            # y at lower bound --> choice_view[n, k, 0] stays the same deterministically
            
            # If boundary is negative (or 0) already, we flip a coin
            if not decision_taken:
                if boundary_view[ix] <= 0:
                    if random_uniform() <= 0.5:
                        choices_view[n, k, 0] += 1
                # Otherwise apply rule from above
                elif random_uniform() <= ((y_l + boundary_view[ix]) / (2 * boundary_view[ix])):
                    choices_view[n, k, 0] += 1

    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'metadata': {'vh': vh,
                                                            'vl1': vl1,
                                                            'vl2': vl2,
                                                            'a': a,
                                                            'zh': zh,
                                                            'zl1': zl1,
                                                            'zl2': zl2,
                                                            't': t,
                                                            'deadline': deadline,
                                                            's': s,
                                                            **boundary_params,
                                                            'delta_t': delta_t,
                                                            'max_t': max_t,
                                                            'n_samples': n_samples,
                                                            'n_trials': n_trials,
                                                            'simulator': 'ddm_flexbound',
                                                            'boundary_fun_type': boundary_fun.__name__,
                                                            'trajectory': traj,
                                                            'possible_choices': [0, 1, 2, 3],
                                                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices, 'metadata': {'simulator': 'ddm_flexbound', 
                                                             'possible_choices': [0, 1, 2, 3],
                                                             'boundary_fun_type': boundary_fun.__name__,
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             }}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# -----------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
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

        # Precompute boundary evaluations
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        
        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
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
            
            if smooth_unif:
                if t_h == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif fmax(t_h, t_l) < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

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

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999

    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'rts_low': rts_low, 'rts_high': rts_high, 
                'metadata': {'vh': vh,
                            'vl1': vl1,
                            'vl2': vl2,
                            'a': a,
                            'zh': zh,
                            'zl1': zl1,
                            'zl2': zl2,
                            't': t,
                            'deadline': deadline,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'n_trials': n_trials,
                            'simulator': 'ddm_flexbound',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [0, 1, 2, 3],
                            'trajectory': traj,
                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices, 'rts_low': rts_low, 'rts_high': rts_high, 
                'metadata': {'simulator': 'ddm_flexbound', 
                             'possible_choices': [0, 1, 2, 3],
                             'boundary_fun_type': boundary_fun.__name__,
                             'n_samples': n_samples,
                             'n_trials': n_trials,
                             }}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# -----------------------------------------------------------------------------------------------

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
                                boundary_multiplicative = True,
                                boundary_params = {},
                                random_state = None,
                                return_option = 'full',
                                smooth_unif = False,
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
    boundary_multiplicative : bool, optional
        Whether the boundary function is multiplicative (default: True).
    boundary_params : dict, optional
        Parameters for the boundary function (default: {}).
    random_state : int or None, optional
        Random seed for reproducibility (default: None).
    return_option : str, optional
        Determines what to return, either 'full' or 'minimal' (default: 'full').
    smooth_unif : bool, optional
        Whether to use smooth uniform distribution for RT jitter (default: False).

    Returns:
    --------
    dict
        Dictionary containing simulated data and metadata. The exact contents depend on the return_option.
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
                if random_uniform() <= 0.5:
                    choices_view[n, k, 0] += 2
            # Otherwise, apply rule from above
            elif random_uniform() <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 2

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

            if smooth_unif:
                if t_h == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif fmax(t_h, t_l) < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = fmax(t_h, t_l) + t_view[k]
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
            elif random_uniform() <= ((y_l + boundary_view[ix_l]) / (2 * boundary_view[ix_l])):
                choices_view[n, k, 0] += 1

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999

    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'rts_high': rts_high, 'rts_low': rts_low, 
                'metadata': {'vh': vh,
                            'vl1': vl1,
                            'vl2': vl2,
                            'a': a,
                            'zh': zh,
                            'zl1': zl1,
                            'zl2': zl2,
                            'd': d,
                            't': t,
                            'deadline': deadline,
                            's_pre_high_level_choice': s_pre_high_level_choice,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'n_trials': n_trials,
                            'simulator': 'ddm_flexbound_mic2_adj',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [0, 1, 2, 3],
                            'trajectory': traj,
                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices, 'rts_high': rts_high, 'rts_low': rts_low, 
                'metadata': {'simulator': 'ddm_flexbound_mic2_adj', 
                             'possible_choices': [0, 1, 2, 3],
                             'boundary_fun_type': boundary_fun.__name__,
                             'n_samples': n_samples,
                             'n_trials': n_trials,
                             }}
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
                                  boundary_multiplicative = True,
                                  boundary_params = {},
                                  random_state = None,
                                  return_option = 'full',
                                  smooth_unif = False,
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
    boundary_multiplicative : bool, optional
        Whether the boundary function is multiplicative (default: True).
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
                if random_uniform() <= 0.5:
                    choices_view[n, k, 0] += 2
            # Otherwise, apply rule from above
            elif random_uniform() <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 2

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

            if smooth_unif:
                if t_h == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif fmax(t_h, t_l) < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = fmax(t_h, t_l) + t_view[k]
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

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999

    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'rts_high': rts_high, 'rts_low': rts_low, 
                'metadata': {'vh': vh,
                            'vl1': vl1,
                            'vl2': vl2,
                            'a': a,
                            'zh': zh,
                            'zl1': zl1,
                            'zl2': zl2,
                            'd': d,
                            't': t,
                            'deadline': deadline,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'n_trials': n_trials,
                            'simulator': 'ddm_flexbound_mic2_adj',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [0, 1, 2, 3],
                            'trajectory': traj,
                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices, 'rts_high': rts_high, 'rts_low': rts_low, 
                'metadata': {'simulator': 'ddm_flexbound_mic2_adj', 
                             'possible_choices': [0, 1, 2, 3],
                             'boundary_fun_type': boundary_fun.__name__,
                             'n_samples': n_samples,
                             'n_trials': n_trials,
                             }}
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
                                           boundary_multiplicative = True,
                                           boundary_params = {},
                                           random_state = None,
                                           return_option = 'full',
                                           smooth_unif = False,
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
    boundary_multiplicative : bool, optional
        Whether the boundary function is multiplicative (default is True).
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
                if random_uniform() <= 0.5:
                    choices_view[n, k, 0] += 2
            # Otherwise, apply rule from above
            elif random_uniform() <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 2
           
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

            if smooth_unif:
                if t_h == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif fmax(t_h, t_l) < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = fmax(t_h, t_l) + t_view[k]
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

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999

    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'rts_high': rts_high, 'rts_low': rts_low, 
                'metadata': {'vh': vh,
                            'vl1': vl1,
                            'vl2': vl2,
                            'a': a,
                            'zh': zh,
                            'zl1': zl1,
                            'zl2': zl2,
                            'd': d,
                            't': t,
                            'deadline': deadline,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'n_trials': n_trials,
                            'simulator': 'ddm_flexbound_mic2_adj',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [0, 1, 2, 3],
                            'trajectory': traj,
                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices, 'rts_high': rts_high, 'rts_low': rts_low, 
                'metadata': {'simulator': 'ddm_flexbound_mic2_adj', 
                             'possible_choices': [0, 1, 2, 3],
                             'boundary_fun_type': boundary_fun.__name__,
                             'n_samples': n_samples,
                             'n_trials': n_trials,
                             }}
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
            if rts_view[n, k, 0] >= deadline_view[k]:
                rts_view[n, k, 0] = -999
        

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
            if rts_view[n, k, 0] >= deadline_view[k]:
                rts_view[n, k, 0] = -999

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
        **kwargs
        ):

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
            if rts_view[n, k, 0] >= deadline_view[k]:
                rts_view[n, k, 0] = -999
        

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
            if rts_view[n, k, 0] >= deadline_view[k]:
                rts_view[n, k, 0] = -999
        

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
                                                        boundary_multiplicative = True,
                                                        boundary_params = {},
                                                        random_state = None,
                                                        return_option = 'full',
                                                        smooth_unif = False,
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
    boundary_multiplicative : bool, optional
        If True, boundary function is multiplicative; if False, additive (default: True).
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
                if random_uniform() <= 0.5:
                    choices_view[n, k, 0] += 2
            # Otherwise, apply rule from above
            elif random_uniform() <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 2
           
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

            if smooth_unif:
                if t_h == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif fmax(t_h, t_l) < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = fmax(t_h, t_l) + t_view[k]
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

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999

    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'rts_high': rts_high, 'rts_low': rts_low, 
                'metadata': {'vh': vh,
                            'vl1': vl1,
                            'vl2': vl2,
                            'a': a,
                            'zh': zh,
                            'zl1': zl1,
                            'zl2': zl2,
                            'd': d,
                            't': t,
                            'deadline': deadline,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'n_trials': n_trials,
                            'simulator': 'ddm_flexbound_mic2_adj',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [0, 1, 2, 3],
                            'trajectory': traj,
                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices, 'rts_high': rts_high, 'rts_low': rts_low, 
                'metadata': {'simulator': 'ddm_flexbound_mic2_adj', 
                             'possible_choices': [0, 1, 2, 3],
                             'boundary_fun_type': boundary_fun.__name__,
                             'n_samples': n_samples,
                             'n_trials': n_trials,
                             }}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# ----------------------------------------------------------------------------------------------------


# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
