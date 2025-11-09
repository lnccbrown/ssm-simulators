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
"""

import cython
from libc.math cimport sqrt, log, exp, fmax

import numpy as np
cimport numpy as np

# Import utility functions from the _utils module
from cssm._utils import set_seed, random_uniform, draw_gaussian, sign

DTYPE = np.float32

# Simulate (rt, choice) tuples from: FULL DDM (HDDM BASE) ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)

def full_ddm_hddm_base(np.ndarray[float, ndim = 1] v, # = 0,
                       np.ndarray[float, ndim = 1] a, # = 1,
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
                       **kwargs,
                       ):
    """
    Simulate reaction times and choices from a full drift diffusion model with flexible bounds.

    Args:
        v (np.ndarray): Drift rate for each trial.
        a (np.ndarray): Boundary separation for each trial.
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

    set_seed(random_state)
    # cdef int cov_length = np.max([v.size, a.size, w.size, t.size]).astype(int)

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
    # Data-structs for trajectory storage
    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:, :] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    
    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)

    cdef float y, t_particle, t_tmp, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0
    cdef float drift_increment = 0.0
    cdef float[:] gaussian_values = draw_gaussian(num_draws) 

    # Loop over trials
    for k in range(n_trials): 
        # Loop over samples
        deadline_tmp = min(max_t, deadline_view[k] - t_view[k]) 
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
            if smooth_unif :
                if t_particle == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif t_particle < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = t_particle + t_tmp + smooth_u # Store rt
            
            if y < 0:
                choices_view[n, k, 0] = 0 # Store choice
            else:
                choices_view[n, k, 0] = 1

            # If the rt exceeds the deadline, set rt to -999 and choice to -1 
            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999

    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'metadata': {'v': v,
                                'a': a,
                                'z': z,
                                't': t,
                                'sz': sz,
                                'sv': sv,
                                'st': st,
                                'deadline': deadline,
                                's': s,
                                'delta_t': delta_t,
                                'max_t': max_t,
                                'n_samples': n_samples,
                                'n_trials': n_trials,
                                'simulator': 'full_ddm_hddm_base',
                                'possible_choices': [0, 1],
                                'boundary_fun_type': 'constant',
                                'trajectory': traj}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices, 'metadata': {'simulator': 'full_ddm_hddm_base', 
                                                             'possible_choices': [0, 1],
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             'boundary_fun_type': 'constant'}}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# -------------------------------------------------------------------------------------------------


# Simulate (rt, choice) tuples from: SIMPLE DDM -----------------------------------------------
# Simplest algorithm
# delete random comment
# delete random comment 2
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
        **kwargs: Additional keyword arguments.

    Returns:
        dict: A dictionary containing simulated reaction times, choices, and metadata.
              The exact contents depend on the return_option.

    Raises:
        ValueError: If return_option is neither 'full' nor 'minimal'.
    """

    set_seed(random_state)
    # Param views
    cdef float[:] v_view = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] s_view = s
    cdef float[:] deadline_view = deadline

    # Data-structs for trajectory storage
    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:, :] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t)
    #cdef float sqrt_st = delta_t_sqrt * s

    cdef float y, t_particle, smooth_u, deadline_tmp, sqrt_st

    #cdef int n
    cdef Py_ssize_t n, ix, k
    cdef int m = 0
    cdef int num_draws = int(max_t / delta_t + 1)
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    
    for k in range(n_trials):
        # Loop over samples
        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
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
            if smooth_unif :
                if t_particle == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif t_particle < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u # store rt
            choices_view[n, k, 0] = sign(y) # store choice

            # If the rt exceeds the deadline, set rt to -999 and choice to -1 
            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999

    if return_option == 'full':
        return {'rts': rts, 'choices': choices,  'metadata': {'v': v,
                                                            'a': a,
                                                            'z': z,
                                                            't': t,
                                                            'deadline': deadline,
                                                            's': s,
                                                            'delta_t': delta_t,
                                                            'max_t': max_t,
                                                            'n_samples': n_samples,
                                                            'n_trials': n_trials,
                                                            'simulator': 'ddm',
                                                            'boundary_fun_type': 'constant',
                                                            'possible_choices': [-1, 1],
                                                            'trajectory': traj}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices,  'metadata': {'simulator': 'ddm', 
                                                             'possible_choices': [-1, 1],
                                                             'boundary_fun_type': 'constant',
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,}}
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
                  boundary_multiplicative = True,
                  boundary_params = {},
                  random_state = None,
                  return_option = 'full',
                  smooth_unif  = False,
                  **kwargs,
                  ):
    """
    Simulate reaction times and choices from a drift diffusion model with flexible boundaries.

    Args:
        v (np.ndarray): Drift rate for each trial.
        a (np.ndarray): Boundary separation for each trial.
        z (np.ndarray): Starting point bias for each trial (between 0 and 1).
        t (np.ndarray): Non-decision time for each trial.
        deadline (np.ndarray): Maximum allowed reaction time for each trial.
        s (np.ndarray): Noise (sigma) for each trial.
        max_t (float): Maximum time for simulation.
        delta_t (float): Time step for simulation.
        n_samples (int): Number of samples to simulate per trial.
        n_trials (int): Number of trials to simulate.
        boundary_fun (callable): Function defining the shape of the boundary.
        boundary_multiplicative (bool): If True, boundary function is multiplied by 'a', else added to 'a'.
        boundary_params (dict): Parameters for the boundary function.
        random_state (int or None): Seed for random number generator.
        return_option (str): 'full' for complete output, 'minimal' for basic output.
        smooth_unif (bool): Whether to apply uniform smoothing to reaction times.
        **kwargs: Additional keyword arguments.

    Returns:
        dict: A dictionary containing simulated reaction times, choices, and metadata.
    """

    set_seed(random_state)
    #cdef int cov_length = np.max([v.size, a.size, w.size, t.size]).astype(int)
    # Param views:
    cdef float[:] v_view  = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s
    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:,:] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    #cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)

    cdef float y, t_particle, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n
    cdef Py_ssize_t ix
    cdef Py_ssize_t m = 0
    cdef Py_ssize_t k
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    cdef float[:] boundary_view = boundary

    # Loop over samples
    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
    
        deadline_tmp = min(max_t, deadline_view[k] - t_view[k]) 
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

            #rts_view[n, k, 0] = t_particle + t_view[k] # Store rt
            choices_view[n, k, 0] = sign(y) # Store choice

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999

    if return_option == 'full':
        return {'rts': rts, 'choices': choices,  'metadata': {'v': v,
                                                              'a': a,
                                                              'z': z,
                                                              't': t,
                                                              's': s,
                                                              'deadline': deadline,
                                                              **boundary_params,
                                                              'delta_t': delta_t,
                                                              'max_t': max_t,
                                                              'n_samples': n_samples,
                                                              'n_trials': n_trials,
                                                              'simulator': 'ddm_flexbound',
                                                              'boundary_fun_type': boundary_fun.__name__,
                                                              'possible_choices': [-1, 1],
                                                              'trajectory': traj,
                                                              'boundary': boundary,
                                                             }}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices,  'metadata': {'simulator': 'ddm_flexbound', 
                                                             'possible_choices': [-1, 1],
                                                             'boundary_fun_type': boundary_fun.__name__,
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             }}
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
             boundary_multiplicative = True,
             boundary_params = {},
             drift_params = {},
             random_state = None,
             return_option = 'full',
             smooth_unif  = False,
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
        boundary_multiplicative (bool): If True, boundary function is multiplicative; if False, additive.
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

    set_seed(random_state)
    # Param views:
    cdef float[:] v_view = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s

    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:,:] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    #cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    drift = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float y, t_particle, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n 
    cdef Py_ssize_t ix
    cdef Py_ssize_t m = 0
    cdef Py_ssize_t k
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    cdef float[:] boundary_view = boundary
    cdef float[:] drift_view = drift

    # Loop over samples
    for k in range(n_trials):
        # Precompute boundary evaluations and drift evaluations
        
        # Drift
        drift_params_tmp = {key: drift_params[key][k] for key in drift_params.keys()}
        drift[:] = np.add(v_view[k], drift_fun(t = t_s, **drift_params_tmp)).astype(DTYPE)

        # Boundary
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)

        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
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

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999
            
    if return_option == 'full':
        return {'rts': rts, 'choices': choices,  'metadata': {'v': v,
                                                            'a': a,
                                                            'z': z,
                                                            't': t,
                                                            'deadline': deadline,
                                                            's': s,
                                                            **boundary_params,
                                                            **drift_params,
                                                            'delta_t': delta_t,
                                                            'max_t': max_t,
                                                            'n_samples': n_samples,
                                                            'n_trials': n_trials,
                                                            'simulator': 'ddm_flex',
                                                            'boundary_fun_type': boundary_fun.__name__,
                                                            'drift_fun_type': boundary_fun.__name__,
                                                            'possible_choices': [-1, 1],
                                                            'trajectory': traj,
                                                            'drift': drift,
                                                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices,  'metadata': {'simulator': 'ddm_flex', 
                                                             'possible_choices': [-1, 1],
                                                             'boundary_fun_type': boundary_fun.__name__,
                                                             'drift_fun_type': boundary_fun.__name__,
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             }}
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
             boundary_multiplicative = True,
             boundary_params = {},
             drift_params = {},
             random_state = None,
             return_option = 'full',
             smooth_unif  = False,
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
        boundary_multiplicative (bool): If True, boundary function is multiplicative; if False, additive.
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

    set_seed(random_state)
    # Param views:
    cdef float[:] v_view = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] g_view = g
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s

    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:,:] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    #cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    drift = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float y, t_particle, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n 
    cdef Py_ssize_t ix
    cdef Py_ssize_t m = 0
    cdef Py_ssize_t k
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    cdef float[:] boundary_view = boundary
    cdef float[:] drift_view = drift

    # Loop over samples
    for k in range(n_trials):
        # Precompute boundary evaluations and drift evaluations
        
        # Drift
        drift_params_tmp = {key: drift_params[key][k] for key in drift_params.keys()}
        drift[:] = np.add(v_view[k], drift_fun(t = t_s, **drift_params_tmp)).astype(DTYPE)

        # Boundary
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)

        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
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

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999
    
    if return_option == 'full':
        return {'rts': rts, 'choices': choices,  'metadata': {'v': v,
                                                            'a': a,
                                                            'z': z,
                                                            'g': g,
                                                            't': t,
                                                            'deadline': deadline,
                                                            's': s,
                                                            **boundary_params,
                                                            **drift_params,
                                                            'delta_t': delta_t,
                                                            'max_t': max_t,
                                                            'n_samples': n_samples,
                                                            'n_trials': n_trials,
                                                            'simulator': 'ddm_flex_leak',
                                                            'boundary_fun_type': boundary_fun.__name__,
                                                            'drift_fun_type': boundary_fun.__name__,
                                                            'possible_choices': [-1, 1],
                                                            'trajectory': traj,
                                                            'drift': drift,
                                                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices,  'metadata': {'simulator': 'ddm_flex_leak', 
                                                             'possible_choices': [-1, 1],
                                                             'boundary_fun_type': boundary_fun.__name__,
                                                             'drift_fun_type': boundary_fun.__name__,
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             }}
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
    boundary_multiplicative = True,
    boundary_params = {},
    drift_params = {},
    random_state = None,
    return_option = 'full',
    smooth_unif  = False,
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
        boundary_multiplicative (bool): If True, boundary function is multiplicative; if False, additive.
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

    set_seed(random_state)
    # Param views:
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] g_t_view = gt
    cdef float[:] g_d_view = gd
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s

    cdef int num_draws = int((max_t / delta_t) + 1)
    
    traj = np.zeros((num_draws, 3), dtype = DTYPE)
    traj[:, :] = -999
    cdef float[:, :] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion

    # Boundary storage for the upper bound
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    drift = np.zeros((t_s.shape[0], 2), dtype = DTYPE)
    cdef float y_t, y_d, y_start, y, t_particle, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n 
    cdef Py_ssize_t ix
    cdef Py_ssize_t m = 0
    cdef Py_ssize_t k
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    cdef float[:] boundary_view = boundary
    cdef float[:, :] drift_view = drift

    # Loop over samples
    for k in range(n_trials):
        # Precompute boundary evaluations and drift evaluations
        
        # Drift
        drift_params_tmp = {key: drift_params[key][k] for key in drift_params.keys()}
        drift[:, :] = drift_fun(t = t_s, **drift_params_tmp).astype(DTYPE)

        # Boundary
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)

        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
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

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999
    
    if return_option == 'full':
        return {'rts': rts, 'choices': choices,  'metadata': {'vt': vt,
                                                            'vd': vd,
                                                            'a': a,
                                                            'z': z,
                                                            'gt': gt,
                                                            'gd': gd,
                                                            't': t,
                                                            'deadline': deadline,
                                                            's': s,
                                                            **boundary_params,
                                                            **drift_params,
                                                            'delta_t': delta_t,
                                                            'max_t': max_t,
                                                            'n_samples': n_samples,
                                                            'n_trials': n_trials,
                                                            'simulator': 'ddm_flex_leak',
                                                            'boundary_fun_type': boundary_fun.__name__,
                                                            'drift_fun_type': boundary_fun.__name__,
                                                            'possible_choices': [-1, 1],
                                                            'trajectory': traj,
                                                            'drift': drift,
                                                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices,  'metadata': {'simulator': 'ddm_flex_leak', 
                                                             'possible_choices': [-1, 1],
                                                             'boundary_fun_type': boundary_fun.__name__,
                                                             'drift_fun_type': boundary_fun.__name__,
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             }}
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
                boundary_multiplicative = True,
                boundary_params = {},
                random_state = None,
                return_option = 'full',
                smooth_unif = False,
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
        boundary_multiplicative (bool): If True, boundary function is multiplicative; if False, additive.
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

    set_seed(random_state)
    # cdef int cov_length = np.max([v.size, a.size, w.size, t.size]).astype(int)
    # Param views
    #set_random_state(random_state)
    cdef float[:] v_view = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s
    sz_samplewise = np.zeros((n_trials, n_samples), dtype = DTYPE)
    sv_samplewise = np.zeros((n_trials, n_samples), dtype = DTYPE)
    st_samplewise = np.zeros((n_trials, n_samples), dtype = DTYPE)

    cdef float[:, :] sz_samplewise_view = sz_samplewise
    cdef float[:, :] sv_samplewise_view = sv_samplewise
    cdef float[:, :] st_samplewise_view = st_samplewise

    # Data-structs for trajectory storage
    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:, :] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    #cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef float y, t_particle, t_tmp, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0
    cdef float drift_increment = 0.0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    # Loop over trials
    sv_samplewise[:, :] = v_dist(size = (n_samples, n_trials)).T
    sz_samplewise[:, :] = z_dist(size = (n_samples, n_trials)).T
    st_samplewise[:, :] = t_dist(size = (n_samples, n_trials)).T

    for k in range(n_trials):
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        if boundary_multiplicative:
            # print(a)
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            # print(a)
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)

        sqrt_st = delta_t_sqrt * s_view[k]

        # Loop over samples
        for n in range(n_samples):
            # displaced_starting_point
            y = (-1) * boundary_view[0] + ((z_view[k] + sz_samplewise_view[k, n]) * 2.0 * (boundary_view[0]))
            
            # displaced drift
            drift_increment = (v_view[k] + sv_samplewise_view[k, n]) * delta_t

            # displaced t
            t_tmp = t_view[k] + st_samplewise_view[k, n]
            deadline_tmp = min(max_t, deadline_view[k] - t_tmp)
            
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

            if smooth_unif:
                if t_particle == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif t_particle < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = t_particle + t_tmp + smooth_u # Store rt
            choices_view[n, k, 0] = np.sign(y) # Store choice

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999
    
    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'metadata': {'v': v,
                                                            'a': a,
                                                            'z': z,
                                                            't': t,
                                                            'z_dist': z_dist,
                                                            'v_dist': v_dist,
                                                            't_dist': t_dist,
                                                            'deadline': deadline,
                                                            's': s,
                                                            **boundary_params,
                                                            'delta_t': delta_t,
                                                            'max_t': max_t,
                                                            'n_samples': n_samples,
                                                            'n_trials': n_trials,
                                                            'simulator': 'full_ddm_rv',
                                                            'boundary_fun_type': boundary_fun.__name__,
                                                            'possible_choices': [-1, 1],
                                                            'trajectory': traj,
                                                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices,  'metadata': {'simulator': 'full_ddm_rv', 
                                                             'possible_choices': [-1, 1],
                                                             'boundary_fun_type': boundary_fun.__name__,
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             }}
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
             boundary_multiplicative = True,
             boundary_params = {},
             random_state = None,
             return_option = 'full',
             smooth_unif = False,
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
        ValueError: If return_option is neither 'full' nor 'minimal'.
    """

    set_seed(random_state)
    # cdef int cov_length = np.max([v.size, a.size, w.size, t.size]).astype(int)
    # Param views
    #set_random_state(random_state)
    cdef float[:] v_view  = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] sz_view = sz
    cdef float[:] sv_view = sv
    cdef float[:] st_view = st
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s

    # Data-structs for trajectory storage
    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:, :] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    #cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef float y, t_particle, t_tmp, smooth_u, deadline_tmp
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0
    cdef float drift_increment = 0.0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    # Loop over trials
    for k in range(n_trials):
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        if boundary_multiplicative:
            # print(a)
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            # print(a)
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        
        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
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

            if smooth_unif:
                if t_particle == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif t_particle < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = t_particle + t_tmp + smooth_u # Store rt
            choices_view[n, k, 0] = np.sign(y) # Store choice

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999
    
    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'metadata': {'v': v,
                                                            'a': a,
                                                            'z': z,
                                                            't': t,
                                                            'sz': sz,
                                                            'sv': sv,
                                                            'st': st,
                                                            'deadline': deadline,
                                                            's': s,
                                                            **boundary_params,
                                                            'delta_t': delta_t,
                                                            'max_t': max_t,
                                                            'n_samples': n_samples,
                                                            'n_trials': n_trials,
                                                            'simulator': 'full_ddm',
                                                            'boundary_fun_type': boundary_fun.__name__,
                                                            'possible_choices': [-1, 1],
                                                            'trajectory': traj,
                                                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices,  'metadata': {'simulator': 'full_ddm', 
                                                             'possible_choices': [-1, 1],
                                                             'boundary_fun_type': boundary_fun.__name__,
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             }}
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
            boundary_multiplicative = True,
            boundary_params = {},
            random_state = None,
            return_option = 'full',
            smooth_unif = False,
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
        ValueError: If return_option is neither 'full' nor 'minimal'.
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
    cdef float[:] t_view = t
    cdef float[:] sv_view = sv
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s
    
    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    #cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef float y, t_particle, smooth_u, deadline_tmp, sqrt_st
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0
    cdef float drift_increment = 0.0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)

        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
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

            if smooth_unif:
                if t_particle == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif t_particle < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u # Store rt
            choices_view[n, k, 0] = np.sign(y) # Store choice

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999

    if return_option == 'full':
        return { 'rts': rts, 'choices': choices, 'metadata': {'v': v,
                                                            'a': a,
                                                            'z': z,
                                                            't': t,
                                                            'sv': sv,
                                                            'deadline': deadline,
                                                            's': s,
                                                            **boundary_params,
                                                            'delta_t': delta_t,
                                                            'max_t': max_t,
                                                            'n_samples': n_samples,
                                                            'n_trials': n_trials,
                                                            'simulator': 'ddm_sdv',
                                                            'boundary_fun_type': boundary_fun.__name__,
                                                            'possible_choices': [-1, 1],
                                                            'trajectory': traj,
                                                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices,  'metadata': {'simulator': 'ddm_sdv', 
                                                             'possible_choices': [-1, 1],
                                                             'boundary_fun_type': boundary_fun.__name__,
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             }}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')


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
                           boundary_multiplicative = True,
                           boundary_params = {},
                           random_state = None,
                           return_option = 'full',
                           smooth_unif = False,
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
    boundary_multiplicative : bool, optional
        Whether the boundary function is multiplicative (default: True).
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
    cdef float[:] deadline_view = deadline
    cdef float[:] s_view = s
    # TD: Add trajectory --> same issue as with par2 model above... might need to make a separate simulator for trajectories

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    #cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    # Y particle trace
    bias_trace = np.zeros(num_draws, dtype = DTYPE)
    cdef float[:] bias_trace_view = bias_trace

    cdef float y_h, y_l, v_l, t_h, t_l, tmp_pos_dep, smooth_u, deadline_tmp
    cdef Py_ssize_t n, ix, ix_tmp, k
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

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add one deterministically
            # y at lower bound --> choice_view[n, k, 0] stays the same deterministically
            if random_uniform() <= ((y_l + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 1

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999

    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'metadata': {'vh': vh,
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
                                                            'trajectory': 'This simulator does not yet allow for trajectory simulation',
                                                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices, 'metadata': {'simulator': 'ddm_flexbound_mic2_adj', 
                                                             'possible_choices': [0, 1, 2, 3],
                                                             'boundary_fun_type': boundary_fun.__name__,
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             }}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# -----------------------------------------------------------------------------------------------