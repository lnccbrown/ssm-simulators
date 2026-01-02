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
from libc.math cimport log, sqrt, pow, fmax

import numpy as np
cimport numpy as np

# Import utility functions from the _utils module
from cssm._utils import (
    set_seed,
    random_uniform,
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
        **kwargs: Additional keyword arguments.

    Returns:
        dict: A dictionary containing simulated reaction times, choices, and metadata.
              The exact contents depend on the return_option.

    Raises:
        ValueError: If return_option is neither 'full' nor 'minimal'.
    """

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
    cdef float[:] alpha_stable_values = draw_random_stable(num_draws, alpha_view[0])

    for k in range(n_trials):
        # AF-TODO: check if this is correct
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
                                           delta_t
                                           )

            rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u # Store rt
            choices_view[n, k, 0] = sign(y) # Store choice
            enforce_deadline(rts_view,
                             deadline_view,
                             n,
                             k,
                             0
                            )

    # Build minimal metadata first
    minimal_meta = build_minimal_metadata(
        simulator_name='levy_flexbound',
        possible_choices=[-1, 1],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=boundary_fun.__name__
    )

    if return_option == 'full':
        sim_config = {'delta_t': delta_t, 'max_t': max_t}
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
