# Globaly settings for cython
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: initializedcheck=False

"""
Linear Ballistic Accumulator (LBA) Models

This module contains simulator functions for Linear Ballistic Accumulator (LBA) models.
LBA models are simplified race models where evidence accumulates deterministically
(no within-trial noise) from random starting points toward fixed thresholds.
"""

import cython
from libc.math cimport sqrt, log, exp, fmax, sin, cos, atan

import numpy as np
cimport numpy as np

# Import utility functions from the _utils module
from cssm._utils import (
    set_seed,
    random_uniform,
    draw_gaussian,
    build_param_dict_from_2d_array,
    build_full_metadata,
    build_minimal_metadata,
    build_return_dict,
)

DTYPE = np.float32

# LBA Models ------------------------------------

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


    # Build v_dict dynamically
    v_dict = build_param_dict_from_2d_array(v, 'v_', nact)

    # LBA models always return full metadata (no return_option)
    minimal_meta = build_minimal_metadata(
        simulator_name='lba_vanilla',
        possible_choices=list(np.arange(0, nact, 1)),
        n_samples=n_samples,
        n_trials=n_trials
    )

    sim_config = {'max_t': max_t}
    params = {'a': a, 'z': z, 'deadline': deadline, 'sd': sd, 't': t}

    full_meta = build_full_metadata(
        minimal_metadata=minimal_meta,
        params=params,
        sim_config=sim_config,
        extra_params=v_dict
    )

    return build_return_dict(rts, choices, full_meta)



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

    # Build v_dict dynamically
    v_dict = build_param_dict_from_2d_array(v, 'v_', nact)

    # LBA models always return full metadata (no return_option)
    minimal_meta = build_minimal_metadata(
        simulator_name='lba_angle',
        possible_choices=list(np.arange(0, nact, 1)),
        n_samples=n_samples,
        n_trials=n_trials
    )

    sim_config = {'max_t': max_t}
    params = {'a': a, 'z': z, 'theta': theta, 'deadline': deadline, 'sd': sd, 't': t}

    full_meta = build_full_metadata(
        minimal_metadata=minimal_meta,
        params=params,
        sim_config=sim_config,
        extra_params=v_dict
    )

    return build_return_dict(rts, choices, full_meta)


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
    """
    Simulate reaction times and choices from a piecewise RLWM Linear Ballistic Accumulator (LBA) model.

    This LBA model simulates a hybrid of reinforcement learning (RL) and working memory (WM) processes.
    On each trial, accumulation for each accumulator starts at a random position drawn uniformly
    between [0, z], with separate drift rates for RL and WM. Before time tWM, only RL accumulates;
    after tWM, RL and WM accumulate in parallel (summed drift).

    Args:
        vRL (np.ndarray[float, ndim=2]):
            RL drift rates for each accumulator and trial.
        vWM (np.ndarray[float, ndim=2]):
            WM drift rates for each accumulator and trial.
        a (np.ndarray[float, ndim=2]):
            Decision threshold (criterion height) for each trial and accumulator.
        z (np.ndarray[float, ndim=2]):
            Starting point upper bound for each trial and accumulator.
        tWM (np.ndarray[float, ndim=2]):
            Switching time to parallel RL+WM accumulation (per trial and accumulator).
        deadline (np.ndarray[float, ndim=1]):
            Maximum allowed decision time for each trial.
        sd (np.ndarray[float, ndim=2]):
            Standard deviation of drift rates for each accumulator and trial.
        t (np.ndarray[float, ndim=1]):
            Non-decision time (per trial).
        nact (int, optional):
            Number of accumulators (default: 3).
        n_samples (int, optional):
            Number of samples to simulate per trial (default: 2000).
        n_trials (int, optional):
            Number of simulated trials (default: 1).
        max_t (float, optional):
            Maximum time for simulation (default: 20).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        dict: Dictionary containing the following keys:
            'rts': Simulated reaction times (shape: [n_samples, n_trials, 1])
            'choices': Simulated choices (shape: [n_samples, n_trials, 1])
            'metadata': Simulation metadata dictionary (full_meta)
    """

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

    # LBA models always return full metadata (no return_option)
    minimal_meta = build_minimal_metadata(
        simulator_name='rlwm_lba_pw_v1',
        possible_choices=list(np.arange(0, nact, 1)),
        n_samples=n_samples,
        n_trials=n_trials
    )

    sim_config = {'max_t': max_t}
    params = {'a': a, 'z': z, 'tWM': tWM, 't': t, 'deadline': deadline, 'sd': sd}

    full_meta = build_full_metadata(
        minimal_metadata=minimal_meta,
        params=params,
        sim_config=sim_config,
        extra_params=v_dict
    )

    return build_return_dict(rts, choices, full_meta)

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

    # LBA models always return full metadata (no return_option)
    minimal_meta = build_minimal_metadata(
        simulator_name='rlwm_lba_race',
        possible_choices=list(np.arange(0, nact, 1)),
        n_samples=n_samples,
        n_trials=n_trials
    )

    sim_config = {'max_t': max_t}
    params = {'a': a, 'z': z, 't': t, 'deadline': deadline, 'sd': sd}

    full_meta = build_full_metadata(
        minimal_metadata=minimal_meta,
        params=params,
        sim_config=sim_config,
        extra_params=v_dict
    )

    return build_return_dict(rts, choices, full_meta)

# ----------------------------------------------------------------------------------------------------
