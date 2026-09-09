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

    sim_config = {'max_t': max_t, 'n_threads': 1}
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

    sim_config = {'max_t': max_t, 'n_threads': 1}
    params = {'a': a, 'z': z, 'theta': theta, 'deadline': deadline, 'sd': sd, 't': t}

    full_meta = build_full_metadata(
        minimal_metadata=minimal_meta,
        params=params,
        sim_config=sim_config,
        extra_params=v_dict
    )

    return build_return_dict(rts, choices, full_meta)

# Simulate (rt, choice) tuples from: Collapsing bound angle LBA Model with
# relative starting-point range -----------------------------------------------

def dev_lba_angle_v2(np.ndarray[float, ndim = 2] v,
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
    Simulate RTs and choices from an angle-LBA with a *relative* start-point range.

    This model is identical to :func:`lba_angle` except in how the start-point
    parameter ``z`` is interpreted. Here ``z`` is the width of the uniform
    start-point distribution expressed as a **fraction of the threshold** ``a``,
    so start points are drawn as ``U(0, z * a)`` rather than ``U(0, z)``.

    Because ``z`` is relative, it is automatically consistent with ``a``: any
    ``z`` in ``[0, 1]`` keeps start points below threshold for every ``a``. That
    decouples the two parameters, which is what allows this model's configuration
    (``dev_lba_angle_3_v2``) to use a much wider threshold range
    (``a`` up to 3.0, ``z`` up to 0.9) than ``lba_angle_3`` can, and why it needs
    no ``a``/``z`` swap constraint during parameter sampling.

    Drift rates are drawn per accumulator as ``|N(v, sd)|`` (absolute value taken
    to avoid negative drifts), and each accumulator's finishing time is

    .. math::
        x_i = \\frac{a - z_i}{v_i + \\tan(\\theta)}

    where the ``tan(theta)`` term implements the collapsing (angled) boundary.
    The observed response is the fastest accumulator, ``argmin(x)``, with
    ``rt = min(x) + t``. Responses at or beyond ``deadline`` are marked with the
    omission sentinel ``-999``.

    Parameters
    ----------
    v : np.ndarray[float, ndim=2]
        Drift rate for each accumulator, shape ``(n_trials, nact)``.
    a : np.ndarray[float, ndim=2]
        Decision threshold, shape ``(n_trials, 1)``.
    z : np.ndarray[float, ndim=2]
        Start-point range as a *fraction of* ``a``, in ``[0, 1]``,
        shape ``(n_trials, 1)``. Start points are drawn from ``U(0, z * a)``.
    theta : np.ndarray[float, ndim=2]
        Angle parameter for the collapsing bound, shape ``(n_trials, 1)``.
    deadline : np.ndarray[float, ndim=1]
        Maximum allowed decision time, shape ``(n_trials,)``.
    sd : np.ndarray[float, ndim=2]
        Standard deviation of the drift rate distribution,
        shape ``(n_trials, nact)``.
    t : np.ndarray[float, ndim=1]
        Non-decision time, shape ``(n_trials,)``.
    nact : int, optional
        Number of accumulators (default is 3).
    n_samples : int, optional
        Number of samples to generate per trial (default is 2000).
    n_trials : int, optional
        Number of trials / parameter sets (default is 1).
    max_t : float, optional
        Maximum time to simulate (default is 20).

    Returns
    -------
    dict
        A dictionary containing:
        - 'rts': simulated reaction times
        - 'choices': simulated choices
        - 'metadata': additional information about the simulation

    Notes
    -----
    Ported from the ``dev_lba_angle_3_v2`` development branch of
    ``ssm-simulators`` (``origin/dev_lba_angle_3_v2``, commits 97e1e45/10b288c),
    which predates the modularization of ``config.py`` and ``cssm.pyx``. The
    original took 1-D ``a``, ``z`` and ``theta``; this port takes 2-D arrays to
    match the current ``lba_models`` module convention and the shapes produced by
    ``ExpandDimension``. Simulated output is bit-for-bit identical to the
    original.

    Like every other simulator in this module, this function does not consume
    ``random_state``; draws come from NumPy's global legacy random stream.

    See Also
    --------
    lba_angle : same model with an absolute (non-relative) start-point range.

    Examples
    --------
    >>> from ssms import Simulator
    >>> sim = Simulator("dev_lba_angle_3_v2")
    >>> out = sim.simulate(
    ...     theta={"v0": 0.5, "v1": 0.3, "v2": 0.2, "a": 0.5, "z": 0.2, "theta": 0.0},
    ...     n_samples=1000,
    ... )
    >>> out["rts"].shape
    (1000, 1)
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
    cdef float a_tmp, z_range_tmp
    # NOTE: tan(theta) is deliberately left as an untyped (float64) Python
    # object rather than a `cdef float`. The original implementation kept it
    # in double precision inside the expression below; rounding it to float32
    # here would perturb the low-order bits of every RT.

    for k in range(n_trials):
        a_tmp = a_view[k, 0]
        # Start-point range is RELATIVE to the threshold: this is the single
        # line that distinguishes this model from `lba_angle`.
        z_range_tmp = z_view[k, 0] * a_view[k, 0]
        tan_theta_tmp = np.tan(theta_view[k, 0])

        for n in range(n_samples):
            zs = np.random.uniform(0, z_range_tmp, nact)

            vs = np.abs(np.random.normal(v_view[k], sd_view[k])) # np.abs() to avoid negative vs
            x_t = (a_tmp - zs) / (vs + tan_theta_tmp)

            choices_view[n, k, 0] = np.argmin(x_t) # store choices for sample n
            rts_view[n, k, 0] = np.min(x_t) + t_view[k] # store reaction time for sample n

            # If the rt exceeds the deadline, mark it as an omission
            if rts_view[n, k, 0] >= deadline_view[k]:
                rts_view[n, k, 0] = -999

    # Build v_dict dynamically
    v_dict = build_param_dict_from_2d_array(v, 'v_', nact)

    # LBA models always return full metadata (no return_option)
    minimal_meta = build_minimal_metadata(
        simulator_name='dev_lba_angle_v2',
        possible_choices=list(np.arange(0, nact, 1)),
        n_samples=n_samples,
        n_trials=n_trials
    )

    sim_config = {'max_t': max_t, 'n_threads': 1}
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

    sim_config = {'max_t': max_t, 'n_threads': 1}
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

    sim_config = {'max_t': max_t, 'n_threads': 1}
    params = {'a': a, 'z': z, 't': t, 'deadline': deadline, 'sd': sd}

    full_meta = build_full_metadata(
        minimal_metadata=minimal_meta,
        params=params,
        sim_config=sim_config,
        extra_params=v_dict
    )

    return build_return_dict(rts, choices, full_meta)

# ----------------------------------------------------------------------------------------------------
