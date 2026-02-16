# Globaly settings for cython
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: initializedcheck=False

"""
Poisson race model simulators.

This module implements simulators for Poisson race decision models, in which
multiple Poisson processes race to reach a criterion number of events and the
first to finish determines the choice and response time.
"""
import numpy as np

DTYPE = np.float32

# Import utility functions from the _utils module
from cssm._utils import (
    set_seed,
    compute_smooth_unif,
    enforce_deadline,
    compute_deadline_tmp,
    build_param_dict_from_2d_array,
    build_full_metadata,
    build_minimal_metadata,
    build_return_dict,
)
# @cython.boundscheck(False)
# @cython.wraparound(False)
def poisson_race(
    r,  # rate parameters
    k,  # shape parameters
    t,  # non-decision times
    s = None,               # unused, kept for interface compatibility
    deadline = None,        # deadline per trial
    float delta_t = 0.001,  # time-step size in simulator
    float max_t = 20.0,     # maximal time
    int n_samples = 2000,   # number of samples to produce
    int n_trials = 1,       # number of trials
    boundary_fun = None,    # unused, kept for interface compatibility
    boundary_multiplicative = True,
    boundary_params = {},
    random_state = None,
    return_option = 'full',
    smooth_unif = False,
    **kwargs
):
    """
    Simulate response times and choices for a 2-choice Poisson race model.

    Each accumulator is a Poisson process with rate ``r``; the finishing time is the
    time to the ``k``-th event, i.e., a Gamma(shape=k, scale=1/r). The choice is the
    accumulator with the smaller finishing time; exact ties are broken at random.

    Notes:
        The ``s`` parameter is unused and only present for interface compatibility.
    """

    if r is None or k is None or t is None or deadline is None:
        raise ValueError("poisson_race requires r, k, t, and deadline.")

    # Ensure arrays are float32 and shaped correctly
    r = np.asarray(r, dtype=DTYPE)
    k = np.asarray(k, dtype=DTYPE)
    t = np.asarray(t, dtype=DTYPE)
    deadline = np.asarray(deadline, dtype=DTYPE)
    if s is None:
        s = np.zeros_like(r, dtype=DTYPE)
    else:
        s = np.asarray(s, dtype=DTYPE)

    set_seed(random_state)
    rng = np.random.default_rng(random_state)

    if r.ndim != 2 or k.ndim != 2 or t.ndim != 2 or r.shape[1] != 2 or k.shape[1] != 2:
        raise ValueError("poisson_race currently supports exactly two accumulators and 2D inputs for r, k, t.")
    if s.shape != r.shape:
        raise ValueError("s must match shape of r.")
    if n_trials <= 0 or n_samples <= 0:
        raise ValueError("n_trials and n_samples must be positive.")
    if (not np.isfinite(r).all()) or (r <= 0).any():
        raise ValueError("All rate parameters must be finite and > 0.")
    if (not np.isfinite(k).all()) or (k <= 0).any():
        raise ValueError("All k parameters must be finite and > 0.")
    if (not np.isfinite(t).all()) or (t < 0).any():
        raise ValueError("All non-decision times must be finite and >= 0.")

    n_trials = r.shape[0]  # align with provided rates

    cdef float[:, :] r_view = r
    cdef float[:, :] k_view = k
    cdef float[:, :] t_view = t
    cdef float[:, :] s_view = s
    cdef float[:] deadline_view = deadline

    rts = np.zeros((n_samples, n_trials, 1), dtype=DTYPE)
    cdef float[:, :, :] rts_view = rts
    choices = np.zeros((n_samples, n_trials, 1), dtype=np.intc)
    cdef int[:, :, :] choices_view = choices

    finish_times = np.zeros((n_samples, 2), dtype=DTYPE)
    min_times = np.zeros(n_samples, dtype=DTYPE)
    winners = np.zeros(n_samples, dtype=np.intc)
    cdef float[:] min_times_view = min_times
    cdef int[:] winners_view = winners

    cdef Py_ssize_t trial_ix, sample_ix
    cdef float deadline_tmp, smooth_u, rt_value

    for trial_ix in range(n_trials):
        deadline_tmp = compute_deadline_tmp(max_t, deadline_view[trial_ix], t_view[trial_ix, 0])

        finish_times[:, :] = rng.gamma(
            shape=np.asarray(k_view[trial_ix, :], dtype=np.float32),
            scale=np.asarray(np.reciprocal(r_view[trial_ix, :]), dtype=np.float32),
            size=(n_samples, 2),
        ).astype(DTYPE)

        # Determine winners and minimum finishing times
        lt = finish_times[:, 0] < finish_times[:, 1]
        gt = finish_times[:, 0] > finish_times[:, 1]
        tie = ~(lt | gt)

        winners[lt] = 0
        winners[gt] = 1
        if tie.any():
            winners[tie] = rng.integers(0, 2, size=tie.sum(), endpoint=False, dtype=np.int64)

        min_times[:] = np.minimum(finish_times[:, 0], finish_times[:, 1])

        for sample_ix in range(n_samples):
            smooth_u = compute_smooth_unif(smooth_unif, min_times_view[sample_ix], deadline_tmp, delta_t)
            rt_value = min_times_view[sample_ix] + t_view[trial_ix, 0] + smooth_u
            if min_times_view[sample_ix] > deadline_tmp:
                rts_view[sample_ix, trial_ix, 0] = -999
            else:
                rts_view[sample_ix, trial_ix, 0] = rt_value
            if winners_view[sample_ix] == 0:
                choices_view[sample_ix, trial_ix, 0] = -1
            else:
                choices_view[sample_ix, trial_ix, 0] = 1
            enforce_deadline(rts_view, deadline_view, sample_ix, trial_ix, 0)

    minimal_meta = build_minimal_metadata(
        simulator_name='poisson_race',
        possible_choices=[-1, 1],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=None
    )

    if return_option == 'full':
        r_dict = build_param_dict_from_2d_array(r, 'r', 2)
        k_dict = build_param_dict_from_2d_array(k, 'k', 2)

        sim_config = {'delta_t': delta_t, 'max_t': max_t}
        params = {'t': t, 'deadline': deadline, 's': s}
        full_meta = build_full_metadata(
            minimal_metadata=minimal_meta,
            params=params,
            sim_config=sim_config,
            extra_params={**r_dict, **k_dict}
        )
        return build_return_dict(rts, choices, full_meta)

    elif return_option == 'minimal':
        return build_return_dict(rts, choices, minimal_meta)

    else:
        raise ValueError('return_option must be either "full" or "minimal"')