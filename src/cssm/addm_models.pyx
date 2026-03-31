# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: initializedcheck=False

"""
Attentional Drift Diffusion Model (ADDM) Simulator.

Model parameters (per-trial arrays of length n_trials):
    eta   – attention discount
    kappa – drift scaling
    a     – boundary height (upper = +a, lower = -a)
    z     – starting point bias  (relative, in [0,1]; 0.5 = unbiased)
    t     – non-decision time (added to simulated RT)

All other quantities (r1, r2, fixation durations, flag, mu1, mu2) are
sampled internally on every (trial, sample) pair.

The function signature and return format mirror ddm_models.pyx so that
the rest of the pipeline (dataset generation, config, etc.) can treat
this model identically.
"""

import cython
from libc.math cimport sqrt, fabs, log, exp, fmax, fmin
from libc.stdlib cimport rand, RAND_MAX
import numpy as np
cimport numpy as np

# Import utility functions from the _utils module
from cssm._utils import (
    set_seed,
    random_uniform,
    draw_gaussian,
    draw_uniform,
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


include "_rng_wrappers.pxi"
include "_constants.pxi"


DTYPE = np.float32


# ---------------------------------------------------------------------------
# Internal C-level helper: look up piecewise-constant drift at time t
# ---------------------------------------------------------------------------
cdef double _piecewise_drift(double t,
                             double* mu_arr,
                             double* sacc_arr,
                             int d) nogil:
    cdef int i
    for i in range(d - 1, -1, -1):
        if t >= sacc_arr[i]:
            return mu_arr[i]
    return mu_arr[0]



def addm(np.ndarray[float, ndim=1] eta,
                  np.ndarray[float, ndim=1] kappa,
                  np.ndarray[float, ndim=1] a,
                  np.ndarray[float, ndim=1] z,
                  np.ndarray[float, ndim=1] b,
                  np.ndarray[float, ndim=1] t,
                  np.ndarray[float, ndim=1] deadline,
                  np.ndarray[float, ndim=1] s,
                  # ---- simulation settings (not model params) ----
                  float gamma_shape = 6.0,
                  float gamma_scale = 0.1,
                  int max_fixations = 100,
                  float delta_t = 0.0001,
                  float max_t = 20.0,
                  int n_samples = 1000,
                  int n_trials = 10,
                  return_option = 'full',
                  smooth_unif = False,
                  random_state = None,
                  int n_threads = 1,
                  **kwargs):
    """
    Simulate the aDDM with constant symmetric boundaries.

    Returns a dict with the same structure as ddm_constant so that
    downstream code (dataset builders, etc.) can consume it uniformly.

    The returned ``rts`` array has shape ``(n_samples, n_trials, 1)``
    and ``choices`` has the same shape, following the convention in
    ddm_models.pyx.
    """


    cdef uint64_t seed = random_state if random_state is not None else np.random.randint(0, 2**31)
    rng = np.random.default_rng(seed)
    setup = setup_simulation(n_samples, n_trials, max_t, delta_t, random_state)
    traj = setup['traj']
    rts = setup['rts'] 
    choices = setup['choices']
    cdef float[:, :] traj_view = traj 
    cdef float[:, :, :] rts_view     = rts
    cdef int[:, :, :] choices_view = choices
    cdef float[:] gaussian_values = setup['gaussian_values']
    cdef float[:] uniform_values  = setup['uniform_values']
    t_s = setup['t_s']
    cdef int num_draws = setup['num_draws']
    cdef float delta_t_sqrt = setup['delta_t_sqrt']

    # Param Views 
    cdef float[:] eta_view   = eta
    cdef float[:] kappa_view = kappa
    cdef float[:] a_view     = a
    cdef float[:] z_view     = z
    cdef float[:] b_view     = b
    cdef float[:] t_view     = t
    cdef float[:] s_view     = s
    cdef float[:] deadline_view = deadline

   
    cdef int k, n, ix, d, flag, r1, r2
    cdef float y, t_particle, drift_val, deadline_tmp, sqrt_st
    cdef float mu1, mu2
    cdef int m = 0
    cdef int mu = 0

    cdef double[::1] mu_view
    cdef double[::1] sacc_view

    for k in range(n_trials):
        deadline_tmp = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])
        sqrt_st = delta_t_sqrt * s_view[k]
        for n in range(n_samples):
            # =========================================================
            # (a) Sample latent trial-level variables
            # =========================================================
            r1   = int(rng.integers(1, 6, endpoint=False))
            r2   = int(rng.integers(1, 6, endpoint=False))
            flag = int(rng.integers(0, 2))            # 0 or 1

            mu1 = kappa_view[k] * (r1 - eta_view[k] * r2)
            mu2 = kappa_view[k] * (eta_view[k] * r1 - r2)

            # ---- fixation durations → saccade switch-times ----------
            fixations_np  = rng.gamma(gamma_shape, gamma_scale, max_fixations)
            sacc_full     = np.insert(np.cumsum(fixations_np), 0, 0.0)
            sacc_np       = sacc_full[sacc_full < max_t].astype(np.float64)

            d = len(sacc_np)
            if d == 0:
                d = 1
                sacc_np = np.array([0.0], dtype=np.float64)

            # ---- alternating drift array ----------------------------
            mu_np = np.empty(d, dtype=np.float64)
            for i in range(d):
                if flag == 0:
                    mu_np[i] = mu1 if (i % 2 == 0) else mu2
                else:
                    mu_np[i] = mu2 if (i % 2 == 0) else mu1

            # typed views for the C-level helper
            mu_view   = mu_np
            sacc_view = sacc_np

           
            # =========================================================
            # (b) Euler-Maruyama diffusion (constant boundaries)
            # =========================================================
            y = z_view[k]   # actual starting position
            t_particle = 0.0
            ix = 0
            while y <= a_view[k] - b_view[k] * t_particle and y >= -a_view[k] + b_view[k] * t_particle and t_particle <= deadline_tmp:
                # --- step the particle ---
                drift_val = _piecewise_drift(t_particle,
                                             &mu_view[0],
                                             &sacc_view[0],
                                             d)
                y += drift_val * delta_t + sqrt_st * gaussian_values[ix]
                t_particle += delta_t
                m += 1 
                ix += 1 

                if n == 0: 
                    if k == 0: 
                        traj_view[ix, 0] = y 

                if m == num_draws: 
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            # Apply smoothing
            smooth_u = compute_smooth_unif(smooth_unif, t_particle, deadline_tmp, delta_t, uniform_values[mu])
            mu += 1 
            if mu == num_draws: 
                uniform_values = draw_uniform(num_draws)
                mu = 0 
            
            rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u
            choices_view[n, k, 0] = sign(y)
        
    minimal_meta = build_minimal_metadata(
        simulator_name = 'addm', 
        possible_choices = [-1, 1],
        n_samples = n_samples, 
        n_trials = n_trials,
        boundary_fun_name = 'constant'
    )
    if return_option == 'full': 
        sim_config = {'delta_t': delta_t, 'max_t': max_t}
        params = {
            'eta': eta,
            'kappa': kappa,
            'a': a,
            'z': z,
            'b': b,
            't': t,
            's': s
        }
        full_meta = build_full_metadata(minimal_metadata= minimal_meta, sim_config=sim_config, params=params, traj=traj)
        return build_return_dict(rts, choices, full_meta)

    elif return_option == 'minimal': 
        return build_return_dict(rts, choices, minimal_meta)

    else: 
        raise ValueError(f"Invalid return_option: {return_option}. Must be 'full' or 'minimal'.")