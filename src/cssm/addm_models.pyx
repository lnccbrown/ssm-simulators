# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""Attentional Drift Diffusion Model (aDDM) simulator.

The simulation engine (inline xoshiro256++/Box-Muller PRNG + the stage-indexed
``_run_heterog_trial`` / ``_simulate_heterog_multistage`` kernel) is **vendored
verbatim from efficient-fpt** (efpt) @ commit ``d97a451``, MIT (c) 2025 Sicheng
Liu — the same in-house source as HSSM's JAX aDDM likelihood, so simulator and
likelihood share one engine (tight sim<->likelihood parity). efpt is an in-house
ecosystem project; the intended end-state is to absorb & relicense it under the
ecosystem license (TODO: author sign-off). Do not edit the vendored engine in
place; re-vendor and re-apply the ssm-simulators glue below.

Why efpt's xoshiro rather than ssm-simulators' GSL RNG (``_rng_wrappers.pxi``):
per-trial-seeded xoshiro256++ gives results that are (a) identical to efpt's own
simulator on the same seeds — enabling a fixed-seed parity oracle — and (b)
deterministic regardless of ``n_threads``. It needs no GSL.

Canonical aDDM parameter contract (matches HSSM's sampled columns):
    eta   – attention discount            kappa – drift scaling
    a     – boundary intercept at t=0      b     – boundary collapse slope (>= 0)
    x0    – starting position (ABSOLUTE, default 0.0; was the mislabeled 'z')
    t     – non-decision time (added to the simulated RT)
Diffusion ``sigma`` is a fixed scalar (per ssm-simulators it rides the ``s`` noise
slot; Mode 2 / PPC may pin it explicitly). Boundaries collapse as +-(a - b*t)
(the node-anchored slopes in the engine cancel to this wall-clock form), matching
the JAX likelihood.

Two modes, one engine:
* Mode 1 (self-sampled, DEFAULT): r1, r2, flag, and fixation onsets are drawn
  internally per (sample, trial) — prior / dataset generation.
* Mode 2 (covariate-conditioned): observed r1, r2, flag, sacc_array, d are passed
  in (HSSM extra_fields) and the engine only samples the trajectory — used for
  posterior-predictive checks conditioned on the real fixations.
Mode 2 is selected when ``sacc_array`` is not None.
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, cos, sin, M_PI
from libc.stdint cimport uint64_t
from cython.parallel cimport prange

from cssm._utils import (
    setup_simulation,
    build_minimal_metadata,
    build_full_metadata,
    build_return_dict,
)

# ssm-simulators' omission sentinel (RT for a trial that did not terminate).
cdef double OMISSION = -999.0


# ===========================================================================
# Vendored from efficient-fpt @ d97a451 (src/efficient_fpt/cython/simulator.pyx)
# xoshiro256++ seeded per trial via SplitMix64 + Box-Muller Gaussian transform.
# Do not edit in place — re-vendor.
# ===========================================================================
cdef struct Xoshiro256State:
    uint64_t s0
    uint64_t s1
    uint64_t s2
    uint64_t s3


cdef inline uint64_t _rotl(uint64_t x, int k) noexcept nogil:
    return (x << k) | (x >> (64 - k))


cdef inline uint64_t xoshiro256pp_next(Xoshiro256State *state) noexcept nogil:
    cdef uint64_t result = _rotl(state.s0 + state.s3, 23) + state.s0
    cdef uint64_t t = state.s1 << 17
    state.s2 ^= state.s0
    state.s3 ^= state.s1
    state.s1 ^= state.s2
    state.s0 ^= state.s3
    state.s2 ^= t
    state.s3 = _rotl(state.s3, 45)
    return result


cdef inline uint64_t splitmix64_next(uint64_t *state) noexcept nogil:
    state[0] += <uint64_t>0x9e3779b97f4a7c15
    cdef uint64_t z = state[0]
    z = (z ^ (z >> 30)) * <uint64_t>0xbf58476d1ce4e5b9
    z = (z ^ (z >> 27)) * <uint64_t>0x94d049bb133111eb
    return z ^ (z >> 31)


cdef inline void seed_xoshiro256(Xoshiro256State *state, uint64_t seed) noexcept nogil:
    cdef uint64_t sm_state = seed
    state.s0 = splitmix64_next(&sm_state)
    state.s1 = splitmix64_next(&sm_state)
    state.s2 = splitmix64_next(&sm_state)
    state.s3 = splitmix64_next(&sm_state)


cdef inline double uint64_to_double(uint64_t x) noexcept nogil:
    return <double>(x >> 11) * (1.0 / 9007199254740992.0)  # 2^53


cdef struct BoxMullerState:
    double spare
    int has_spare


cdef inline double box_muller_next(Xoshiro256State *rng_state, BoxMullerState *bm_state) noexcept nogil:
    cdef double u1, u2, mag
    if bm_state.has_spare:
        bm_state.has_spare = 0
        return bm_state.spare
    u1 = uint64_to_double(xoshiro256pp_next(rng_state))
    u2 = uint64_to_double(xoshiro256pp_next(rng_state))
    if u1 < 1e-300:
        u1 = 1e-300
    mag = sqrt(-2.0 * log(u1))
    bm_state.spare = mag * sin(2.0 * M_PI * u2)
    bm_state.has_spare = 1
    return mag * cos(2.0 * M_PI * u2)


cdef void _run_heterog_trial(
    double[:, ::1] mu_array_data,
    double[:, ::1] sigma_array_data,
    double[:, ::1] node_array_data,
    int d,
    double[:, ::1] ub_array_data,
    double[:, ::1] b1_array_data,
    double[:, ::1] lb_array_data,
    double[:, ::1] b2_array_data,
    int trial_idx,
    double x0,
    double dt,
    int max_steps,
    double T,
    uint64_t seed,
    double *rt_out,
    int *choice_out,
    double *x_final_out,
    float *traj_out,  # ssm-sim MOD (not efpt): optional per-step trajectory sink; NULL = don't record
) noexcept nogil:
    """Single stage-indexed trial (nogil). Piecewise-constant drift/diffusion,
    piecewise-linear boundaries; stage advances when the particle clock passes the
    next node. Vendored from efpt simulator.pyx:_run_heterog_trial.

    ssm-sim MOD: the trailing ``traj_out`` param records ``y`` per step (for the
    model-cartoon path); when NULL the loop is byte-for-byte the efpt original.
    Re-apply this param + the one write below on re-vendor."""
    cdef:
        Xoshiro256State rng_state
        BoxMullerState bm_state
        double y, z, t_particle, upper, lower, dt_curr, sqrt_dt_curr, half_dt_curr
        int step, stage

    seed_xoshiro256(&rng_state, seed)
    bm_state.has_spare = 0
    y = x0
    t_particle = 0.0
    stage = 0
    rt_out[0] = -1.0
    choice_out[0] = 0

    for step in range(max_steps):
        dt_curr = T - t_particle
        if dt_curr <= 0.0:
            break
        if dt_curr > dt:
            dt_curr = dt
        sqrt_dt_curr = sqrt(dt_curr)
        half_dt_curr = 0.5 * dt_curr
        if traj_out != NULL:  # ssm-sim MOD (not efpt): record y at time step*dt (traj[0]=x0), aligned to t_s
            traj_out[step] = <float>y
        z = box_muller_next(&rng_state, &bm_state)
        y = y + mu_array_data[trial_idx, stage] * dt_curr + sigma_array_data[trial_idx, stage] * sqrt_dt_curr * z
        t_particle = t_particle + dt_curr

        upper = ub_array_data[trial_idx, stage] + b1_array_data[trial_idx, stage] * (t_particle - node_array_data[trial_idx, stage])
        lower = lb_array_data[trial_idx, stage] + b2_array_data[trial_idx, stage] * (t_particle - node_array_data[trial_idx, stage])
        if y >= upper:
            rt_out[0] = t_particle - half_dt_curr
            choice_out[0] = 1
            break
        elif y <= lower:
            rt_out[0] = t_particle - half_dt_curr
            choice_out[0] = -1
            break

        while stage + 1 < d and t_particle >= node_array_data[trial_idx, stage + 1]:
            stage = stage + 1

    x_final_out[0] = y


def _simulate_heterog_multistage(
    double[:, ::1] mu_array_data,
    double[:, ::1] sigma_array_data,
    double[:, ::1] node_array_data,
    int[::1] d_data,
    double[:, ::1] ub_array_data,
    double[:, ::1] b1_array_data,
    double[:, ::1] lb_array_data,
    double[:, ::1] b2_array_data,
    double[::1] x0_data,
    double dt,
    double T,
    uint64_t[::1] trial_seeds,
    int n_threads=1,
    float[:, ::1] traj_out=None,  # ssm-sim MOD (not efpt): optional trajectory sink for `record_trial`
    int record_trial=-1,          # row index to record into traj_out; -1 = record nothing (default)
):
    """Batch of heterogeneous multi-stage trials via OpenMP prange. Per-trial
    xoshiro seeds make output independent of ``n_threads``. Vendored from efpt
    simulator.pyx:simulate_heterog_multistage_fpt. Returns (rt, choice, x_final)
    with rt = -1.0 / choice = 0 for trials that did not terminate by ``T``.

    ssm-sim MOD: ``traj_out``/``record_trial`` let one row's per-step path be
    recorded (for the model cartoon); default -1 leaves the efpt behavior intact."""
    cdef:
        int n_trials = mu_array_data.shape[0]
        int max_steps
        int trial
        float* tptr
        float* traj_ptr = NULL

    max_steps = int(np.ceil(T / dt)) if T > 0.0 else 0
    rt_out = np.empty(n_trials, dtype=np.float64)
    choice_out = np.empty(n_trials, dtype=np.int32)
    x_final_out = np.empty(n_trials, dtype=np.float64)

    cdef double[::1] rt_view = rt_out
    cdef int[::1] choice_view = choice_out
    cdef double[::1] x_final_view = x_final_out

    if T <= 0.0:
        rt_out.fill(-1.0)
        choice_out.fill(0)
        x_final_out[:] = np.asarray(x0_data)
        return rt_out, choice_out, x_final_out

    # ssm-sim MOD: resolve the sink base pointer once, under the GIL. traj has
    # >= max_steps rows (setup: num_draws = int(max_t/delta_t)+1 >= ceil(max_t/delta_t)),
    # so recording traj_out[step] for step < max_steps never overflows.
    if record_trial >= 0 and traj_out is not None:
        traj_ptr = &traj_out[0, 0]

    for trial in prange(n_trials, nogil=True, num_threads=n_threads, schedule='dynamic'):
        # ssm-sim MOD: hand the sink only to the recording row; all others get NULL.
        if trial == record_trial:
            tptr = traj_ptr
        else:
            tptr = NULL
        _run_heterog_trial(
            mu_array_data, sigma_array_data, node_array_data,
            d_data[trial],
            ub_array_data, b1_array_data, lb_array_data, b2_array_data,
            trial, x0_data[trial], dt, max_steps, T,
            trial_seeds[trial],
            &rt_view[trial], &choice_view[trial], &x_final_view[trial],
            tptr,
        )

    return rt_out, choice_out, x_final_out


# ===========================================================================
# aDDM array construction (ported from efpt addm_helpers.py — pure numpy).
# ===========================================================================
def _build_addm_mu_array_data(eta, kappa, r1_data, r2_data, flag_data, d_data, max_d):
    """Padded (n, max_d) alternating drift array from aDDM covariates.

    mu1 = kappa*(r1 - eta*r2), mu2 = kappa*(eta*r1 - r2); the flag selects which
    drift the first fixation uses, then they alternate. Inactive stages (>= d) are
    zero. Ported verbatim from efpt _build_addm_mu_array_data. ``eta``/``kappa`` may
    be scalars or per-row arrays (broadcast against r1/r2)."""
    r1_data = np.asarray(r1_data, dtype=np.float64)
    r2_data = np.asarray(r2_data, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)
    mu1 = kappa * (r1_data - eta * r2_data)
    mu2 = kappa * (eta * r1_data - r2_data)
    mu1_eff = np.where(flag_data == 0, mu1, mu2).astype(np.float64)
    mu2_eff = np.where(flag_data == 0, mu2, mu1).astype(np.float64)
    if max_d == 0:
        return np.empty((len(mu1_eff), 0), dtype=np.float64)
    stages = np.arange(max_d)
    parity = stages[np.newaxis, :] % 2
    mu = np.where(parity == 0, mu1_eff[:, np.newaxis], mu2_eff[:, np.newaxis])
    mask = stages[np.newaxis, :] < np.asarray(d_data)[:, np.newaxis]
    return np.ascontiguousarray(mu * mask, dtype=np.float64)


def _generate_sacc_array_data(rng, n, T, gamma_shape, gamma_scale):
    """Self-sample padded (n, max_d) saccade onsets + per-row stage counts.

    Fixation durations ~ Gamma(shape, scale), cumulatively summed; column 0 is
    anchored at 0.0. d = number of onsets before ``T`` (d==1 means no saccade by
    T). Ported from efpt _generate_sacc_array_data."""
    avg_fixation = gamma_shape * gamma_scale
    budget = max(int(T / avg_fixation) + 50, 10)
    durations = rng.gamma(gamma_shape, gamma_scale, (n, budget))
    cum = np.cumsum(durations, axis=1)
    sacc = np.concatenate([np.zeros((n, 1), dtype=np.float64), cum], axis=1)
    d_data = (sacc < T).sum(axis=1).astype(np.int32)
    d_data = np.maximum(d_data, 1)
    max_d = int(d_data.max())
    return np.ascontiguousarray(sacc[:, :max_d], dtype=np.float64), d_data, max_d


cdef _col(x, int n):
    """Broadcast a scalar/array param to a contiguous float64 (n,) array."""
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.shape[0] == 1:
        arr = np.full(n, arr[0], dtype=np.float64)
    return np.ascontiguousarray(arr, dtype=np.float64)


def addm(eta, kappa, a, b, x0, t, deadline, s,
         sigma=None,
         r1=None, r2=None, flag=None, sacc_array=None, d=None,
         float gamma_shape=6.0,
         float gamma_scale=0.1,
         float delta_t=0.001,
         float max_t=20.0,
         int n_samples=1000,
         int n_trials=10,
         return_option='full',
         smooth_unif=False,
         random_state=None,
         int n_threads=1,
         **kwargs):
    """Simulate the aDDM with collapsing boundaries +-(a - b*t).

    Parameters (each a scalar or per-trial array of length ``n_trials``):
    ``eta, kappa, a, b, x0, t`` (see module docstring). ``s`` is ssm-simulators'
    noise vector (diffusion sigma); ``sigma`` overrides it when given (Mode 2).
    ``deadline`` censors RTs beyond it to the omission sentinel.

    Mode 2 (covariate-conditioned) is used when ``sacc_array`` is not None; then
    ``r1, r2, flag, sacc_array, d`` are the observed per-trial fixations and only
    the trajectory is sampled. Otherwise Mode 1 self-samples them.

    Returns the standard ssm-simulators dict; ``rts``/``choices`` have shape
    ``(n_samples, n_trials, 1)`` (rt == -999.0 marks an omission)."""
    setup = setup_simulation(n_samples, n_trials, max_t, delta_t, random_state)
    rts = setup['rts']
    choices = setup['choices']
    traj = setup['traj']

    cdef int N = n_samples * n_trials
    cdef double T = float(max_t)

    seed = random_state if random_state is not None else np.random.randint(0, 2**31)
    rng = np.random.default_rng(seed)

    # Per-trial -> per-row (row = sample*n_trials + trial) params.
    eta_t = _col(eta, n_trials); kappa_t = _col(kappa, n_trials)
    a_t = _col(a, n_trials); b_t = _col(b, n_trials)
    x0_t = _col(x0, n_trials); t_t = _col(t, n_trials)
    deadline_t = _col(deadline, n_trials)
    sigma_t = _col(sigma if sigma is not None else s, n_trials)

    eta_r = np.tile(eta_t, n_samples)
    kappa_r = np.tile(kappa_t, n_samples)
    a_r = np.tile(a_t, n_samples)
    b_r = np.tile(b_t, n_samples)
    x0_r = np.ascontiguousarray(np.tile(x0_t, n_samples), dtype=np.float64)
    t_r = np.tile(t_t, n_samples)
    deadline_r = np.tile(deadline_t, n_samples)
    sigma_r = np.tile(sigma_t, n_samples)

    mode2 = sacc_array is not None
    if mode2:
        # Observed fixations (n_trials, ...) -> tiled to N rows.
        max_d = int(np.asarray(sacc_array).shape[1])
        r1_r = np.tile(_col(r1, n_trials), n_samples)
        r2_r = np.tile(_col(r2, n_trials), n_samples)
        flag_r = np.tile(_col(flag, n_trials), n_samples).astype(np.int64)
        d_r = np.tile(np.asarray(d, dtype=np.int32).reshape(-1), n_samples).astype(np.int32)
        node_r = np.ascontiguousarray(
            np.tile(np.asarray(sacc_array, dtype=np.float64), (n_samples, 1)),
            dtype=np.float64,
        )
    else:
        # Mode 1: self-sample fixations + ratings per row.
        r1_r = rng.integers(1, 6, N).astype(np.float64)
        r2_r = rng.integers(1, 6, N).astype(np.float64)
        flag_r = rng.integers(0, 2, N).astype(np.int64)
        node_r, d_r, max_d = _generate_sacc_array_data(rng, N, T, gamma_shape, gamma_scale)

    mu_r = _build_addm_mu_array_data(eta_r, kappa_r, r1_r, r2_r, flag_r, d_r, max_d)

    # Per-stage diffusion + collapsing boundaries: ub = a - b*node, slope -b
    # (upper) / +b (lower); node-anchoring cancels to the wall-clock +-(a - b*t).
    sigma_arr = np.ascontiguousarray(
        np.broadcast_to(sigma_r[:, None], (N, max_d)), dtype=np.float64)
    ub = np.ascontiguousarray(a_r[:, None] - b_r[:, None] * node_r, dtype=np.float64)
    lb = np.ascontiguousarray(-ub, dtype=np.float64)
    b1 = np.ascontiguousarray(np.broadcast_to(-b_r[:, None], (N, max_d)), dtype=np.float64)
    b2 = np.ascontiguousarray(np.broadcast_to(b_r[:, None], (N, max_d)), dtype=np.float64)

    # Per-row seeds fixed BEFORE the parallel loop -> n_threads-independent.
    seeds = rng.integers(0, 2**64, size=N, dtype=np.uint64)

    # record_trial=0 records row 0's (sample 0, trial 0) per-step path into `traj`
    # for the model cartoon; faithful for both no_noise and noisy sims.
    rt, choice, _xf = _simulate_heterog_multistage(
        mu_r, sigma_arr, node_r, np.ascontiguousarray(d_r, dtype=np.int32),
        ub, b1, lb, b2, x0_r, float(delta_t), T,
        np.ascontiguousarray(seeds, dtype=np.uint64), n_threads,
        traj, 0,
    )

    rt = np.asarray(rt, dtype=np.float64)
    choice = np.asarray(choice, dtype=np.int32)
    rt_shifted = rt + t_r
    omit = (rt < 0.0) | (rt_shifted > deadline_r)
    rt_final = np.where(omit, OMISSION, rt_shifted)
    choice_final = np.where(omit, 0, choice)

    rts[:] = rt_final.reshape(n_samples, n_trials, 1).astype(np.float32)
    choices[:] = choice_final.reshape(n_samples, n_trials, 1).astype(np.int32)

    minimal_meta = build_minimal_metadata(
        simulator_name='addm',
        possible_choices=[-1, 1],
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name='addm_collapse',
    )
    if return_option == 'full':
        sim_config = {'delta_t': delta_t, 'max_t': max_t}
        # Cartoon support: upper boundary +(a - b*t) over the sim time grid, clamped
        # at 0 where the wedge has closed; plus a relative start z in [0,1] for the
        # start-point marker (aDDM's native start is the ABSOLUTE x0, so map
        # x0 -> 0.5 + 0.5*x0/boundary[0]; x0=0 lands mid-way at z=0.5). `traj` (row 0)
        # was filled by the sim above.
        t_s = setup['t_s']
        boundary = np.maximum(a_t[0] - b_t[0] * t_s, 0.0).astype(np.float32)
        b0 = float(boundary[0]) if boundary[0] != 0.0 else 1.0
        z_rel = np.clip(0.5 + 0.5 * (x0_t / b0), 0.0, 1.0).astype(np.float32)
        params = {
            'eta': eta, 'kappa': kappa, 'a': a, 'b': b,
            'x0': x0, 't': t, 'sigma': sigma_t, 'z': z_rel,
        }
        meta = build_full_metadata(
            minimal_metadata=minimal_meta, sim_config=sim_config,
            params=params, traj=traj, boundary=boundary,
        )
    elif return_option == 'minimal':
        meta = minimal_meta
    else:
        raise ValueError(
            f"Invalid return_option: {return_option}. Must be 'full' or 'minimal'."
        )
    return build_return_dict(rts, choices, meta)
