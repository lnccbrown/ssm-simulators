# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""Forward simulator for independent, multi-stage race models.

This module implements the Monte Carlo model in ``race_model.pdf``: every
accumulator has an independent Brownian motion, a native stage partition,
piecewise-constant drift/diffusion, and a piecewise-linear *upper* boundary.
The race ends at the first upper-boundary crossing.  It intentionally has no
lower decision boundary and does not clip paths at zero.

The random-number implementation follows ``addm_models.pyx``.  One xoshiro
state is seeded for every (sample, trial) pair before the OpenMP loop, making
results reproducible for a fixed ``random_state`` independently of scheduling
or ``n_threads``.
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, cos, sin, M_PI
from libc.stdlib cimport free, malloc
from libc.stdint cimport uint64_t
from cython.parallel cimport prange

from cssm._utils import (
    setup_simulation,
    build_minimal_metadata,
    build_full_metadata,
    build_return_dict,
)

cdef double OMISSION = -999.0
DEF MAX_ACCUMULATORS = 32
cdef int UNIFORM_BITS = 11
cdef double UINT64_TO_DOUBLE = 1.0 / 9007199254740992.0
cdef double MIN_UNIFORM = 1e-300
cdef double TWO_PI = 2.0 * M_PI


cdef struct SimulationConfig:
    double dt
    double horizon
    int max_steps


cdef struct TrialResult:
    double rt
    int choice


# Inline xoshiro256++ / Box-Muller RNG, matching the Efficient-FPT-derived
# aDDM engine.  Keep this local for now; a future shared RNG module can remove
# the duplication once the race simulator interface has settled.
cdef struct Xoshiro256State:
    uint64_t s0
    uint64_t s1
    uint64_t s2
    uint64_t s3


cdef inline uint64_t _rotl(uint64_t x, int k) noexcept nogil:
    return (x << k) | (x >> (64 - k))


cdef inline uint64_t _next(Xoshiro256State *state) noexcept nogil:
    cdef uint64_t result = _rotl(state.s0 + state.s3, 23) + state.s0
    cdef uint64_t t = state.s1 << 17
    state.s2 ^= state.s0
    state.s3 ^= state.s1
    state.s1 ^= state.s2
    state.s0 ^= state.s3
    state.s2 ^= t
    state.s3 = _rotl(state.s3, 45)
    return result


cdef inline uint64_t _splitmix64(uint64_t *state) noexcept nogil:
    state[0] += <uint64_t>0x9e3779b97f4a7c15
    cdef uint64_t z = state[0]
    z = (z ^ (z >> 30)) * <uint64_t>0xbf58476d1ce4e5b9
    z = (z ^ (z >> 27)) * <uint64_t>0x94d049bb133111eb
    return z ^ (z >> 31)


cdef inline void _seed(Xoshiro256State *state, uint64_t seed) noexcept nogil:
    cdef uint64_t sm_state = seed
    state.s0 = _splitmix64(&sm_state)
    state.s1 = _splitmix64(&sm_state)
    state.s2 = _splitmix64(&sm_state)
    state.s3 = _splitmix64(&sm_state)


cdef struct BoxMullerState:
    # Box-Muller produces two normals from a pair of uniforms. ``spare``
    # stores the second value, and ``has_spare`` marks it for the next call.
    double spare
    int has_spare


cdef inline double _normal(Xoshiro256State *rng, BoxMullerState *bm) noexcept nogil:
    """Return one standard normal and cache its Box-Muller companion.

    A call first consumes ``bm.spare`` when available. Otherwise it generates
    a pair of normals, returns the cosine component, and caches the sine
    component for the following call.
    """
    cdef double u1, u2, magnitude
    # Consume the cached companion before drawing new uniforms.
    if bm.has_spare:
        bm.has_spare = 0
        return bm.spare
    u1 = <double>(_next(rng) >> UNIFORM_BITS) * UINT64_TO_DOUBLE
    u2 = <double>(_next(rng) >> UNIFORM_BITS) * UINT64_TO_DOUBLE
    if u1 < MIN_UNIFORM:
        u1 = MIN_UNIFORM
    magnitude = sqrt(-2.0 * log(u1))
    bm.spare = magnitude * sin(TWO_PI * u2)
    bm.has_spare = 1
    return magnitude * cos(TWO_PI * u2)


cdef inline double _upper_boundary_at(
    double[:, :, ::1] upper_intercept,
    double[:, :, ::1] upper_slope,
    double[:, :, ::1] nodes,
    int row,
    int accumulator,
    int stage,
    double time,
) noexcept nogil:
    """Return an accumulator's upper boundary at ``time`` in ``stage``."""
    return (
        upper_intercept[row, accumulator, stage]
        + upper_slope[row, accumulator, stage]
        * (time - nodes[row, accumulator, stage])
    )


cdef inline bint _has_reached_next_stage(
    double[:, :, ::1] nodes,
    int[:, ::1] d,
    int row,
    int accumulator,
    int current_stage,
    double time,
) noexcept nogil:
    """Return whether an accumulator should advance to its next stage."""
    return (
        current_stage + 1 < d[row, accumulator]
        and time >= nodes[row, accumulator, current_stage + 1]
    )


cdef void _run_race_trial(
    double[:, :, ::1] mu,
    double[:, :, ::1] sigma,
    double[:, :, ::1] nodes,
    int[:, ::1] d,
    double[:, :, ::1] upper_intercept,
    double[:, :, ::1] upper_slope,
    int row,
    double[:, ::1] x0,
    SimulationConfig config,
    uint64_t seed,
    TrialResult *result,
    double *x_final_out,
) noexcept nogil:
    """Simulate one race. A same-grid-step tie uses the lowest index.

    Continuous-time ties have probability zero. The deterministic grid-tie rule
    therefore only resolves a discretisation artefact and keeps seeded runs
    deterministic.
    """
    cdef:
        Xoshiro256State rng
        BoxMullerState bm
        double particle[MAX_ACCUMULATORS]
        double t_particle, dt_current, sqrt_dt, boundary, next_node
        double drift_increment, diffusion_increment
        int stage[MAX_ACCUMULATORS]
        int i, step, winner, stage_changed, n_accumulators

    _seed(&rng, seed)
    bm.has_spare = 0
    t_particle = 0.0
    result.rt = -1.0
    result.choice = 0
    n_accumulators = mu.shape[1]
    for i in range(n_accumulators):
        particle[i] = x0[row, i]
        stage[i] = 0

    for step in range(config.max_steps):
        # A node reached by the preceding propagation begins its new stage
        # before this iteration can draw additional noise.
        stage_changed = 0
        for i in range(n_accumulators):
            while _has_reached_next_stage(
                nodes, d, row, i, stage[i], t_particle
            ):
                stage[i] += 1
                stage_changed = 1

        # A discontinuity in the boundary can itself end the race. Use the
        # previous Euler-step midpoint convention (or zero at the start).
        if stage_changed:
            winner = -1
            for i in range(n_accumulators):
                boundary = _upper_boundary_at(
                    upper_intercept, upper_slope, nodes,
                    row, i, stage[i], t_particle,
                )
                if particle[i] >= boundary and winner < 0:
                    winner = i
            if winner >= 0:
                result.rt = t_particle - 0.5 * dt_current if step > 0 else 0.0
                result.choice = winner
                break

        dt_current = config.horizon - t_particle
        if dt_current <= 0.0:
            break
        dt_current = min(dt_current, config.dt)

        # Do not propagate through a stage node with the preceding stage's
        # dynamics. The earliest pending node controls this Euler step.
        for i in range(n_accumulators):
            if stage[i] + 1 < d[row, i]:
                next_node = nodes[row, i, stage[i] + 1]
                if next_node < t_particle + dt_current:
                    dt_current = next_node - t_particle
        sqrt_dt = sqrt(dt_current)

        for i in range(n_accumulators):
            drift_increment = mu[row, i, stage[i]] * dt_current
            diffusion_increment = (
                sigma[row, i, stage[i]] * sqrt_dt * _normal(&rng, &bm)
            )
            particle[i] += drift_increment + diffusion_increment
        t_particle += dt_current

        winner = -1
        for i in range(n_accumulators):
            boundary = _upper_boundary_at(
                upper_intercept, upper_slope, nodes,
                row, i, stage[i], t_particle,
            )
            if particle[i] >= boundary and winner < 0:
                winner = i

        if winner >= 0:
            # The aDDM Efficient-FPT-compatible simulator reports the midpoint
            # of the Euler step; use the same first-order convention here.
            result.rt = t_particle - 0.5 * dt_current
            result.choice = winner
            break

        # Stage updates happen at the beginning of the next iteration, where
        # the new boundary is checked before another noise draw.

    for i in range(n_accumulators):
        x_final_out[i] = particle[i]


cdef void _validate_race_inputs(
    double[:, :, ::1] mu,
    double[:, :, ::1] sigma,
    double[:, :, ::1] nodes,
    int[:, ::1] d,
    double[:, :, ::1] upper_intercept,
    double[:, :, ::1] upper_slope,
    double[:, ::1] x0,
    double dt,
    uint64_t[::1] seeds,
) except *:
    """Validate the low-level multi-stage race simulator input contract."""
    cdef:
        int n_rows = mu.shape[0]
        int n_accumulators = mu.shape[1]

    if n_accumulators > MAX_ACCUMULATORS:
        raise ValueError(
            f"race_multistage supports at most {MAX_ACCUMULATORS} accumulators; "
            f"got {n_accumulators}"
        )
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if (
        sigma.shape[0] != n_rows or sigma.shape[1] != n_accumulators or sigma.shape[2] != mu.shape[2]
        or nodes.shape[0] != n_rows or nodes.shape[1] != n_accumulators or nodes.shape[2] != mu.shape[2]
        or upper_intercept.shape[0] != n_rows or upper_intercept.shape[1] != n_accumulators or upper_intercept.shape[2] != mu.shape[2]
        or upper_slope.shape[0] != n_rows or upper_slope.shape[1] != n_accumulators or upper_slope.shape[2] != mu.shape[2]
    ):
        raise ValueError("stage arrays must have the same (rows, accumulators, stages) shape")
    if d.shape[0] != n_rows or d.shape[1] != n_accumulators:
        raise ValueError("d must have shape (rows, accumulators)")
    if x0.shape[0] != n_rows or x0.shape[1] != n_accumulators:
        raise ValueError("x0 must have shape (rows, accumulators)")
    if seeds.shape[0] != n_rows:
        raise ValueError("seeds must contain one seed per row")
    if np.any(np.asarray(d) < 1) or np.any(np.asarray(d) > mu.shape[2]):
        raise ValueError("each d entry must lie between 1 and the padded stage count")


def _simulate_race_multistage(
    double[:, :, ::1] mu,
    double[:, :, ::1] sigma,
    double[:, :, ::1] nodes,
    int[:, ::1] d,
    double[:, :, ::1] upper_intercept,
    double[:, :, ::1] upper_slope,
    double[:, ::1] x0,
    double dt,
    double horizon,
    uint64_t[::1] seeds,
    int n_threads=1,
):
    """Low-level batch kernel used by :func:`race_multistage` and tests."""
    cdef:
        int n_rows = mu.shape[0]
        int n_accumulators = mu.shape[1]
        int max_steps = int(np.ceil(horizon / dt)) if horizon > 0.0 else 0
        int row
        SimulationConfig config
        TrialResult *trial_results

    _validate_race_inputs(
        mu, sigma, nodes, d, upper_intercept, upper_slope, x0, dt, seeds
    )

    rt = np.empty(n_rows, dtype=np.float64)
    choice = np.empty(n_rows, dtype=np.int32)
    x_final = np.empty((n_rows, n_accumulators), dtype=np.float64)
    cdef double[::1] rt_view = rt
    cdef int[::1] choice_view = choice
    cdef double[:, ::1] final_view = x_final

    # Match the Efficient-FPT-derived aDDM batch engine: there is no path to
    # evolve at a zero/negative horizon, so avoid starting the parallel loop.
    if horizon <= 0.0:
        rt.fill(-1.0)
        choice.fill(0)
        x_final[:] = np.asarray(x0)
        return rt, choice, x_final

    config.dt = dt
    config.horizon = horizon
    config.max_steps = max_steps
    trial_results = <TrialResult *>malloc(n_rows * sizeof(TrialResult))
    if trial_results == NULL:
        raise MemoryError("could not allocate race trial results")
    try:
        for row in prange(n_rows, nogil=True, num_threads=n_threads, schedule='dynamic'):
            _run_race_trial(
                mu, sigma, nodes, d, upper_intercept, upper_slope, row, x0,
                config, seeds[row], &trial_results[row], &final_view[row, 0],
            )
        for row in range(n_rows):
            rt_view[row] = trial_results[row].rt
            choice_view[row] = trial_results[row].choice
    finally:
        free(trial_results)
    return rt, choice, x_final


def race_multistage(
    mu_array,
    sigma_array,
    node_array,
    d_array,
    upper_intercept_array,
    upper_slope_array,
    x0_array,
    nondecision_time=None,
    deadline=None,
    float delta_t=0.001,
    float max_t=20.0,
    int n_samples=1000,
    int n_trials=0,
    return_option='full',
    random_state=None,
    int n_threads=1,
    **kwargs,
):
    """Simulate independent multi-stage race-model trajectories.

    Array inputs describe one row per *experimental trial*, with shape
    ``(n_trials, K, max_stages)`` except ``d_array`` and ``x0_array``, whose
    shapes are ``(n_trials, K)``. ``node_array`` contains stage start times;
    stage parameters are used from a start time until the next node. The
    boundary in stage ``k`` is ``upper_intercept + upper_slope * elapsed``.

    ``n_samples`` repeats each experimental trial. Outputs follow the SSMS
    contract: arrays are ``(n_samples, n_trials, 1)``, choices are zero based,
    and an unobserved response has RT ``-999.0``.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    mu = np.ascontiguousarray(mu_array, dtype=np.float64)
    sigma = np.ascontiguousarray(sigma_array, dtype=np.float64)
    nodes = np.ascontiguousarray(node_array, dtype=np.float64)
    d = np.ascontiguousarray(d_array, dtype=np.int32)
    intercept = np.ascontiguousarray(upper_intercept_array, dtype=np.float64)
    slope = np.ascontiguousarray(upper_slope_array, dtype=np.float64)
    x0 = np.ascontiguousarray(x0_array, dtype=np.float64)
    if mu.ndim != 3:
        raise ValueError("mu_array must have shape (n_trials, K, max_stages)")
    if n_trials == 0:
        n_trials = mu.shape[0]
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    if mu.shape[0] != n_trials:
        raise ValueError("n_trials must equal the first dimension of mu_array")

    setup = setup_simulation(n_samples, n_trials, max_t, delta_t, random_state)
    N = n_samples * n_trials
    seed = random_state if random_state is not None else np.random.randint(0, 2**31)
    rng = np.random.default_rng(seed)
    seeds = np.ascontiguousarray(rng.integers(0, 2**64, size=N, dtype=np.uint64))

    # sample-major tiling matches SSMS' (sample, trial) output layout.
    mu_rows = np.ascontiguousarray(np.tile(mu, (n_samples, 1, 1)))
    sigma_rows = np.ascontiguousarray(np.tile(sigma, (n_samples, 1, 1)))
    nodes_rows = np.ascontiguousarray(np.tile(nodes, (n_samples, 1, 1)))
    d_rows = np.ascontiguousarray(np.tile(d, (n_samples, 1)))
    intercept_rows = np.ascontiguousarray(np.tile(intercept, (n_samples, 1, 1)))
    slope_rows = np.ascontiguousarray(np.tile(slope, (n_samples, 1, 1)))
    x0_rows = np.ascontiguousarray(np.tile(x0, (n_samples, 1)))

    rt, choice, x_final = _simulate_race_multistage(
        mu_rows, sigma_rows, nodes_rows, d_rows, intercept_rows, slope_rows,
        x0_rows, delta_t, max_t, seeds, n_threads,
    )

    ndt = np.zeros(n_trials, dtype=np.float64) if nondecision_time is None else np.asarray(nondecision_time, dtype=np.float64).reshape(-1)
    ddl = np.full(n_trials, max_t, dtype=np.float64) if deadline is None else np.asarray(deadline, dtype=np.float64).reshape(-1)
    if ndt.size == 1:
        ndt = np.full(n_trials, ndt[0])
    if ddl.size == 1:
        ddl = np.full(n_trials, ddl[0])
    if ndt.size != n_trials or ddl.size != n_trials:
        raise ValueError("nondecision_time and deadline must be scalars or length n_trials")
    ndt_rows = np.tile(ndt, n_samples)
    ddl_rows = np.tile(ddl, n_samples)
    shifted_rt = rt + ndt_rows
    omitted = (rt < 0.0) | (shifted_rt > ddl_rows)
    final_rt = np.where(omitted, OMISSION, shifted_rt)
    final_choice = np.where(omitted, 0, choice)

    rts = setup['rts']
    choices = setup['choices']
    rts[:] = final_rt.reshape(n_samples, n_trials, 1).astype(np.float32)
    choices[:] = final_choice.reshape(n_samples, n_trials, 1).astype(np.int32)

    possible_choices = list(range(mu.shape[1]))
    minimal_meta = build_minimal_metadata(
        simulator_name='race_multistage',
        possible_choices=possible_choices,
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name='piecewise_linear_upper',
    )
    if return_option == 'minimal':
        metadata = minimal_meta
    elif return_option == 'full':
        metadata = build_full_metadata(
            minimal_metadata=minimal_meta,
            params={
                'mu_array': mu, 'sigma_array': sigma, 'node_array': nodes,
                'd_array': d, 'upper_intercept_array': intercept,
                'upper_slope_array': slope, 'x0_array': x0,
            },
            sim_config={'delta_t': delta_t, 'max_t': max_t, 'n_threads': n_threads},
            traj=setup['traj'],
            boundary=np.array([], dtype=np.float32),
        )
        metadata['x_final'] = x_final.reshape(n_samples, n_trials, mu.shape[1])
    else:
        raise ValueError("return_option must be either 'full' or 'minimal'")
    return build_return_dict(rts, choices, metadata)
