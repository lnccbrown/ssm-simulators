"""
Numba-based DDM Simulators with Parallel Execution

This module implements DDM simulators using Numba's JIT compilation
with automatic parallelization via prange.

Key optimizations:
1. JIT compilation for C-like performance
2. prange for automatic parallel execution
3. Fastmath for aggressive floating-point optimizations
4. Cache=True for faster subsequent imports
"""

import numpy as np
from numba import njit, prange, float32, uint64

# ============================================================================
# Fast Random Number Generation (Thread-Safe)
# Using xoroshiro128+ algorithm compatible with Numba
# ============================================================================


@njit(cache=True, fastmath=True)
def _rotl(x: uint64, k: int) -> uint64:
    """Rotate left helper for xoroshiro128+"""
    return (x << k) | (x >> (64 - k))


@njit(cache=True, fastmath=True)
def _init_rng(seed: uint64, thread_id: int):
    """Initialize RNG state with seed and thread_id"""
    # Mix seed and thread_id using splitmix64
    z = seed + np.uint64(thread_id) * np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * np.uint64(0x94D049BB133111EB)
    s0 = z ^ (z >> 31)

    z = s0 + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * np.uint64(0x94D049BB133111EB)
    s1 = z ^ (z >> 31)

    if s0 == 0 and s1 == 0:
        s0 = np.uint64(1)

    return s0, s1


@njit(cache=True, fastmath=True)
def _xoroshiro_next(s0: uint64, s1: uint64):
    """Generate next random uint64 using xoroshiro128+"""
    result = s0 + s1

    s1 ^= s0
    new_s0 = _rotl(s0, 24) ^ s1 ^ (s1 << 16)
    new_s1 = _rotl(s1, 37)

    return result, new_s0, new_s1


@njit(cache=True, fastmath=True)
def _random_uniform(s0: uint64, s1: uint64):
    """Generate uniform random number in [0, 1)"""
    x, new_s0, new_s1 = _xoroshiro_next(s0, s1)
    # Use top 53 bits for double precision
    u = np.float64(x >> 11) * (1.0 / 9007199254740992.0)
    return u, new_s0, new_s1


@njit(cache=True, fastmath=True)
def _random_gaussian(s0: uint64, s1: uint64):
    """Generate standard normal using Box-Muller"""
    u1, s0, s1 = _random_uniform(s0, s1)
    while u1 <= 1e-10:
        u1, s0, s1 = _random_uniform(s0, s1)
    u2, s0, s1 = _random_uniform(s0, s1)

    g = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    return np.float32(g), s0, s1


# ============================================================================
# Core DDM Kernel (Single Sample)
# ============================================================================


@njit(cache=True, fastmath=True)
def _ddm_simulate_single(
    v: float32,
    a: float32,
    z: float32,
    t: float32,
    deadline: float32,
    s: float32,
    delta_t: float32,
    max_t: float32,
    seed: uint64,
    sample_idx: int,
    trial_idx: int,
    smooth_unif: bool,
):
    """
    Simulate a single DDM sample.

    Returns (rt, choice) tuple.
    """
    # Initialize RNG for this sample
    s0, s1 = _init_rng(seed + np.uint64(trial_idx * 1000000 + sample_idx), sample_idx)

    # Precompute constants
    delta_t_sqrt = np.sqrt(delta_t)
    sqrt_st = delta_t_sqrt * s
    deadline_tmp = min(max_t, deadline - t)

    # Initialize particle
    y = z * a
    t_particle = np.float32(0.0)

    # Random walk
    while y <= a and y >= 0 and t_particle <= deadline_tmp:
        g, s0, s1 = _random_gaussian(s0, s1)
        y = y + v * delta_t + sqrt_st * g
        t_particle = t_particle + delta_t

    # Apply smoothing
    smooth_u = np.float32(0.0)
    if smooth_unif:
        u, s0, s1 = _random_uniform(s0, s1)
        if t_particle == 0.0:
            smooth_u = np.float32(u * 0.5 * delta_t)
        elif t_particle < deadline_tmp:
            smooth_u = np.float32((0.5 - u) * delta_t)

    # Compute RT
    rt = t_particle + t + smooth_u

    # Compute choice and enforce deadline
    if rt >= deadline or deadline <= 0:
        rt = np.float32(-999.0)
        choice = np.int32(-1)
    else:
        choice = np.int32(1) if y > 0 else np.int32(-1)

    return rt, choice


# ============================================================================
# Parallel DDM Simulator
# ============================================================================


@njit(parallel=True, cache=True, fastmath=True)
def _ddm_parallel_kernel(
    v: np.ndarray,
    a: np.ndarray,
    z: np.ndarray,
    t: np.ndarray,
    deadline: np.ndarray,
    s: np.ndarray,
    delta_t: float32,
    max_t: float32,
    n_samples: int,
    n_trials: int,
    seed: uint64,
    smooth_unif: bool,
    rts: np.ndarray,
    choices: np.ndarray,
):
    """
    Parallel DDM kernel using prange.

    Parallelizes across the flattened (sample, trial) space for
    optimal load balancing.
    """
    total_sims = n_samples * n_trials

    for idx in prange(total_sims):
        # Convert flat index to (sample, trial)
        n = idx // n_trials
        k = idx % n_trials

        # Get parameters for this trial
        v_k = v[k]
        a_k = a[k]
        z_k = z[k]
        t_k = t[k]
        deadline_k = deadline[k]
        s_k = s[k]

        # Simulate
        rt, choice = _ddm_simulate_single(
            v_k, a_k, z_k, t_k, deadline_k, s_k, delta_t, max_t, seed, n, k, smooth_unif
        )

        rts[n, k, 0] = rt
        choices[n, k, 0] = choice


def ddm_numba(
    v: np.ndarray,
    a: np.ndarray,
    z: np.ndarray,
    t: np.ndarray,
    deadline: np.ndarray | None = None,
    s: np.ndarray | None = None,
    delta_t: float = 0.001,
    max_t: float = 20.0,
    n_samples: int = 20000,
    n_trials: int | None = None,
    random_state: int | None = None,
    return_option: str = "full",
    smooth_unif: bool = False,
    **kwargs,
) -> dict:
    """
    Parallel DDM simulator using Numba JIT with automatic parallelization.

    This function provides a drop-in replacement for the standard DDM simulator
    with multi-threaded execution.

    Parameters
    ----------
    v : np.ndarray
        Drift rate for each trial
    a : np.ndarray
        Boundary separation for each trial
    z : np.ndarray
        Starting point (as proportion of a) for each trial
    t : np.ndarray
        Non-decision time for each trial
    deadline : np.ndarray, optional
        Maximum allowed RT for each trial (default: 999)
    s : np.ndarray, optional
        Noise standard deviation for each trial (default: 1.0)
    delta_t : float
        Time step size (default: 0.001)
    max_t : float
        Maximum simulation time (default: 20.0)
    n_samples : int
        Number of samples per trial (default: 20000)
    n_trials : int, optional
        Number of trials (inferred from parameter arrays if not given)
    random_state : int, optional
        Random seed for reproducibility
    return_option : str
        'full' or 'minimal' (default: 'full')
    smooth_unif : bool
        Apply uniform smoothing (default: False)

    Returns
    -------
    dict
        Dictionary with 'rts', 'choices', and 'metadata'
    """
    # Ensure arrays are float32 and contiguous
    v = np.ascontiguousarray(v, dtype=np.float32)
    a = np.ascontiguousarray(a, dtype=np.float32)
    z = np.ascontiguousarray(z, dtype=np.float32)
    t = np.ascontiguousarray(t, dtype=np.float32)

    # Infer n_trials if not provided
    if n_trials is None:
        n_trials = len(v)

    # Set defaults
    if deadline is None:
        deadline = np.full(n_trials, 999.0, dtype=np.float32)
    else:
        deadline = np.ascontiguousarray(deadline, dtype=np.float32)

    if s is None:
        s = np.ones(n_trials, dtype=np.float32)
    else:
        s = np.ascontiguousarray(s, dtype=np.float32)

    # Handle random state
    if random_state is None:
        seed = np.uint64(np.random.default_rng().integers(0, 2**63))
    else:
        seed = np.uint64(random_state)

    # Allocate output arrays
    rts = np.zeros((n_samples, n_trials, 1), dtype=np.float32)
    choices = np.zeros((n_samples, n_trials, 1), dtype=np.int32)

    # Run parallel simulation
    _ddm_parallel_kernel(
        v,
        a,
        z,
        t,
        deadline,
        s,
        np.float32(delta_t),
        np.float32(max_t),
        n_samples,
        n_trials,
        seed,
        smooth_unif,
        rts,
        choices,
    )

    # Build metadata
    metadata = {
        "simulator": "ddm_numba",
        "possible_choices": [-1, 1],
        "n_samples": n_samples,
        "n_trials": n_trials,
        "boundary_fun_type": "constant",
    }

    if return_option == "full":
        metadata.update(
            {
                "delta_t": delta_t,
                "max_t": max_t,
                "v": v,
                "a": a,
                "z": z,
                "t": t,
                "deadline": deadline,
                "s": s,
            }
        )

    return {"rts": rts, "choices": choices, "metadata": metadata}


# ============================================================================
# Single-threaded version for comparison
# ============================================================================


@njit(cache=True, fastmath=True)
def _ddm_sequential_kernel(
    v: np.ndarray,
    a: np.ndarray,
    z: np.ndarray,
    t: np.ndarray,
    deadline: np.ndarray,
    s: np.ndarray,
    delta_t: float32,
    max_t: float32,
    n_samples: int,
    n_trials: int,
    seed: uint64,
    smooth_unif: bool,
    rts: np.ndarray,
    choices: np.ndarray,
):
    """Sequential DDM kernel for comparison."""
    for k in range(n_trials):
        for n in range(n_samples):
            rt, choice = _ddm_simulate_single(
                v[k],
                a[k],
                z[k],
                t[k],
                deadline[k],
                s[k],
                delta_t,
                max_t,
                seed,
                n,
                k,
                smooth_unif,
            )
            rts[n, k, 0] = rt
            choices[n, k, 0] = choice


def ddm_numba_single(
    v: np.ndarray,
    a: np.ndarray,
    z: np.ndarray,
    t: np.ndarray,
    deadline: np.ndarray | None = None,
    s: np.ndarray | None = None,
    delta_t: float = 0.001,
    max_t: float = 20.0,
    n_samples: int = 20000,
    n_trials: int | None = None,
    random_state: int | None = None,
    return_option: str = "full",
    smooth_unif: bool = False,
    **kwargs,
) -> dict:
    """Single-threaded Numba DDM for comparison."""
    # Same preprocessing as parallel version
    v = np.ascontiguousarray(v, dtype=np.float32)
    a = np.ascontiguousarray(a, dtype=np.float32)
    z = np.ascontiguousarray(z, dtype=np.float32)
    t = np.ascontiguousarray(t, dtype=np.float32)

    if n_trials is None:
        n_trials = len(v)

    if deadline is None:
        deadline = np.full(n_trials, 999.0, dtype=np.float32)
    else:
        deadline = np.ascontiguousarray(deadline, dtype=np.float32)

    if s is None:
        s = np.ones(n_trials, dtype=np.float32)
    else:
        s = np.ascontiguousarray(s, dtype=np.float32)

    if random_state is None:
        seed = np.uint64(np.random.default_rng().integers(0, 2**63))
    else:
        seed = np.uint64(random_state)

    rts = np.zeros((n_samples, n_trials, 1), dtype=np.float32)
    choices = np.zeros((n_samples, n_trials, 1), dtype=np.int32)

    _ddm_sequential_kernel(
        v,
        a,
        z,
        t,
        deadline,
        s,
        np.float32(delta_t),
        np.float32(max_t),
        n_samples,
        n_trials,
        seed,
        smooth_unif,
        rts,
        choices,
    )

    metadata = {
        "simulator": "ddm_numba_single",
        "possible_choices": [-1, 1],
        "n_samples": n_samples,
        "n_trials": n_trials,
        "boundary_fun_type": "constant",
    }

    if return_option == "full":
        metadata.update(
            {
                "delta_t": delta_t,
                "max_t": max_t,
                "v": v,
                "a": a,
                "z": z,
                "t": t,
                "deadline": deadline,
                "s": s,
            }
        )

    return {"rts": rts, "choices": choices, "metadata": metadata}


# ============================================================================
# DDM with Flexible Boundary
# ============================================================================


@njit(parallel=True, cache=True, fastmath=True)
def _ddm_flexbound_parallel_kernel(
    v: np.ndarray,
    z: np.ndarray,
    t: np.ndarray,
    deadline: np.ndarray,
    s: np.ndarray,
    boundary: np.ndarray,  # Precomputed boundary for all time steps
    delta_t: float32,
    max_t: float32,
    n_samples: int,
    n_trials: int,
    num_steps: int,
    seed: uint64,
    smooth_unif: bool,
    rts: np.ndarray,
    choices: np.ndarray,
):
    """Parallel DDM kernel with flexible boundary."""
    total_sims = n_samples * n_trials
    delta_t_sqrt = np.sqrt(delta_t)

    for idx in prange(total_sims):
        n = idx // n_trials
        k = idx % n_trials

        # Initialize RNG
        s0, s1 = _init_rng(seed + np.uint64(k * 1000000 + n), n)

        # Get parameters
        sqrt_st = delta_t_sqrt * s[k]
        deadline_tmp = min(max_t, deadline[k] - t[k])

        # Initialize particle with flexible boundary
        y = (-1) * boundary[k, 0] + (z[k] * 2 * boundary[k, 0])
        t_particle = np.float32(0.0)
        ix = 0

        # Random walk
        while (
            (y >= (-1) * boundary[k, ix])
            and (y <= boundary[k, ix])
            and (t_particle <= deadline_tmp)
        ):
            g, s0, s1 = _random_gaussian(s0, s1)
            y = y + v[k] * delta_t + sqrt_st * g
            t_particle = t_particle + delta_t
            ix = ix + 1
            if ix >= num_steps:
                ix = num_steps - 1

        # Smoothing
        smooth_u = np.float32(0.0)
        if smooth_unif:
            u, s0, s1 = _random_uniform(s0, s1)
            if t_particle == 0.0:
                smooth_u = np.float32(u * 0.5 * delta_t)
            elif t_particle < deadline_tmp:
                smooth_u = np.float32((0.5 - u) * delta_t)

        rt = t_particle + t[k] + smooth_u

        if rt >= deadline[k] or deadline[k] <= 0:
            rts[n, k, 0] = np.float32(-999.0)
            choices[n, k, 0] = np.int32(-1)
        else:
            rts[n, k, 0] = rt
            choices[n, k, 0] = np.int32(1) if y > 0 else np.int32(-1)


def ddm_flexbound_numba(
    v: np.ndarray,
    a: np.ndarray,
    z: np.ndarray,
    t: np.ndarray,
    deadline: np.ndarray | None = None,
    s: np.ndarray | None = None,
    delta_t: float = 0.001,
    max_t: float = 20.0,
    n_samples: int = 20000,
    n_trials: int | None = None,
    boundary_fun=None,
    boundary_params: dict | None = None,
    random_state: int | None = None,
    return_option: str = "full",
    smooth_unif: bool = False,
    **kwargs,
) -> dict:
    """
    Parallel DDM simulator with flexible boundaries using Numba.

    The boundary is precomputed for all trials before the parallel section.
    """
    # Ensure arrays are float32
    v = np.ascontiguousarray(v, dtype=np.float32)
    a = np.ascontiguousarray(a, dtype=np.float32)
    z = np.ascontiguousarray(z, dtype=np.float32)
    t = np.ascontiguousarray(t, dtype=np.float32)

    if n_trials is None:
        n_trials = len(v)

    if deadline is None:
        deadline = np.full(n_trials, 999.0, dtype=np.float32)
    else:
        deadline = np.ascontiguousarray(deadline, dtype=np.float32)

    if s is None:
        s = np.ones(n_trials, dtype=np.float32)
    else:
        s = np.ascontiguousarray(s, dtype=np.float32)

    if boundary_params is None:
        boundary_params = {}

    if random_state is None:
        seed = np.uint64(np.random.default_rng().integers(0, 2**63))
    else:
        seed = np.uint64(random_state)

    # Precompute boundary for all trials
    num_steps = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t, dtype=np.float32)
    boundary = np.zeros((n_trials, num_steps), dtype=np.float32)

    for k in range(n_trials):
        boundary_params_tmp = {
            key: boundary_params[key][k] for key in boundary_params.keys()
        }
        boundary[k, : len(t_s)] = boundary_fun(t=t_s, **boundary_params_tmp).astype(
            np.float32
        )[:num_steps]

    # Allocate output
    rts = np.zeros((n_samples, n_trials, 1), dtype=np.float32)
    choices = np.zeros((n_samples, n_trials, 1), dtype=np.int32)

    # Run simulation
    _ddm_flexbound_parallel_kernel(
        v,
        z,
        t,
        deadline,
        s,
        boundary,
        np.float32(delta_t),
        np.float32(max_t),
        n_samples,
        n_trials,
        num_steps,
        seed,
        smooth_unif,
        rts,
        choices,
    )

    metadata = {
        "simulator": "ddm_flexbound_numba",
        "possible_choices": [-1, 1],
        "n_samples": n_samples,
        "n_trials": n_trials,
        "boundary_fun_type": boundary_fun.__name__ if boundary_fun else "unknown",
    }

    if return_option == "full":
        metadata.update(
            {
                "delta_t": delta_t,
                "max_t": max_t,
                "v": v,
                "a": a,
                "z": z,
                "t": t,
                "deadline": deadline,
                "s": s,
                "boundary": boundary,
            }
        )

    return {"rts": rts, "choices": choices, "metadata": metadata}


# ============================================================================
# Full DDM with Inter-trial Variability
# ============================================================================


@njit(cache=True, fastmath=True)
def _full_ddm_simulate_single(
    v: float32,
    a: float32,
    z: float32,
    t: float32,
    sz: float32,
    sv: float32,
    st: float32,
    deadline: float32,
    s: float32,
    boundary: np.ndarray,
    num_steps: int,
    delta_t: float32,
    max_t: float32,
    seed: uint64,
    sample_idx: int,
    trial_idx: int,
    smooth_unif: bool,
):
    """Simulate single full DDM sample with inter-trial variability."""
    # Initialize RNG
    s0, s1 = _init_rng(seed + np.uint64(trial_idx * 1000000 + sample_idx), sample_idx)

    delta_t_sqrt = np.sqrt(delta_t)
    sqrt_st = delta_t_sqrt * s
    deadline_tmp = min(max_t, deadline - t)

    # Initialize with variability
    y = (-1) * boundary[0] + (z * 2 * boundary[0])
    u, s0, s1 = _random_uniform(s0, s1)
    y = y + 2 * (u - 0.5) * sz

    g, s0, s1 = _random_gaussian(s0, s1)
    drift_increment = (v + sv * g) * delta_t

    u, s0, s1 = _random_uniform(s0, s1)
    t_tmp = t + (2 * (u - 0.5) * st)

    t_particle = np.float32(0.0)
    ix = 0

    # Random walk
    while (
        (y >= (-1) * boundary[ix])
        and (y <= boundary[ix])
        and (t_particle <= deadline_tmp)
    ):
        g, s0, s1 = _random_gaussian(s0, s1)
        y = y + drift_increment + sqrt_st * g
        t_particle = t_particle + delta_t
        ix = ix + 1
        if ix >= num_steps:
            ix = num_steps - 1

    # Smoothing
    smooth_u = np.float32(0.0)
    if smooth_unif:
        u, s0, s1 = _random_uniform(s0, s1)
        if t_particle == 0.0:
            smooth_u = np.float32(u * 0.5 * delta_t)
        elif t_particle < deadline_tmp:
            smooth_u = np.float32((0.5 - u) * delta_t)

    rt = t_particle + t_tmp + smooth_u

    if rt >= deadline or deadline <= 0:
        return np.float32(-999.0), np.int32(-1)
    else:
        choice = np.int32(1) if y > 0 else np.int32(-1)
        return rt, choice


@njit(parallel=True, cache=True, fastmath=True)
def _full_ddm_parallel_kernel(
    v: np.ndarray,
    a: np.ndarray,
    z: np.ndarray,
    t: np.ndarray,
    sz: np.ndarray,
    sv: np.ndarray,
    st: np.ndarray,
    deadline: np.ndarray,
    s: np.ndarray,
    boundary: np.ndarray,
    delta_t: float32,
    max_t: float32,
    n_samples: int,
    n_trials: int,
    num_steps: int,
    seed: uint64,
    smooth_unif: bool,
    rts: np.ndarray,
    choices: np.ndarray,
):
    """Parallel full DDM kernel."""
    total_sims = n_samples * n_trials

    for idx in prange(total_sims):
        n = idx // n_trials
        k = idx % n_trials

        rt, choice = _full_ddm_simulate_single(
            v[k],
            a[k],
            z[k],
            t[k],
            sz[k],
            sv[k],
            st[k],
            deadline[k],
            s[k],
            boundary[k],
            num_steps,
            delta_t,
            max_t,
            seed,
            n,
            k,
            smooth_unif,
        )

        rts[n, k, 0] = rt
        choices[n, k, 0] = choice


def full_ddm_numba(
    v: np.ndarray,
    a: np.ndarray,
    z: np.ndarray,
    t: np.ndarray,
    sz: np.ndarray,
    sv: np.ndarray,
    st: np.ndarray,
    deadline: np.ndarray | None = None,
    s: np.ndarray | None = None,
    delta_t: float = 0.001,
    max_t: float = 20.0,
    n_samples: int = 20000,
    n_trials: int | None = None,
    boundary_fun=None,
    boundary_params: dict | None = None,
    random_state: int | None = None,
    return_option: str = "full",
    smooth_unif: bool = False,
    **kwargs,
) -> dict:
    """
    Parallel full DDM simulator with inter-trial variability using Numba.
    """
    # Ensure arrays are float32
    v = np.ascontiguousarray(v, dtype=np.float32)
    a = np.ascontiguousarray(a, dtype=np.float32)
    z = np.ascontiguousarray(z, dtype=np.float32)
    t = np.ascontiguousarray(t, dtype=np.float32)
    sz = np.ascontiguousarray(sz, dtype=np.float32)
    sv = np.ascontiguousarray(sv, dtype=np.float32)
    st = np.ascontiguousarray(st, dtype=np.float32)

    if n_trials is None:
        n_trials = len(v)

    if deadline is None:
        deadline = np.full(n_trials, 999.0, dtype=np.float32)
    else:
        deadline = np.ascontiguousarray(deadline, dtype=np.float32)

    if s is None:
        s = np.ones(n_trials, dtype=np.float32)
    else:
        s = np.ascontiguousarray(s, dtype=np.float32)

    if boundary_params is None:
        boundary_params = {}

    if random_state is None:
        seed = np.uint64(np.random.default_rng().integers(0, 2**63))
    else:
        seed = np.uint64(random_state)

    # Precompute boundary
    num_steps = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t, dtype=np.float32)
    boundary = np.zeros((n_trials, num_steps), dtype=np.float32)

    for k in range(n_trials):
        boundary_params_tmp = {
            key: boundary_params[key][k] for key in boundary_params.keys()
        }
        boundary[k, : len(t_s)] = boundary_fun(t=t_s, **boundary_params_tmp).astype(
            np.float32
        )[:num_steps]

    rts = np.zeros((n_samples, n_trials, 1), dtype=np.float32)
    choices = np.zeros((n_samples, n_trials, 1), dtype=np.int32)

    _full_ddm_parallel_kernel(
        v,
        a,
        z,
        t,
        sz,
        sv,
        st,
        deadline,
        s,
        boundary,
        np.float32(delta_t),
        np.float32(max_t),
        n_samples,
        n_trials,
        num_steps,
        seed,
        smooth_unif,
        rts,
        choices,
    )

    metadata = {
        "simulator": "full_ddm_numba",
        "possible_choices": [-1, 1],
        "n_samples": n_samples,
        "n_trials": n_trials,
        "boundary_fun_type": boundary_fun.__name__ if boundary_fun else "unknown",
    }

    if return_option == "full":
        metadata.update(
            {
                "delta_t": delta_t,
                "max_t": max_t,
                "v": v,
                "a": a,
                "z": z,
                "t": t,
                "sz": sz,
                "sv": sv,
                "st": st,
                "deadline": deadline,
                "s": s,
                "boundary": boundary,
            }
        )

    return {"rts": rts, "choices": choices, "metadata": metadata}
