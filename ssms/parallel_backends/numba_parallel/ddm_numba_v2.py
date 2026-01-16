"""
Numba DDM Simulator v2 - Using Numba's built-in RNG

This version uses Numba's np.random functions which are reasonably fast
and don't require pre-allocation of noise arrays.
"""

import numpy as np
from numba import njit, prange, float32, int32
from typing import Dict, Any


@njit(cache=True, fastmath=True)
def _ddm_simulate_single_thread_v2(
    v: np.ndarray,
    a: np.ndarray,
    z: np.ndarray,
    t_ndt: np.ndarray,
    deadline: np.ndarray,
    s: np.ndarray,
    delta_t: float,
    max_t: float,
    n_samples: int,
    n_trials: int,
    rts: np.ndarray,
    choices: np.ndarray,
):
    """Single-threaded DDM using Numba's built-in RNG."""
    delta_t_sqrt = np.sqrt(delta_t)

    for k in range(n_trials):
        sqrt_st = delta_t_sqrt * s[k]
        deadline_tmp = min(max_t, deadline[k] - t_ndt[k])
        y_start = z[k] * a[k]
        v_k = v[k]
        a_k = a[k]
        t_k = t_ndt[k]

        for n in range(n_samples):
            y = y_start
            t_particle = float32(0.0)

            # Random walk
            while y > 0 and y < a_k and t_particle <= deadline_tmp:
                y = y + v_k * delta_t + sqrt_st * np.random.randn()
                t_particle = t_particle + delta_t

            # Store results
            rt = t_particle + t_k
            if rt >= deadline[k] or deadline[k] <= 0:
                rts[n, k] = float32(-999.0)
                choices[n, k] = int32(0)
            else:
                rts[n, k] = rt
                choices[n, k] = int32(1) if y >= a_k else int32(-1)


@njit(parallel=True, cache=True, fastmath=True)
def _ddm_simulate_parallel_v2(
    v: np.ndarray,
    a: np.ndarray,
    z: np.ndarray,
    t_ndt: np.ndarray,
    deadline: np.ndarray,
    s: np.ndarray,
    delta_t: float,
    max_t: float,
    n_samples: int,
    n_trials: int,
    rts: np.ndarray,
    choices: np.ndarray,
):
    """Parallel DDM using Numba's prange."""
    delta_t_sqrt = np.sqrt(delta_t)
    total_sims = n_samples * n_trials

    for idx in prange(total_sims):
        n = idx // n_trials
        k = idx % n_trials

        sqrt_st = delta_t_sqrt * s[k]
        deadline_tmp = min(max_t, deadline[k] - t_ndt[k])

        y = z[k] * a[k]
        t_particle = float32(0.0)
        v_k = v[k]
        a_k = a[k]

        # Random walk
        while y > 0 and y < a_k and t_particle <= deadline_tmp:
            y = y + v_k * delta_t + sqrt_st * np.random.randn()
            t_particle = t_particle + delta_t

        # Store results
        rt = t_particle + t_ndt[k]
        if rt >= deadline[k] or deadline[k] <= 0:
            rts[n, k] = float32(-999.0)
            choices[n, k] = int32(0)
        else:
            rts[n, k] = rt
            choices[n, k] = int32(1) if y >= a_k else int32(-1)


def ddm_numba_v2(
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
    parallel: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    DDM simulator using Numba with built-in RNG.

    This version uses Numba's np.random functions which are reasonably fast
    and allow inline random number generation (no pre-allocation needed).
    """
    v = np.ascontiguousarray(v, dtype=np.float32)
    a = np.ascontiguousarray(a, dtype=np.float32)
    z = np.ascontiguousarray(z, dtype=np.float32)
    t_ndt = np.ascontiguousarray(t, dtype=np.float32)

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

    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)

    # Allocate output arrays
    rts = np.zeros((n_samples, n_trials), dtype=np.float32)
    choices = np.zeros((n_samples, n_trials), dtype=np.int32)

    # Run simulation
    if parallel:
        _ddm_simulate_parallel_v2(
            v,
            a,
            z,
            t_ndt,
            deadline,
            s,
            np.float32(delta_t),
            np.float32(max_t),
            n_samples,
            n_trials,
            rts,
            choices,
        )
    else:
        _ddm_simulate_single_thread_v2(
            v,
            a,
            z,
            t_ndt,
            deadline,
            s,
            np.float32(delta_t),
            np.float32(max_t),
            n_samples,
            n_trials,
            rts,
            choices,
        )

    # Reshape output
    rts = rts.reshape(n_samples, n_trials, 1)
    choices = choices.reshape(n_samples, n_trials, 1)

    metadata = {
        "simulator": "ddm_numba_v2",
        "possible_choices": [-1, 1],
        "n_samples": n_samples,
        "n_trials": n_trials,
        "boundary_fun_type": "constant",
        "parallel": parallel,
    }

    if return_option == "full":
        metadata.update(
            {
                "delta_t": delta_t,
                "max_t": max_t,
                "v": v,
                "a": a,
                "z": z,
                "t": t_ndt,
                "deadline": deadline,
                "s": s,
            }
        )

    return {"rts": rts, "choices": choices, "metadata": metadata}


def ddm_numba_v2_single(
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
    **kwargs,
) -> Dict[str, Any]:
    """Single-threaded version for comparison."""
    return ddm_numba_v2(
        v,
        a,
        z,
        t,
        deadline,
        s,
        delta_t,
        max_t,
        n_samples,
        n_trials,
        random_state,
        return_option,
        parallel=False,
        **kwargs,
    )
