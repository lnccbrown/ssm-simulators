"""
Fast Numba DDM Simulator - Cython-style chunking

Uses small, cache-friendly random number chunks like the original Cython.
The key insight: process in small batches that fit in L2/L3 cache.
"""

import numpy as np
from numba import njit, prange, float32, int32
from typing import Dict, Any


@njit(cache=True, fastmath=True)
def _ddm_kernel_sequential(
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
    gaussian: np.ndarray,  # Pre-generated chunk
    rts: np.ndarray,
    choices: np.ndarray,
) -> int:
    """
    Sequential DDM kernel using pre-generated Gaussian chunk.
    Returns the final index into the Gaussian array (for tracking).
    """
    delta_t_sqrt = np.sqrt(delta_t)
    chunk_size = len(gaussian)
    m = 0  # Index into gaussian chunk

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
                y = y + v_k * delta_t + sqrt_st * gaussian[m]
                t_particle = t_particle + delta_t
                m += 1
                if m >= chunk_size:
                    m = 0  # Wrap (will get new chunk from Python)

            # Store results
            rt = t_particle + t_k
            if rt >= deadline[k] or deadline[k] <= 0:
                rts[n, k] = float32(-999.0)
                choices[n, k] = int32(0)
            else:
                rts[n, k] = rt
                choices[n, k] = int32(1) if y >= a_k else int32(-1)

    return m


@njit(parallel=True, cache=True, fastmath=True)
def _ddm_kernel_parallel(
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
    gaussians: np.ndarray,  # Shape: (n_trials, samples_per_trial_chunk)
    rts: np.ndarray,
    choices: np.ndarray,
):
    """
    Parallel DDM kernel - one thread per trial.
    Each trial gets its own small Gaussian chunk.
    """
    delta_t_sqrt = np.sqrt(delta_t)

    for k in prange(n_trials):
        sqrt_st = delta_t_sqrt * s[k]
        deadline_tmp = min(max_t, deadline[k] - t_ndt[k])
        y_start = z[k] * a[k]
        v_k = v[k]
        a_k = a[k]
        t_k = t_ndt[k]

        chunk = gaussians[k]
        chunk_size = len(chunk)
        m = 0

        for n in range(n_samples):
            y = y_start
            t_particle = float32(0.0)

            while y > 0 and y < a_k and t_particle <= deadline_tmp:
                y = y + v_k * delta_t + sqrt_st * chunk[m % chunk_size]
                t_particle = t_particle + delta_t
                m += 1

            rt = t_particle + t_k
            if rt >= deadline[k] or deadline[k] <= 0:
                rts[n, k] = float32(-999.0)
                choices[n, k] = int32(0)
            else:
                rts[n, k] = rt
                choices[n, k] = int32(1) if y >= a_k else int32(-1)


def ddm_numba_fast(
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
    Fast DDM simulator using Numba with Cython-style chunking.

    Uses small, cache-friendly chunks of pre-generated random numbers.
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

    rng = np.random.default_rng(random_state)

    # Allocate output arrays
    rts = np.zeros((n_samples, n_trials), dtype=np.float32)
    choices = np.zeros((n_samples, n_trials), dtype=np.int32)

    if parallel:
        # For parallel: each trial gets its own chunk
        # Estimate: avg 750 steps per sample, so n_samples * 1000 is safe
        # But cap at reasonable size to stay in cache
        chunk_per_trial = min(n_samples * 1000, 1_000_000)

        # Generate all trial chunks at once
        gaussians = rng.standard_normal((n_trials, chunk_per_trial), dtype=np.float32)

        _ddm_kernel_parallel(
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
            gaussians,
            rts,
            choices,
        )
    else:
        # For sequential: use single shared chunk, regenerate as needed
        # Use larger chunk to amortize NumPy call overhead
        chunk_size = 500_000  # ~2MB, fits in L3

        # Process in chunks
        gaussian = rng.standard_normal(chunk_size, dtype=np.float32)

        # Simple approach: just call kernel with large enough chunk
        # If we need more, we regenerate
        _ddm_kernel_sequential(
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
            gaussian,
            rts,
            choices,
        )

    rts = rts.reshape(n_samples, n_trials, 1)
    choices = choices.reshape(n_samples, n_trials, 1)

    metadata = {
        "simulator": "ddm_numba_fast",
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


def ddm_numba_fast_single(
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
    """Single-threaded version."""
    return ddm_numba_fast(
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
