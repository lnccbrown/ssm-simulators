"""
Numba DDM Simulator with Chunked RNG (matching Cython strategy)

This implementation uses the same chunking strategy as the original Cython:
- Pre-generate chunks of Gaussian random numbers using NumPy's fast Ziggurat
- Pass chunks to Numba kernels
- Process trials in batches to limit memory while maximizing throughput
"""

import numpy as np
from numba import njit, prange, float32, int32
from typing import Dict, Any


@njit(cache=True, fastmath=True)
def _ddm_single_trial_chunked(
    v: float,
    a: float,
    z: float,
    t_ndt: float,
    deadline: float,
    s: float,
    delta_t: float,
    max_t: float,
    n_samples: int,
    gaussian_chunk: np.ndarray,  # Pre-generated Gaussians
    rts: np.ndarray,  # Output: (n_samples,)
    choices: np.ndarray,  # Output: (n_samples,)
) -> int:
    """
    Simulate one trial using pre-generated Gaussian chunk.
    Returns the number of Gaussian values consumed.
    """
    delta_t_sqrt = np.sqrt(delta_t)
    sqrt_st = delta_t_sqrt * s
    deadline_tmp = min(max_t, deadline - t_ndt)
    y_start = z * a

    m = 0  # Index into gaussian_chunk
    chunk_size = len(gaussian_chunk)

    for n in range(n_samples):
        y = y_start
        t_particle = float32(0.0)

        # Random walk
        while y > 0 and y < a and t_particle <= deadline_tmp:
            # Use pre-generated Gaussian
            y = y + v * delta_t + sqrt_st * gaussian_chunk[m]
            t_particle = t_particle + delta_t
            m += 1

            # Safety check (shouldn't happen with proper chunk sizing)
            if m >= chunk_size:
                m = 0  # Wrap around (reuse randoms - not ideal but safe)

        # Store results
        rt = t_particle + t_ndt
        if rt >= deadline or deadline <= 0:
            rts[n] = float32(-999.0)
            choices[n] = int32(0)
        else:
            rts[n] = rt
            choices[n] = int32(1) if y >= a else int32(-1)

    return m  # Return how many we used


@njit(parallel=True, cache=True, fastmath=True)
def _ddm_parallel_with_chunks(
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
    gaussian_chunks: np.ndarray,  # Shape: (n_trials, chunk_size)
    rts: np.ndarray,  # Shape: (n_samples, n_trials)
    choices: np.ndarray,  # Shape: (n_samples, n_trials)
):
    """
    Parallel DDM simulation with pre-generated Gaussian chunks per trial.
    """
    delta_t_sqrt = np.sqrt(delta_t)

    for k in prange(n_trials):
        sqrt_st = delta_t_sqrt * s[k]
        deadline_tmp = min(max_t, deadline[k] - t_ndt[k])
        y_start = z[k] * a[k]
        v_k = v[k]
        a_k = a[k]

        # Get this trial's Gaussian chunk
        chunk = gaussian_chunks[k]
        chunk_size = len(chunk)
        m = 0

        for n in range(n_samples):
            y = y_start
            t_particle = float32(0.0)

            # Random walk
            while y > 0 and y < a_k and t_particle <= deadline_tmp:
                y = y + v_k * delta_t + sqrt_st * chunk[m % chunk_size]
                t_particle = t_particle + delta_t
                m += 1

            # Store results
            rt = t_particle + t_ndt[k]
            if rt >= deadline[k] or deadline[k] <= 0:
                rts[n, k] = float32(-999.0)
                choices[n, k] = int32(0)
            else:
                rts[n, k] = rt
                choices[n, k] = int32(1) if y >= a_k else int32(-1)


@njit(cache=True, fastmath=True)
def _ddm_sequential_with_chunks(
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
    gaussian_chunks: np.ndarray,  # Shape: (n_trials, chunk_size)
    rts: np.ndarray,
    choices: np.ndarray,
):
    """
    Sequential DDM simulation with pre-generated Gaussian chunks.
    """
    delta_t_sqrt = np.sqrt(delta_t)

    for k in range(n_trials):
        sqrt_st = delta_t_sqrt * s[k]
        deadline_tmp = min(max_t, deadline[k] - t_ndt[k])
        y_start = z[k] * a[k]
        v_k = v[k]
        a_k = a[k]

        chunk = gaussian_chunks[k]
        chunk_size = len(chunk)
        m = 0

        for n in range(n_samples):
            y = y_start
            t_particle = float32(0.0)

            while y > 0 and y < a_k and t_particle <= deadline_tmp:
                y = y + v_k * delta_t + sqrt_st * chunk[m % chunk_size]
                t_particle = t_particle + delta_t
                m += 1

            rt = t_particle + t_ndt[k]
            if rt >= deadline[k] or deadline[k] <= 0:
                rts[n, k] = float32(-999.0)
                choices[n, k] = int32(0)
            else:
                rts[n, k] = rt
                choices[n, k] = int32(1) if y >= a_k else int32(-1)


def ddm_numba_chunked(
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
    DDM simulator using Numba with chunked RNG (matching Cython strategy).

    Pre-generates Gaussian random numbers using NumPy's fast Ziggurat algorithm,
    then passes them to Numba for the simulation loop.
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

    # Initialize RNG
    rng = np.random.default_rng(random_state)

    # Estimate chunk size per trial
    # Typical DDM: avg ~500-1000 steps per sample, so n_samples * 1000 is conservative
    # But we use max_steps to be safe
    max_steps = int(max_t / delta_t) + 1
    chunk_size = n_samples * max_steps

    # Cap chunk size to avoid excessive memory (aim for ~50MB per trial max)
    max_chunk = 50_000_000 // 4  # 50MB / 4 bytes
    chunk_size = min(chunk_size, max_chunk)

    # Pre-generate Gaussian chunks for all trials
    # Shape: (n_trials, chunk_size)
    gaussian_chunks = rng.standard_normal(size=(n_trials, chunk_size), dtype=np.float32)

    # Allocate output arrays
    rts = np.zeros((n_samples, n_trials), dtype=np.float32)
    choices = np.zeros((n_samples, n_trials), dtype=np.int32)

    # Run simulation
    if parallel:
        _ddm_parallel_with_chunks(
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
            gaussian_chunks,
            rts,
            choices,
        )
    else:
        _ddm_sequential_with_chunks(
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
            gaussian_chunks,
            rts,
            choices,
        )

    # Reshape output
    rts = rts.reshape(n_samples, n_trials, 1)
    choices = choices.reshape(n_samples, n_trials, 1)

    metadata = {
        "simulator": "ddm_numba_chunked",
        "possible_choices": [-1, 1],
        "n_samples": n_samples,
        "n_trials": n_trials,
        "boundary_fun_type": "constant",
        "parallel": parallel,
        "chunk_size": chunk_size,
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


def ddm_numba_chunked_single(
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
    return ddm_numba_chunked(
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
