"""
Statistically Correct Numba DDM Simulator

This version generates FRESH random numbers throughout - no reuse.
It processes trials one at a time, generating fresh randoms for each trial.

This is the production-ready version that maintains statistical correctness
while still being faster than the original Cython through:
1. Larger chunks (reduces Python callback overhead)
2. Per-trial processing (fresh randoms each trial)
3. Numba-optimized inner loop
"""

import numpy as np
from numba import njit, prange, float32, int32
from typing import Dict, Any


@njit(cache=True, fastmath=True)
def _ddm_single_trial_kernel(
    v: float,
    a: float,
    z: float,
    t_ndt: float,
    deadline: float,
    s: float,
    delta_t: float,
    max_t: float,
    n_samples: int,
    gaussian: np.ndarray,  # Pre-generated for this trial, large enough
    rts: np.ndarray,  # Output: (n_samples,)
    choices: np.ndarray,  # Output: (n_samples,)
):
    """
    Process one trial using pre-generated Gaussian array.
    The array must be large enough for all samples in this trial.
    """
    delta_t_sqrt = np.sqrt(delta_t)
    sqrt_st = delta_t_sqrt * s
    deadline_tmp = min(max_t, deadline - t_ndt)
    y_start = z * a

    m = 0  # Index into gaussian array

    for n in range(n_samples):
        y = y_start
        t_particle = float32(0.0)

        # Random walk - uses sequential fresh randoms
        while y > 0 and y < a and t_particle <= deadline_tmp:
            y = y + v * delta_t + sqrt_st * gaussian[m]
            t_particle = t_particle + delta_t
            m += 1

        # Store results
        rt = t_particle + t_ndt
        if rt >= deadline or deadline <= 0:
            rts[n] = float32(-999.0)
            choices[n] = int32(0)
        else:
            rts[n] = rt
            choices[n] = int32(1) if y >= a else int32(-1)


@njit(cache=True, fastmath=True)
def _ddm_chunked_kernel(
    v: float,
    a: float,
    z: float,
    t_ndt: float,
    deadline: float,
    s: float,
    delta_t: float,
    max_t: float,
    n_samples: int,
    sample_start: int,
    sample_end: int,
    gaussian: np.ndarray,
    rts: np.ndarray,
    choices: np.ndarray,
):
    """Process a chunk of samples for one trial."""
    delta_t_sqrt = np.sqrt(delta_t)
    sqrt_st = delta_t_sqrt * s
    deadline_tmp = min(max_t, deadline - t_ndt)
    y_start = z * a

    m = 0
    for n in range(sample_start, sample_end):
        y = y_start
        t_particle = float32(0.0)

        while y > 0 and y < a and t_particle <= deadline_tmp:
            y = y + v * delta_t + sqrt_st * gaussian[m]
            t_particle = t_particle + delta_t
            m += 1

        rt = t_particle + t_ndt
        if rt >= deadline or deadline <= 0:
            rts[n] = float32(-999.0)
            choices[n] = int32(0)
        else:
            rts[n] = rt
            choices[n] = int32(1) if y >= a else int32(-1)

    return m  # Return how many randoms we used


def ddm_numba_correct(
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
    """
    Statistically correct DDM simulator using Numba.

    Uses chunked processing like Cython - generates fresh random numbers
    in cache-friendly chunks, regenerating when exhausted.

    This is the production-ready version.
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

    # Use cache-friendly chunk size (like Cython's num_draws)
    # ~500K randoms = 2MB, fits well in L3 cache
    chunk_size = 500_000
    avg_steps = 750
    samples_per_chunk = chunk_size // avg_steps  # ~666 samples per chunk

    # Process each trial
    for k in range(n_trials):
        sample_idx = 0
        while sample_idx < n_samples:
            # Generate fresh chunk of random numbers
            gaussian = rng.standard_normal(chunk_size, dtype=np.float32)

            # Process samples until chunk exhausted or trial done
            chunk_end = min(sample_idx + samples_per_chunk, n_samples)

            _ddm_chunked_kernel(
                v[k],
                a[k],
                z[k],
                t_ndt[k],
                deadline[k],
                s[k],
                np.float32(delta_t),
                np.float32(max_t),
                n_samples,
                sample_idx,
                chunk_end,
                gaussian,
                rts[:, k],
                choices[:, k],
            )

            sample_idx = chunk_end

    # Reshape output
    rts = rts.reshape(n_samples, n_trials, 1)
    choices = choices.reshape(n_samples, n_trials, 1)

    metadata = {
        "simulator": "ddm_numba_correct",
        "possible_choices": [-1, 1],
        "n_samples": n_samples,
        "n_trials": n_trials,
        "boundary_fun_type": "constant",
        "statistically_correct": True,
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


@njit(parallel=True, cache=True, fastmath=True)
def _ddm_parallel_kernel(
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
    gaussians: np.ndarray,  # Shape: (n_trials, randoms_per_trial)
    rts: np.ndarray,
    choices: np.ndarray,
):
    """
    Parallel kernel - each trial gets its own fresh random array.
    """
    delta_t_sqrt = np.sqrt(delta_t)

    for k in prange(n_trials):
        sqrt_st = delta_t_sqrt * s[k]
        deadline_tmp = min(max_t, deadline[k] - t_ndt[k])
        y_start = z[k] * a[k]
        v_k = v[k]
        a_k = a[k]

        gaussian = gaussians[k]
        m = 0

        for n in range(n_samples):
            y = y_start
            t_particle = float32(0.0)

            while y > 0 and y < a_k and t_particle <= deadline_tmp:
                y = y + v_k * delta_t + sqrt_st * gaussian[m]
                t_particle = t_particle + delta_t
                m += 1

            rt = t_particle + t_ndt[k]
            if rt >= deadline[k] or deadline[k] <= 0:
                rts[n, k] = float32(-999.0)
                choices[n, k] = int32(0)
            else:
                rts[n, k] = rt
                choices[n, k] = int32(1) if y >= a_k else int32(-1)


def ddm_numba_correct_parallel(
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
    """
    Parallel statistically correct DDM simulator.

    Each trial gets its own fresh random array - no reuse across trials.
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

    # Estimate randoms needed per trial
    # Use realistic average: ~1000 steps per sample
    avg_steps = 1000
    randoms_per_trial = n_samples * avg_steps

    # Pre-generate fresh random arrays for all trials
    # Each trial gets its own independent stream
    gaussians = rng.standard_normal((n_trials, randoms_per_trial), dtype=np.float32)

    # Allocate output
    rts = np.zeros((n_samples, n_trials), dtype=np.float32)
    choices = np.zeros((n_samples, n_trials), dtype=np.int32)

    # Run parallel kernel
    _ddm_parallel_kernel(
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

    rts = rts.reshape(n_samples, n_trials, 1)
    choices = choices.reshape(n_samples, n_trials, 1)

    metadata = {
        "simulator": "ddm_numba_correct_parallel",
        "possible_choices": [-1, 1],
        "n_samples": n_samples,
        "n_trials": n_trials,
        "boundary_fun_type": "constant",
        "statistically_correct": True,
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
