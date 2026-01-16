"""
Optimized Numba-based DDM Simulators

This version uses NumPy's fast Ziggurat-based RNG for Gaussian generation
(pre-generated before the hot loop), matching the original Cython approach.

Key insight: NumPy's RNG generates ~250M samples/s vs ~10M for custom Box-Muller.
"""

import numpy as np
from numba import njit, prange, float32, int32
from typing import Dict, Any


@njit(cache=True, fastmath=True)
def _ddm_simulate_batch_pregenerated(
    v: np.ndarray,  # (n_trials,)
    a: np.ndarray,
    z: np.ndarray,
    t_ndt: np.ndarray,
    deadline: np.ndarray,
    s: np.ndarray,
    noise: np.ndarray,  # Pre-generated: (n_samples, n_trials, max_steps)
    delta_t: float,
    max_t: float,
    n_samples: int,
    n_trials: int,
    max_steps: int,
    rts: np.ndarray,  # Output: (n_samples, n_trials)
    choices: np.ndarray,  # Output: (n_samples, n_trials)
):
    """
    Core DDM simulation using pre-generated noise.
    Single-threaded version for baseline comparison.
    """
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
            step = 0

            # Random walk
            while y > 0 and y < a_k and t_particle <= deadline_tmp and step < max_steps:
                y = y + v_k * delta_t + sqrt_st * noise[n, k, step]
                t_particle = t_particle + delta_t
                step += 1

            # Store results
            rt = t_particle + t_k
            if rt >= deadline[k] or deadline[k] <= 0:
                rts[n, k] = float32(-999.0)
                choices[n, k] = int32(0)
            else:
                rts[n, k] = rt
                choices[n, k] = int32(1) if y > 0 else int32(-1)


@njit(parallel=True, cache=True, fastmath=True)
def _ddm_simulate_batch_parallel(
    v: np.ndarray,
    a: np.ndarray,
    z: np.ndarray,
    t_ndt: np.ndarray,
    deadline: np.ndarray,
    s: np.ndarray,
    noise: np.ndarray,  # Pre-generated: (n_samples, n_trials, max_steps)
    delta_t: float,
    max_t: float,
    n_samples: int,
    n_trials: int,
    max_steps: int,
    rts: np.ndarray,
    choices: np.ndarray,
):
    """
    Parallel DDM simulation using pre-generated noise.
    Parallelizes across the flattened (sample, trial) space.
    """
    delta_t_sqrt = np.sqrt(delta_t)
    total_sims = n_samples * n_trials

    for idx in prange(total_sims):
        n = idx // n_trials
        k = idx % n_trials

        sqrt_st = delta_t_sqrt * s[k]
        deadline_tmp = min(max_t, deadline[k] - t_ndt[k])

        y = z[k] * a[k]
        t_particle = float32(0.0)
        step = 0
        v_k = v[k]
        a_k = a[k]

        # Random walk
        while y > 0 and y < a_k and t_particle <= deadline_tmp and step < max_steps:
            y = y + v_k * delta_t + sqrt_st * noise[n, k, step]
            t_particle = t_particle + delta_t
            step += 1

        # Store results
        rt = t_particle + t_ndt[k]
        if rt >= deadline[k] or deadline[k] <= 0:
            rts[n, k] = float32(-999.0)
            choices[n, k] = int32(0)
        else:
            rts[n, k] = rt
            choices[n, k] = int32(1) if y > 0 else int32(-1)


def ddm_numba_optimized(
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
    Optimized DDM simulator using Numba with NumPy's fast RNG.

    This version processes trials in batches to avoid excessive memory allocation
    for pre-generated noise.

    Parameters
    ----------
    parallel : bool
        Use parallel execution (default: True)
    """
    # Ensure arrays are float32 and contiguous
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

    # Calculate max steps needed
    max_steps = int((max_t / delta_t) + 1)

    # Allocate output arrays
    rts = np.zeros((n_samples, n_trials), dtype=np.float32)
    choices = np.zeros((n_samples, n_trials), dtype=np.int32)

    # Process in chunks to limit memory usage
    # Target ~500MB for noise array
    max_noise_elements = 500_000_000 // 4  # 500MB / 4 bytes per float32
    chunk_size = max(1, max_noise_elements // (n_trials * max_steps))
    chunk_size = min(chunk_size, n_samples)

    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        current_chunk = end_idx - start_idx

        # Pre-generate noise for this chunk
        noise = rng.standard_normal(
            size=(current_chunk, n_trials, max_steps), dtype=np.float32
        )

        # Create views for this chunk
        rts_chunk = rts[start_idx:end_idx]
        choices_chunk = choices[start_idx:end_idx]

        # Run simulation
        if parallel:
            _ddm_simulate_batch_parallel(
                v,
                a,
                z,
                t_ndt,
                deadline,
                s,
                noise,
                np.float32(delta_t),
                np.float32(max_t),
                current_chunk,
                n_trials,
                max_steps,
                rts_chunk,
                choices_chunk,
            )
        else:
            _ddm_simulate_batch_pregenerated(
                v,
                a,
                z,
                t_ndt,
                deadline,
                s,
                noise,
                np.float32(delta_t),
                np.float32(max_t),
                current_chunk,
                n_trials,
                max_steps,
                rts_chunk,
                choices_chunk,
            )

        # Copy results back
        rts[start_idx:end_idx] = rts_chunk
        choices[start_idx:end_idx] = choices_chunk

    # Reshape to match expected output format
    rts = rts.reshape(n_samples, n_trials, 1)
    choices = choices.reshape(n_samples, n_trials, 1)

    # Build metadata
    metadata = {
        "simulator": "ddm_numba_optimized",
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


def ddm_numba_optimized_single(
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
    return ddm_numba_optimized(
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
