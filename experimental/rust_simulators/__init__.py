"""
Rust Parallel Backend

This module provides high-performance DDM simulators implemented in Rust
using Rayon for parallelism and PyO3 for Python bindings.

Key features:
1. Native Rust performance with zero-copy NumPy integration
2. Rayon for work-stealing parallel execution
3. SIMD optimizations where possible
4. Memory-safe implementation

Setup:
    The Rust backend requires compilation. To build:

    1. Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    2. Install maturin: pip install maturin
    3. Build the extension: cd ssms/parallel_backends/rust_parallel && maturin develop --release

Example:
    from ssms.parallel_backends.rust_parallel import ddm_rust

    result = ddm_rust(
        v=np.array([0.5] * 100, dtype=np.float32),
        a=np.array([1.5] * 100, dtype=np.float32),
        z=np.array([0.5] * 100, dtype=np.float32),
        t=np.array([0.3] * 100, dtype=np.float32),
        n_samples=10000,
    )
"""

import numpy as np
from typing import Dict, Any

try:
    from ssms_rust import (
        ddm_rust_sequential,
        ddm_rust_parallel,
        ddm_flexbound_rust_parallel,
        full_ddm_rust_parallel,
        get_rust_info,
    )

    RUST_AVAILABLE = True
except ImportError as e:
    RUST_AVAILABLE = False
    _import_error = str(e)
    ddm_rust_sequential = None

    def _not_available(*args, **kwargs):
        raise ImportError(
            f"Rust parallel backend not available. "
            f"Please build the Rust extension:\n"
            f"  1. Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh\n"
            f"  2. Install maturin: pip install maturin\n"
            f"  3. Build: cd ssms/parallel_backends/rust_parallel && maturin develop --release\n"
            f"Original error: {_import_error}"
        )

    ddm_rust_parallel = _not_available
    ddm_rust_sequential = _not_available
    ddm_flexbound_rust_parallel = _not_available
    full_ddm_rust_parallel = _not_available

    def get_rust_info():
        return {"available": False, "error": _import_error}


def ddm_rust_single(
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
) -> Dict[str, Any]:
    """
    Single-threaded DDM simulator using Rust (no Rayon overhead).

    This provides a fair single-threaded comparison against Cython and Numba.
    """
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
        random_state = np.random.default_rng().integers(0, 2**63)

    # Call Rust sequential implementation
    rts, choices = ddm_rust_sequential(
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
        smooth_unif,
    )

    rts = rts.reshape(n_samples, n_trials, 1)
    choices = choices.reshape(n_samples, n_trials, 1)

    metadata = {
        "simulator": "ddm_rust_single",
        "possible_choices": [-1, 1],
        "n_samples": n_samples,
        "n_trials": n_trials,
        "n_threads": 1,
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


def ddm_rust(
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
    n_threads: int = 0,
    random_state: int | None = None,
    return_option: str = "full",
    smooth_unif: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    DDM simulator using Rust with Rayon parallelism.

    This function provides maximum performance through native Rust
    implementation with SIMD optimizations and Rayon's work-stealing
    parallel execution.

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
    n_threads : int
        Number of threads (0 = auto, default: 0)
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
        random_state = np.random.default_rng().integers(0, 2**63)

    if n_threads <= 0:
        import os

        n_threads = os.cpu_count() or 4

    # Call Rust implementation
    rts, choices = ddm_rust_parallel(
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
        n_threads,
        random_state,
        smooth_unif,
    )

    # Reshape outputs
    rts = rts.reshape(n_samples, n_trials, 1)
    choices = choices.reshape(n_samples, n_trials, 1)

    # Build metadata
    metadata = {
        "simulator": "ddm_rust",
        "possible_choices": [-1, 1],
        "n_samples": n_samples,
        "n_trials": n_trials,
        "n_threads": n_threads,
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


def ddm_flexbound_rust(
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
    n_threads: int = 0,
    boundary_fun=None,
    boundary_params: dict | None = None,
    random_state: int | None = None,
    return_option: str = "full",
    smooth_unif: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """DDM with flexible boundaries using Rust."""
    if boundary_params is None:
        boundary_params = {}

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
        random_state = np.random.default_rng().integers(0, 2**63)

    if n_threads <= 0:
        import os

        n_threads = os.cpu_count() or 4

    # Precompute boundary
    num_steps = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t, dtype=np.float32)
    boundary = np.zeros((n_trials, num_steps), dtype=np.float32)

    for k in range(n_trials):
        boundary_params_tmp = {
            key_: boundary_params[key_][k] for key_ in boundary_params.keys()
        }
        boundary[k, : len(t_s)] = boundary_fun(t=t_s, **boundary_params_tmp).astype(
            np.float32
        )[:num_steps]

    boundary = np.ascontiguousarray(boundary)

    # Call Rust implementation
    rts, choices = ddm_flexbound_rust_parallel(
        v,
        z,
        t,
        deadline,
        s,
        boundary,
        delta_t,
        max_t,
        n_samples,
        n_trials,
        num_steps,
        n_threads,
        random_state,
        smooth_unif,
    )

    rts = rts.reshape(n_samples, n_trials, 1)
    choices = choices.reshape(n_samples, n_trials, 1)

    metadata = {
        "simulator": "ddm_flexbound_rust",
        "possible_choices": [-1, 1],
        "n_samples": n_samples,
        "n_trials": n_trials,
        "n_threads": n_threads,
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


def full_ddm_rust(
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
    n_threads: int = 0,
    boundary_fun=None,
    boundary_params: dict | None = None,
    random_state: int | None = None,
    return_option: str = "full",
    smooth_unif: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Full DDM with inter-trial variability using Rust."""
    if boundary_params is None:
        boundary_params = {}

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

    if random_state is None:
        random_state = np.random.default_rng().integers(0, 2**63)

    if n_threads <= 0:
        import os

        n_threads = os.cpu_count() or 4

    # Precompute boundary
    num_steps = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t, dtype=np.float32)
    boundary = np.zeros((n_trials, num_steps), dtype=np.float32)

    for k in range(n_trials):
        boundary_params_tmp = {
            key_: boundary_params[key_][k] for key_ in boundary_params.keys()
        }
        boundary[k, : len(t_s)] = boundary_fun(t=t_s, **boundary_params_tmp).astype(
            np.float32
        )[:num_steps]

    boundary = np.ascontiguousarray(boundary)

    # Call Rust implementation
    rts, choices = full_ddm_rust_parallel(
        v,
        z,
        t,
        sz,
        sv,
        st,
        deadline,
        s,
        boundary,
        delta_t,
        max_t,
        n_samples,
        n_trials,
        num_steps,
        n_threads,
        random_state,
        smooth_unif,
    )

    rts = rts.reshape(n_samples, n_trials, 1)
    choices = choices.reshape(n_samples, n_trials, 1)

    metadata = {
        "simulator": "full_ddm_rust",
        "possible_choices": [-1, 1],
        "n_samples": n_samples,
        "n_trials": n_trials,
        "n_threads": n_threads,
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


__all__ = [
    "ddm_rust",
    "ddm_rust_single",
    "ddm_flexbound_rust",
    "full_ddm_rust",
    "get_rust_info",
    "RUST_AVAILABLE",
]
