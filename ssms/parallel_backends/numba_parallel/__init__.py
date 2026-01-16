"""
Numba Parallel Backend

This module provides multi-threaded DDM simulators using Numba's JIT compilation
with automatic parallelization.

Key features:
1. No separate compilation step required
2. Automatic parallelization with parallel=True
3. SIMD vectorization for inner loops
4. GPU support via CUDA (if available)

Example:
    from ssms.parallel_backends.numba_parallel import ddm_numba

    result = ddm_numba(
        v=np.array([0.5] * 100, dtype=np.float32),
        a=np.array([1.5] * 100, dtype=np.float32),
        z=np.array([0.5] * 100, dtype=np.float32),
        t=np.array([0.3] * 100, dtype=np.float32),
        n_samples=10000,
    )
"""

try:
    from ssms.parallel_backends.numba_parallel.ddm_numba import (
        ddm_numba,
        ddm_flexbound_numba,
        full_ddm_numba,
        ddm_numba_single,
    )
    from ssms.parallel_backends.numba_parallel.ddm_numba_optimized import (
        ddm_numba_optimized,
        ddm_numba_optimized_single,
    )

    NUMBA_AVAILABLE = True
except ImportError as e:
    NUMBA_AVAILABLE = False
    _import_error = str(e)

    def _not_available(*args, **kwargs):
        raise ImportError(
            f"Numba parallel backend not available. "
            f"Please install numba: pip install numba. "
            f"Original error: {_import_error}"
        )

    ddm_numba = _not_available
    ddm_flexbound_numba = _not_available
    full_ddm_numba = _not_available
    ddm_numba_single = _not_available
    ddm_numba_optimized = _not_available
    ddm_numba_optimized_single = _not_available

__all__ = [
    "ddm_numba",
    "ddm_flexbound_numba",
    "full_ddm_numba",
    "ddm_numba_single",
    "ddm_numba_optimized",
    "ddm_numba_optimized_single",
    "NUMBA_AVAILABLE",
]
