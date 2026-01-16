"""
Cython nogil Parallel Backend

This module provides multi-threaded DDM simulators using Cython's prange
with nogil for true parallel execution.

The key differences from the standard Cython implementation:
1. Uses prange instead of range for parallel loops
2. Releases the GIL using 'with nogil:' blocks
3. Uses thread-local random state for reproducibility
4. Avoids Python object creation in hot loops

Example:
    from ssms.parallel_backends.cython_parallel import ddm_parallel

    result = ddm_parallel(
        v=np.array([0.5] * 100, dtype=np.float32),
        a=np.array([1.5] * 100, dtype=np.float32),
        z=np.array([0.5] * 100, dtype=np.float32),
        t=np.array([0.3] * 100, dtype=np.float32),
        n_samples=10000,
        n_threads=8
    )
"""

# Note: The actual parallel implementation is in ddm_nogil.pyx
# This module provides the Python interface

try:
    from ssms.parallel_backends.cython_parallel.ddm_nogil import (
        ddm_parallel,
        ddm_flexbound_parallel,
        full_ddm_parallel,
    )

    CYTHON_PARALLEL_AVAILABLE = True
except ImportError:
    CYTHON_PARALLEL_AVAILABLE = False

    def _not_available(*args, **kwargs):
        raise ImportError(
            "Cython parallel backend not available. "
            "Please compile the Cython extensions with: "
            "pip install -e '.[dev]' or python setup.py build_ext --inplace"
        )

    ddm_parallel = _not_available
    ddm_flexbound_parallel = _not_available
    full_ddm_parallel = _not_available

__all__ = [
    "ddm_parallel",
    "ddm_flexbound_parallel",
    "full_ddm_parallel",
    "CYTHON_PARALLEL_AVAILABLE",
]
