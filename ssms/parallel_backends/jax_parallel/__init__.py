"""
JAX Parallel Backend

This module provides vectorized DDM simulators using JAX's transformations
for automatic vectorization and XLA compilation.

Key features:
1. Automatic vectorization with vmap
2. XLA compilation for CPU/GPU/TPU
3. Efficient batched operations
4. GPU acceleration when available

Example:
    from ssms.parallel_backends.jax_parallel import ddm_jax

    result = ddm_jax(
        v=np.array([0.5] * 100, dtype=np.float32),
        a=np.array([1.5] * 100, dtype=np.float32),
        z=np.array([0.5] * 100, dtype=np.float32),
        t=np.array([0.3] * 100, dtype=np.float32),
        n_samples=10000,
    )
"""

try:
    from ssms.parallel_backends.jax_parallel.ddm_jax import (
        ddm_jax,
        ddm_flexbound_jax,
        full_ddm_jax,
        ddm_jax_vectorized,
        get_jax_device_info,
    )

    JAX_AVAILABLE = True
except ImportError as e:
    JAX_AVAILABLE = False
    _import_error = str(e)

    def _not_available(*args, **kwargs):
        raise ImportError(
            f"JAX parallel backend not available. "
            f"Please install JAX: pip install jax jaxlib. "
            f"For GPU support: pip install 'jax[cuda12]'. "
            f"Original error: {_import_error}"
        )

    ddm_jax = _not_available
    ddm_flexbound_jax = _not_available
    full_ddm_jax = _not_available
    ddm_jax_vectorized = _not_available
    get_jax_device_info = lambda: {"available": False, "error": _import_error}

__all__ = [
    "ddm_jax",
    "ddm_flexbound_jax",
    "full_ddm_jax",
    "ddm_jax_vectorized",
    "get_jax_device_info",
    "JAX_AVAILABLE",
]
