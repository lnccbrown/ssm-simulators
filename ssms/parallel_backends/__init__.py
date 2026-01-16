"""
Parallel Backends for SSM Simulators

This module provides four different parallel/multi-threaded implementations
for Sequential Sampling Model simulations:

1. **Cython nogil** (`cython_parallel`):
   Uses Cython's prange with nogil for true multi-threaded parallelism.
   Best for CPU-bound workloads when you want to stay close to the original implementation.

2. **Numba** (`numba_parallel`):
   Uses Numba's JIT compilation with parallel=True for automatic parallelization.
   Good balance of ease-of-use and performance, no compilation step needed.

3. **JAX** (`jax_parallel`):
   Uses JAX's vmap and jit for vectorization and XLA compilation.
   Excellent for GPU acceleration and very large batches.

4. **Rust** (`rust_parallel`):
   Uses Rust via PyO3 for maximum performance with Rayon for parallelism.
   Best raw performance but requires Rust toolchain.

Example usage:

    from ssms.parallel_backends import benchmark_all

    # Compare all implementations
    results = benchmark_all(
        model='ddm',
        n_samples=10000,
        n_trials=100,
        n_threads=8
    )

    # Use a specific backend
    from ssms.parallel_backends.numba_parallel import ddm_numba
    out = ddm_numba(v=0.5, a=1.5, z=0.5, t=0.3, n_samples=1000)

"""

from ssms.parallel_backends.benchmark import benchmark_all, compare_backends

__all__ = [
    "benchmark_all",
    "compare_backends",
]


# Lazy imports for backends that may have optional dependencies
def get_cython_backend():
    """Get Cython nogil parallel backend."""
    from ssms.parallel_backends import cython_parallel

    return cython_parallel


def get_numba_backend():
    """Get Numba parallel backend."""
    from ssms.parallel_backends import numba_parallel

    return numba_parallel


def get_jax_backend():
    """Get JAX parallel backend."""
    from ssms.parallel_backends import jax_parallel

    return jax_parallel


def get_rust_backend():
    """Get Rust parallel backend."""
    from ssms.parallel_backends import rust_parallel

    return rust_parallel
