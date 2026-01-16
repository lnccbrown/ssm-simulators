"""
Benchmark and Comparison Tools for Parallel Backends

This module provides utilities to benchmark and compare the performance
of different parallel backend implementations for DDM simulators.

Usage:
    python -m ssms.parallel_backends.benchmark --n-samples 10000 --n-trials 100

    Or from Python:
        from ssms.parallel_backends.benchmark import benchmark_all, compare_backends

        results = benchmark_all(n_samples=10000, n_trials=100)
        compare_backends(results)
"""

import time
import numpy as np
from typing import Dict, Callable, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    backend: str
    n_samples: int
    n_trials: int
    n_threads: int
    time_seconds: float
    samples_per_second: float
    available: bool
    error: Optional[str] = None

    def __repr__(self):
        if self.available and self.error is None:
            return (
                f"BenchmarkResult(backend='{self.backend}', "
                f"time={self.time_seconds:.3f}s, "
                f"throughput={self.samples_per_second / 1e6:.2f}M samples/s)"
            )
        else:
            return f"BenchmarkResult(backend='{self.backend}', error='{self.error}')"


def _create_test_params(n_trials: int, random_state: int = 42) -> Dict[str, np.ndarray]:
    """Create test parameters for benchmarking."""
    rng = np.random.default_rng(random_state)

    return {
        "v": rng.uniform(0.3, 0.8, n_trials).astype(np.float32),
        "a": rng.uniform(1.0, 2.0, n_trials).astype(np.float32),
        "z": rng.uniform(0.4, 0.6, n_trials).astype(np.float32),
        "t": rng.uniform(0.2, 0.4, n_trials).astype(np.float32),
        "deadline": np.full(n_trials, 999.0, dtype=np.float32),
        "s": np.ones(n_trials, dtype=np.float32),
    }


def _warmup(
    func: Callable, params: Dict, n_samples: int, n_trials: int, n_warmup: int = 2
):
    """Warmup function to trigger JIT compilation."""
    for _ in range(n_warmup):
        _ = func(**params, n_samples=min(100, n_samples), n_trials=min(5, n_trials))


def benchmark_original_cython(
    params: Dict[str, np.ndarray],
    n_samples: int,
    n_trials: int,
    n_threads: int = 1,
    n_runs: int = 3,
    random_state: int = 42,
) -> BenchmarkResult:
    """Benchmark the original Cython implementation."""
    try:
        from cssm.ddm_models import ddm

        # Warmup
        _warmup(ddm, params, n_samples, n_trials)

        # Benchmark
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = ddm(
                **params,
                n_samples=n_samples,
                n_trials=n_trials,
                random_state=random_state,
                return_option="minimal",
            )
            times.append(time.perf_counter() - start)

        avg_time = np.median(times)
        total_samples = n_samples * n_trials

        return BenchmarkResult(
            backend="cython_original",
            n_samples=n_samples,
            n_trials=n_trials,
            n_threads=1,
            time_seconds=avg_time,
            samples_per_second=total_samples / avg_time,
            available=True,
        )
    except ImportError as e:
        return BenchmarkResult(
            backend="cython_original",
            n_samples=n_samples,
            n_trials=n_trials,
            n_threads=1,
            time_seconds=0,
            samples_per_second=0,
            available=False,
            error=str(e),
        )
    except Exception as e:
        return BenchmarkResult(
            backend="cython_original",
            n_samples=n_samples,
            n_trials=n_trials,
            n_threads=1,
            time_seconds=0,
            samples_per_second=0,
            available=False,
            error=str(e),
        )


def benchmark_cython_parallel(
    params: Dict[str, np.ndarray],
    n_samples: int,
    n_trials: int,
    n_threads: int = 0,
    n_runs: int = 3,
    random_state: int = 42,
) -> BenchmarkResult:
    """Benchmark the Cython nogil parallel implementation."""
    try:
        from ssms.parallel_backends.cython_parallel import ddm_parallel

        # Warmup
        _warmup(ddm_parallel, params, n_samples, n_trials)

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = ddm_parallel(
                **params,
                n_samples=n_samples,
                n_trials=n_trials,
                n_threads=n_threads,
                random_state=random_state,
                return_option="minimal",
            )
            times.append(time.perf_counter() - start)

        avg_time = np.median(times)
        total_samples = n_samples * n_trials

        import os

        actual_threads = n_threads if n_threads > 0 else (os.cpu_count() or 4)

        return BenchmarkResult(
            backend="cython_parallel",
            n_samples=n_samples,
            n_trials=n_trials,
            n_threads=actual_threads,
            time_seconds=avg_time,
            samples_per_second=total_samples / avg_time,
            available=True,
        )
    except ImportError as e:
        return BenchmarkResult(
            backend="cython_parallel",
            n_samples=n_samples,
            n_trials=n_trials,
            n_threads=n_threads,
            time_seconds=0,
            samples_per_second=0,
            available=False,
            error=str(e),
        )


def benchmark_numba(
    params: Dict[str, np.ndarray],
    n_samples: int,
    n_trials: int,
    n_threads: int = 0,
    n_runs: int = 3,
    random_state: int = 42,
) -> BenchmarkResult:
    """Benchmark the Numba parallel implementation."""
    try:
        from ssms.parallel_backends.numba_parallel import ddm_numba

        # Warmup (important for Numba JIT)
        _warmup(ddm_numba, params, n_samples, n_trials, n_warmup=3)

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = ddm_numba(
                **params,
                n_samples=n_samples,
                n_trials=n_trials,
                random_state=random_state,
                return_option="minimal",
            )
            times.append(time.perf_counter() - start)

        avg_time = np.median(times)
        total_samples = n_samples * n_trials

        import os

        actual_threads = os.cpu_count() or 4  # Numba auto-detects

        return BenchmarkResult(
            backend="numba_parallel",
            n_samples=n_samples,
            n_trials=n_trials,
            n_threads=actual_threads,
            time_seconds=avg_time,
            samples_per_second=total_samples / avg_time,
            available=True,
        )
    except ImportError as e:
        return BenchmarkResult(
            backend="numba_parallel",
            n_samples=n_samples,
            n_trials=n_trials,
            n_threads=n_threads,
            time_seconds=0,
            samples_per_second=0,
            available=False,
            error=str(e),
        )


def benchmark_numba_single(
    params: Dict[str, np.ndarray],
    n_samples: int,
    n_trials: int,
    n_runs: int = 3,
    random_state: int = 42,
) -> BenchmarkResult:
    """Benchmark single-threaded Numba for comparison."""
    try:
        from ssms.parallel_backends.numba_parallel import ddm_numba_single

        _warmup(ddm_numba_single, params, n_samples, n_trials, n_warmup=3)

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = ddm_numba_single(
                **params,
                n_samples=n_samples,
                n_trials=n_trials,
                random_state=random_state,
                return_option="minimal",
            )
            times.append(time.perf_counter() - start)

        avg_time = np.median(times)
        total_samples = n_samples * n_trials

        return BenchmarkResult(
            backend="numba_single",
            n_samples=n_samples,
            n_trials=n_trials,
            n_threads=1,
            time_seconds=avg_time,
            samples_per_second=total_samples / avg_time,
            available=True,
        )
    except ImportError as e:
        return BenchmarkResult(
            backend="numba_single",
            n_samples=n_samples,
            n_trials=n_trials,
            n_threads=1,
            time_seconds=0,
            samples_per_second=0,
            available=False,
            error=str(e),
        )


def benchmark_jax(
    params: Dict[str, np.ndarray],
    n_samples: int,
    n_trials: int,
    n_runs: int = 3,
    random_state: int = 42,
) -> BenchmarkResult:
    """Benchmark the JAX implementation."""
    try:
        from ssms.parallel_backends.jax_parallel import ddm_jax, get_jax_device_info

        device_info = get_jax_device_info()

        # Warmup (important for JAX JIT)
        _warmup(ddm_jax, params, n_samples, n_trials, n_warmup=3)

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            result = ddm_jax(
                **params,
                n_samples=n_samples,
                n_trials=n_trials,
                random_state=random_state,
                return_option="minimal",
            )
            # Force synchronization for accurate timing
            _ = (
                result["rts"].block_until_ready()
                if hasattr(result["rts"], "block_until_ready")
                else result["rts"]
            )
            times.append(time.perf_counter() - start)

        avg_time = np.median(times)
        total_samples = n_samples * n_trials

        return BenchmarkResult(
            backend=f"jax ({device_info.get('default_backend', 'cpu')})",
            n_samples=n_samples,
            n_trials=n_trials,
            n_threads=device_info.get("device_count", 1),
            time_seconds=avg_time,
            samples_per_second=total_samples / avg_time,
            available=True,
        )
    except ImportError as e:
        return BenchmarkResult(
            backend="jax",
            n_samples=n_samples,
            n_trials=n_trials,
            n_threads=1,
            time_seconds=0,
            samples_per_second=0,
            available=False,
            error=str(e),
        )


def benchmark_jax_vectorized(
    params: Dict[str, np.ndarray],
    n_samples: int,
    n_trials: int,
    n_runs: int = 3,
    random_state: int = 42,
) -> BenchmarkResult:
    """Benchmark the fully vectorized JAX implementation."""
    try:
        from ssms.parallel_backends.jax_parallel import (
            ddm_jax_vectorized,
            get_jax_device_info,
        )

        device_info = get_jax_device_info()

        # Check memory - vectorized version uses a lot
        import psutil

        available_mem_gb = psutil.virtual_memory().available / (1024**3)
        estimated_mem_gb = (n_samples * n_trials * 20000 * 4) / (
            1024**3
        )  # Rough estimate

        if estimated_mem_gb > available_mem_gb * 0.5:
            return BenchmarkResult(
                backend="jax_vectorized",
                n_samples=n_samples,
                n_trials=n_trials,
                n_threads=1,
                time_seconds=0,
                samples_per_second=0,
                available=False,
                error=f"Skipped: would use ~{estimated_mem_gb:.1f}GB (only {available_mem_gb:.1f}GB available)",
            )

        _warmup(ddm_jax_vectorized, params, n_samples, n_trials, n_warmup=2)

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            result = ddm_jax_vectorized(
                **params,
                n_samples=n_samples,
                n_trials=n_trials,
                random_state=random_state,
                return_option="minimal",
            )
            _ = (
                result["rts"].block_until_ready()
                if hasattr(result["rts"], "block_until_ready")
                else result["rts"]
            )
            times.append(time.perf_counter() - start)

        avg_time = np.median(times)
        total_samples = n_samples * n_trials

        return BenchmarkResult(
            backend=f"jax_vectorized ({device_info.get('default_backend', 'cpu')})",
            n_samples=n_samples,
            n_trials=n_trials,
            n_threads=device_info.get("device_count", 1),
            time_seconds=avg_time,
            samples_per_second=total_samples / avg_time,
            available=True,
        )
    except Exception as e:
        return BenchmarkResult(
            backend="jax_vectorized",
            n_samples=n_samples,
            n_trials=n_trials,
            n_threads=1,
            time_seconds=0,
            samples_per_second=0,
            available=False,
            error=str(e),
        )


def benchmark_rust(
    params: Dict[str, np.ndarray],
    n_samples: int,
    n_trials: int,
    n_threads: int = 0,
    n_runs: int = 3,
    random_state: int = 42,
) -> BenchmarkResult:
    """Benchmark the Rust parallel implementation."""
    try:
        from ssms.parallel_backends.rust_parallel import ddm_rust, get_rust_info

        _ = get_rust_info()  # Verify Rust backend is available

        # Warmup
        _warmup(ddm_rust, params, n_samples, n_trials)

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = ddm_rust(
                **params,
                n_samples=n_samples,
                n_trials=n_trials,
                n_threads=n_threads,
                random_state=random_state,
                return_option="minimal",
            )
            times.append(time.perf_counter() - start)

        avg_time = np.median(times)
        total_samples = n_samples * n_trials

        import os

        actual_threads = n_threads if n_threads > 0 else (os.cpu_count() or 4)

        return BenchmarkResult(
            backend="rust_parallel",
            n_samples=n_samples,
            n_trials=n_trials,
            n_threads=actual_threads,
            time_seconds=avg_time,
            samples_per_second=total_samples / avg_time,
            available=True,
        )
    except ImportError as e:
        return BenchmarkResult(
            backend="rust_parallel",
            n_samples=n_samples,
            n_trials=n_trials,
            n_threads=n_threads,
            time_seconds=0,
            samples_per_second=0,
            available=False,
            error=str(e),
        )


def benchmark_all(
    n_samples: int = 10000,
    n_trials: int = 100,
    n_threads: int = 0,
    n_runs: int = 3,
    random_state: int = 42,
    include_original: bool = True,
    include_single_threaded: bool = True,
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark all available parallel backends.

    Parameters
    ----------
    n_samples : int
        Number of samples per trial (default: 10000)
    n_trials : int
        Number of trials (default: 100)
    n_threads : int
        Number of threads (0 = auto, default: 0)
    n_runs : int
        Number of runs for averaging (default: 3)
    random_state : int
        Random seed (default: 42)
    include_original : bool
        Include original Cython implementation (default: True)
    include_single_threaded : bool
        Include single-threaded versions (default: True)

    Returns
    -------
    Dict[str, BenchmarkResult]
        Dictionary mapping backend names to benchmark results
    """
    params = _create_test_params(n_trials, random_state)
    results = {}

    print(f"Benchmarking DDM simulators: {n_samples:,} samples × {n_trials:,} trials")
    print(f"Total simulations: {n_samples * n_trials:,}")
    print("-" * 60)

    if include_original:
        print("Benchmarking: cython_original...", end=" ", flush=True)
        results["cython_original"] = benchmark_original_cython(
            params, n_samples, n_trials, 1, n_runs, random_state
        )
        print(f"{'✓' if results['cython_original'].available else '✗'}")

    print("Benchmarking: cython_parallel...", end=" ", flush=True)
    results["cython_parallel"] = benchmark_cython_parallel(
        params, n_samples, n_trials, n_threads, n_runs, random_state
    )
    print(f"{'✓' if results['cython_parallel'].available else '✗'}")

    if include_single_threaded:
        print("Benchmarking: numba_single...", end=" ", flush=True)
        results["numba_single"] = benchmark_numba_single(
            params, n_samples, n_trials, n_runs, random_state
        )
        print(f"{'✓' if results['numba_single'].available else '✗'}")

    print("Benchmarking: numba_parallel...", end=" ", flush=True)
    results["numba_parallel"] = benchmark_numba(
        params, n_samples, n_trials, n_threads, n_runs, random_state
    )
    print(f"{'✓' if results['numba_parallel'].available else '✗'}")

    print("Benchmarking: jax...", end=" ", flush=True)
    results["jax"] = benchmark_jax(params, n_samples, n_trials, n_runs, random_state)
    print(f"{'✓' if results['jax'].available else '✗'}")

    print("Benchmarking: jax_vectorized...", end=" ", flush=True)
    results["jax_vectorized"] = benchmark_jax_vectorized(
        params, n_samples, n_trials, n_runs, random_state
    )
    print(f"{'✓' if results['jax_vectorized'].available else '✗'}")

    print("Benchmarking: rust_parallel...", end=" ", flush=True)
    results["rust_parallel"] = benchmark_rust(
        params, n_samples, n_trials, n_threads, n_runs, random_state
    )
    print(f"{'✓' if results['rust_parallel'].available else '✗'}")

    print("-" * 60)

    return results


def compare_backends(results: Dict[str, BenchmarkResult]) -> str:
    """
    Generate a comparison table from benchmark results.

    Parameters
    ----------
    results : Dict[str, BenchmarkResult]
        Results from benchmark_all()

    Returns
    -------
    str
        Formatted comparison table
    """
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("BENCHMARK RESULTS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(
        f"{'Backend':<25} {'Threads':>8} {'Time (s)':>12} {'Samples/s':>15} {'Speedup':>10}"
    )
    lines.append("-" * 80)

    # Find baseline (original cython or fastest single-threaded)
    baseline_time = None
    for key in ["cython_original", "numba_single"]:
        if key in results and results[key].available:
            baseline_time = results[key].time_seconds
            break

    # Sort by throughput
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if v.available],
        key=lambda x: x[1].samples_per_second,
        reverse=True,
    )

    for name, result in sorted_results:
        speedup = ""
        if baseline_time is not None and result.time_seconds > 0:
            speedup = f"{baseline_time / result.time_seconds:.2f}x"

        lines.append(
            f"{result.backend:<25} "
            f"{result.n_threads:>8} "
            f"{result.time_seconds:>12.4f} "
            f"{result.samples_per_second:>15,.0f} "
            f"{speedup:>10}"
        )

    lines.append("-" * 80)

    # Show unavailable backends
    unavailable = [(k, v) for k, v in results.items() if not v.available]
    if unavailable:
        lines.append("\nUnavailable backends:")
        for name, result in unavailable:
            lines.append(f"  - {name}: {result.error}")

    lines.append("")

    output = "\n".join(lines)
    print(output)
    return output


def main():
    """CLI entry point for benchmarking."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark SSM simulator parallel backends"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of samples per trial (default: 10000)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=100, help="Number of trials (default: 100)"
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=0,
        help="Number of threads (0 = auto, default: 0)",
    )
    parser.add_argument(
        "--n-runs", type=int, default=3, help="Number of benchmark runs (default: 3)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    results = benchmark_all(
        n_samples=args.n_samples,
        n_trials=args.n_trials,
        n_threads=args.n_threads,
        n_runs=args.n_runs,
        random_state=args.seed,
    )

    compare_backends(results)


if __name__ == "__main__":
    main()
