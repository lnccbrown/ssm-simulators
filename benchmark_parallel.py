#!/usr/bin/env python
"""
Benchmark script for SSMS parallel execution with GSL RNG.
"""

import numpy as np
import time
from ssms.basic_simulators import simulator
from cssm._openmp_status import get_openmp_info


def benchmark_model(model, theta, n_samples, n_threads_list, n_runs=3):
    """Benchmark a model with different thread counts."""
    results = {}

    for n_threads in n_threads_list:
        times = []
        for i in range(n_runs):
            start = time.perf_counter()
            _ = simulator.simulator(
                model=model,
                theta=theta,
                n_samples=n_samples,
                random_state=42 + i,
                n_threads=n_threads,
            )
            times.append(time.perf_counter() - start)
        results[n_threads] = (np.mean(times), np.std(times))

    return results


def main():
    info = get_openmp_info()

    print("=" * 70)
    print("SSMS PARALLEL PERFORMANCE BENCHMARK")
    print("=" * 70)
    print("\nParallel Status:")
    print(f"  OpenMP: {'Yes' if info['openmp_available'] else 'No'}")
    print(f"  GSL:    {'Yes' if info['gsl_available'] else 'No'}")
    print(f"  Ready:  {'Yes' if info['parallel_ready'] else 'No'}")
    print(f"  Max threads: {info['max_threads']}")

    # Test configurations
    theta_ddm = {"v": 0.5, "a": 1.0, "z": 0.5, "t": 0.3}
    theta_levy = {"v": 0.5, "a": 1.0, "z": 0.5, "t": 0.3, "alpha": 1.5}

    n_threads_list = [1, 2, 4, 8]
    n_threads_list = [t for t in n_threads_list if t <= info["max_threads"]]

    # DDM Benchmark - 50K samples
    print("\n" + "-" * 70)
    print("DDM Model (n_samples=50,000)")
    print("-" * 70)

    results = benchmark_model("ddm", theta_ddm, 50000, n_threads_list, n_runs=3)
    base_time = results[1][0]

    for n_threads, (mean_t, std_t) in results.items():
        rng_type = "NumPy" if n_threads == 1 else "GSL"
        speedup = base_time / mean_t
        if n_threads == 1:
            print(
                f"  n_threads={n_threads} ({rng_type}):   {mean_t:.4f}s +/- {std_t:.4f}s"
            )
        else:
            print(
                f"  n_threads={n_threads} ({rng_type}):     {mean_t:.4f}s +/- {std_t:.4f}s  (speedup: {speedup:.2f}x)"
            )

    # DDM Benchmark - 200K samples
    print("\n" + "-" * 70)
    print("DDM Model (n_samples=200,000)")
    print("-" * 70)

    results = benchmark_model("ddm", theta_ddm, 200000, n_threads_list, n_runs=3)
    base_time = results[1][0]

    for n_threads, (mean_t, std_t) in results.items():
        rng_type = "NumPy" if n_threads == 1 else "GSL"
        speedup = base_time / mean_t
        if n_threads == 1:
            print(
                f"  n_threads={n_threads} ({rng_type}):   {mean_t:.4f}s +/- {std_t:.4f}s"
            )
        else:
            print(
                f"  n_threads={n_threads} ({rng_type}):     {mean_t:.4f}s +/- {std_t:.4f}s  (speedup: {speedup:.2f}x)"
            )

    # Levy Benchmark
    print("\n" + "-" * 70)
    print("Levy Model (n_samples=50,000)")
    print("-" * 70)

    results = benchmark_model("levy", theta_levy, 50000, n_threads_list, n_runs=3)
    base_time = results[1][0]

    for n_threads, (mean_t, std_t) in results.items():
        rng_type = "NumPy" if n_threads == 1 else "GSL"
        speedup = base_time / mean_t
        if n_threads == 1:
            print(
                f"  n_threads={n_threads} ({rng_type}):   {mean_t:.4f}s +/- {std_t:.4f}s"
            )
        else:
            print(
                f"  n_threads={n_threads} ({rng_type}):     {mean_t:.4f}s +/- {std_t:.4f}s  (speedup: {speedup:.2f}x)"
            )

    # Statistical correctness check
    print("\n" + "-" * 70)
    print("Statistical Correctness Check")
    print("-" * 70)

    result_1 = simulator.simulator(
        model="ddm", theta=theta_ddm, n_samples=100000, random_state=42, n_threads=1
    )

    result_4 = simulator.simulator(
        model="ddm", theta=theta_ddm, n_samples=100000, random_state=42, n_threads=4
    )

    rts_1 = result_1["rts"][result_1["rts"] > 0].flatten()
    rts_4 = result_4["rts"][result_4["rts"] > 0].flatten()

    print(f"  n_threads=1: mean_rt={np.mean(rts_1):.4f}, std_rt={np.std(rts_1):.4f}")
    print(f"  n_threads=4: mean_rt={np.mean(rts_4):.4f}, std_rt={np.std(rts_4):.4f}")

    # KS test
    from scipy import stats

    ks_stat, p_value = stats.ks_2samp(rts_1, rts_4)
    status = "PASS" if p_value > 0.01 else "FAIL"
    print(f"  KS test p-value: {p_value:.4f} [{status}]")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
