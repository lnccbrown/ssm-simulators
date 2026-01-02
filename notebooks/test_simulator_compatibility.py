"""
Tutorial: Testing Function-Based vs. Class-Based Simulator Compatibility

This script tests all available models to ensure that:
1. The class-based Simulator produces equivalent results to the function-based simulator()
2. The class-based interface is solid and production-ready
3. An adapter can seamlessly bridge the two APIs

Run this before transitioning TrainingDataGenerator to use the class-based Simulator.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
import warnings

from ssms.basic_simulators.simulator import simulator as function_simulator
from ssms.basic_simulators.simulator_class import Simulator
from ssms.config import model_config


# ============================================================================
# Part 1: Adapter Function
# ============================================================================

def simulator_to_function(simulator_instance: Simulator):
    """Convert a Simulator instance to function-based API.

    This adapter allows a class-based Simulator to be used wherever
    the function-based simulator() API is expected (e.g., in TrainingDataGenerator).

    Args:
        simulator_instance: Configured Simulator instance

    Returns:
        Function with signature matching simulator()

    Example:
        >>> sim = Simulator("ddm")
        >>> sim_func = simulator_to_function(sim)
        >>> results = sim_func(
        ...     theta={'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3},
        ...     model="ddm",  # Ignored (model already set in Simulator)
        ...     n_samples=1000,
        ...     delta_t=0.001,
        ...     max_t=20,
        ...     random_state=42
        ... )
    """
    def wrapped_simulator(
        theta: dict,
        model: str,  # Ignored - model already configured in Simulator
        n_samples: int = 1000,
        delta_t: float = 0.001,
        max_t: float = 20,
        random_state: int | None = None,
        **kwargs  # Absorb any extra arguments
    ) -> dict:
        """Function-based API wrapper around class-based Simulator."""
        return simulator_instance.simulate(
            theta=theta,
            n_samples=n_samples,
            delta_t=delta_t,
            max_t=max_t,
            random_state=random_state,
        )

    return wrapped_simulator


# ============================================================================
# Part 2: Comparison Utilities
# ============================================================================

def compare_simulation_results(
    func_result: dict,
    class_result: dict,
    model_name: str,
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> tuple[bool, list[str]]:
    """Compare results from function-based and class-based simulators.

    Args:
        func_result: Results from function-based simulator()
        class_result: Results from class-based Simulator.simulate()
        model_name: Name of model being tested
        rtol: Relative tolerance for numerical comparisons
        atol: Absolute tolerance for numerical comparisons

    Returns:
        Tuple of (all_match: bool, differences: list[str])
    """
    differences = []

    # Check that both have the same keys
    func_keys = set(func_result.keys())
    class_keys = set(class_result.keys())

    if func_keys != class_keys:
        differences.append(
            f"Key mismatch: func has {func_keys - class_keys} extra, "
            f"class has {class_keys - func_keys} extra"
        )

    # Compare common keys
    for key in func_keys & class_keys:
        func_val = func_result[key]
        class_val = class_result[key]

        # Handle numpy arrays
        if isinstance(func_val, np.ndarray) and isinstance(class_val, np.ndarray):
            if func_val.shape != class_val.shape:
                differences.append(
                    f"{key}: shape mismatch - func {func_val.shape} vs class {class_val.shape}"
                )
            elif not np.allclose(func_val, class_val, rtol=rtol, atol=atol, equal_nan=True):
                max_diff = np.max(np.abs(func_val - class_val))
                differences.append(
                    f"{key}: values differ (max diff: {max_diff:.6e})"
                )

        # Handle dicts (like metadata)
        elif isinstance(func_val, dict) and isinstance(class_val, dict):
            # Recursively check important metadata fields
            for subkey in ['model', 'n_samples', 'possible_choices']:
                if subkey in func_val and subkey in class_val:
                    if func_val[subkey] != class_val[subkey]:
                        differences.append(
                            f"{key}.{subkey}: {func_val[subkey]} vs {class_val[subkey]}"
                        )

        # Handle scalars
        elif isinstance(func_val, (int, float, np.number)) and isinstance(class_val, (int, float, np.number)):
            if not np.isclose(func_val, class_val, rtol=rtol, atol=atol):
                differences.append(
                    f"{key}: {func_val} vs {class_val}"
                )

    return len(differences) == 0, differences


def get_default_theta(model_name: str, config: dict) -> dict:
    """Get default theta values for a model.

    Args:
        model_name: Name of the model
        config: Model configuration dictionary

    Returns:
        Dictionary of parameter values
    """
    params = config['params']

    # Use default_params if available
    if 'default_params' in config:
        return dict(zip(params, config['default_params']))

    # Otherwise, use midpoint of param_bounds
    if 'param_bounds' in config:
        lower = config['param_bounds'][0]
        upper = config['param_bounds'][1]
        midpoints = [(l + u) / 2 for l, u in zip(lower, upper)]
        return dict(zip(params, midpoints))

    # Fallback: generic defaults
    defaults = {
        'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3,  # DDM-like
        'A': 0.5, 'b': 1.0, 'v0': 0.5, 'v1': 0.6,  # LBA-like
        'theta': 0.5, 's': 0.3,  # Others
    }

    theta = {}
    for param in params:
        if param in defaults:
            theta[param] = defaults[param]
        else:
            theta[param] = 0.5  # Generic fallback

    return theta


# ============================================================================
# Part 3: Comprehensive Model Testing
# ============================================================================

def test_all_models(
    n_samples: int = 100,
    delta_t: float = 0.001,
    max_t: float = 20,
    random_seed: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """Test all available models for function vs. class compatibility.

    Args:
        n_samples: Number of simulation samples
        delta_t: Time step size
        max_t: Maximum simulation time
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress

    Returns:
        DataFrame with test results for each model
    """
    results = []

    # Get all available models
    available_models = list(model_config.keys())

    if verbose:
        print(f"Testing {len(available_models)} models...")
        print("=" * 80)

    for model_name in available_models:
        if verbose:
            print(f"\nTesting {model_name}...", end=" ")

        try:
            # Get model config
            config = model_config[model_name]

            # Get default parameters
            theta = get_default_theta(model_name, config)

            # Run function-based simulator
            func_result = function_simulator(
                theta=theta.copy(),
                model=model_name,
                n_samples=n_samples,
                delta_t=delta_t,
                max_t=max_t,
                random_state=random_seed,
            )

            # Run class-based simulator
            sim = Simulator(model_name)
            class_result = sim.simulate(
                theta=theta.copy(),
                n_samples=n_samples,
                delta_t=delta_t,
                max_t=max_t,
                random_state=random_seed,
            )

            # Compare results
            match, differences = compare_simulation_results(
                func_result, class_result, model_name
            )

            # Store result
            result = {
                'model': model_name,
                'status': 'PASS' if match else 'FAIL',
                'match': match,
                'differences': '; '.join(differences) if differences else None,
                'n_params': len(theta),
                'theta_keys': ', '.join(theta.keys()),
            }
            results.append(result)

            if verbose:
                print(f"✓ PASS" if match else f"✗ FAIL")
                if differences and verbose:
                    for diff in differences:
                        print(f"  └─ {diff}")

        except Exception as e:
            if verbose:
                print(f"✗ ERROR: {str(e)}")
            results.append({
                'model': model_name,
                'status': 'ERROR',
                'match': False,
                'differences': str(e),
                'n_params': None,
                'theta_keys': None,
            })

    if verbose:
        print("\n" + "=" * 80)
        print("\nSummary:")
        df = pd.DataFrame(results)
        print(f"  Total models: {len(df)}")
        print(f"  Passed: {(df['status'] == 'PASS').sum()}")
        print(f"  Failed: {(df['status'] == 'FAIL').sum()}")
        print(f"  Errors: {(df['status'] == 'ERROR').sum()}")

    return pd.DataFrame(results)


# ============================================================================
# Part 4: Adapter Testing
# ============================================================================

def test_adapter_function(
    model_name: str = "ddm",
    n_samples: int = 100,
    random_seed: int = 42,
    verbose: bool = True
) -> bool:
    """Test that the adapter function works correctly.

    Args:
        model_name: Model to test
        n_samples: Number of simulation samples
        random_seed: Random seed
        verbose: Whether to print results

    Returns:
        True if adapter works correctly
    """
    if verbose:
        print(f"\nTesting adapter function with {model_name}...")

    # Get default theta
    config = model_config[model_name]
    theta = get_default_theta(model_name, config)

    # Create class-based simulator
    sim = Simulator(model_name)

    # Wrap it with adapter
    sim_func = simulator_to_function(sim)

    # Call via adapter (function-based API)
    adapter_result = sim_func(
        theta=theta.copy(),
        model=model_name,  # This gets ignored
        n_samples=n_samples,
        random_state=random_seed,
    )

    # Call directly (class-based API)
    direct_result = sim.simulate(
        theta=theta.copy(),
        n_samples=n_samples,
        random_state=random_seed,
    )

    # Compare
    match, differences = compare_simulation_results(
        adapter_result, direct_result, model_name
    )

    if verbose:
        if match:
            print(f"  ✓ Adapter works correctly!")
        else:
            print(f"  ✗ Adapter produced different results:")
            for diff in differences:
                print(f"    └─ {diff}")

    return match


# ============================================================================
# Part 5: Performance Comparison
# ============================================================================

def benchmark_performance(
    model_name: str = "ddm",
    n_samples: int = 1000,
    n_iterations: int = 10,
    verbose: bool = True
) -> dict:
    """Benchmark performance of function vs. class-based simulators.

    Args:
        model_name: Model to benchmark
        n_samples: Number of samples per simulation
        n_iterations: Number of iterations to average
        verbose: Whether to print results

    Returns:
        Dictionary with timing results
    """
    import time

    if verbose:
        print(f"\nBenchmarking {model_name} ({n_iterations} iterations)...")

    config = model_config[model_name]
    theta = get_default_theta(model_name, config)

    # Benchmark function-based
    func_times = []
    for i in range(n_iterations):
        start = time.perf_counter()
        function_simulator(
            theta=theta.copy(),
            model=model_name,
            n_samples=n_samples,
            random_state=i,
        )
        func_times.append(time.perf_counter() - start)

    # Benchmark class-based
    sim = Simulator(model_name)
    class_times = []
    for i in range(n_iterations):
        start = time.perf_counter()
        sim.simulate(
            theta=theta.copy(),
            n_samples=n_samples,
            random_state=i,
        )
        class_times.append(time.perf_counter() - start)

    results = {
        'model': model_name,
        'function_mean_ms': np.mean(func_times) * 1000,
        'function_std_ms': np.std(func_times) * 1000,
        'class_mean_ms': np.mean(class_times) * 1000,
        'class_std_ms': np.std(class_times) * 1000,
        'overhead_percent': (np.mean(class_times) - np.mean(func_times)) / np.mean(func_times) * 100,
    }

    if verbose:
        print(f"  Function-based: {results['function_mean_ms']:.2f} ± {results['function_std_ms']:.2f} ms")
        print(f"  Class-based:    {results['class_mean_ms']:.2f} ± {results['class_std_ms']:.2f} ms")
        print(f"  Overhead:       {results['overhead_percent']:.1f}%")

    return results


# ============================================================================
# Part 6: Main Testing Script
# ============================================================================

def run_full_test_suite():
    """Run the complete test suite."""
    print("=" * 80)
    print("SIMULATOR COMPATIBILITY TEST SUITE")
    print("=" * 80)

    # Test 1: All models compatibility
    print("\n" + "=" * 80)
    print("TEST 1: Function vs. Class Compatibility for All Models")
    print("=" * 80)
    df_results = test_all_models(n_samples=100, verbose=True)

    # Test 2: Adapter function
    print("\n" + "=" * 80)
    print("TEST 2: Adapter Function Correctness")
    print("=" * 80)
    adapter_works = test_adapter_function(model_name="ddm", verbose=True)

    # Test 3: Performance benchmark
    print("\n" + "=" * 80)
    print("TEST 3: Performance Comparison")
    print("=" * 80)
    perf_ddm = benchmark_performance(model_name="ddm", n_iterations=10, verbose=True)
    perf_angle = benchmark_performance(model_name="angle", n_iterations=10, verbose=True)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = (df_results['status'] == 'PASS').sum()
    total = len(df_results)
    pass_rate = passed / total * 100

    print(f"\nCompatibility Test:")
    print(f"  ✓ {passed}/{total} models passed ({pass_rate:.1f}%)")

    if adapter_works:
        print(f"\nAdapter Function:")
        print(f"  ✓ Working correctly")
    else:
        print(f"\nAdapter Function:")
        print(f"  ✗ Issues detected")

    print(f"\nPerformance:")
    print(f"  DDM overhead:   {perf_ddm['overhead_percent']:.1f}%")
    print(f"  Angle overhead: {perf_angle['overhead_percent']:.1f}%")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if pass_rate >= 95 and adapter_works and abs(perf_ddm['overhead_percent']) < 10:
        print("\n✓ Class-based Simulator is PRODUCTION READY")
        print("  - High compatibility with function-based API")
        print("  - Adapter function works correctly")
        print("  - Performance overhead is acceptable")
        print("\n  → Safe to transition TrainingDataGenerator to class-based Simulator")
    else:
        print("\n⚠ Additional work needed before transition:")
        if pass_rate < 95:
            print(f"  - Fix compatibility issues ({total - passed} models failing)")
        if not adapter_works:
            print(f"  - Fix adapter function")
        if abs(perf_ddm['overhead_percent']) >= 10:
            print(f"  - Address performance overhead ({perf_ddm['overhead_percent']:.1f}%)")

    print("\n" + "=" * 80)

    return df_results


# ============================================================================
# Run if executed as script
# ============================================================================

if __name__ == "__main__":
    results_df = run_full_test_suite()

    # Save results
    results_df.to_csv("simulator_compatibility_results.csv", index=False)
    print("\nResults saved to: simulator_compatibility_results.csv")
