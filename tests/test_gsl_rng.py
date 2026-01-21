"""
Tests for GSL-based random number generation.

These tests validate that GSL's implementations produce statistically
correct distributions (mean, variance, etc.).
"""

import numpy as np
import pytest
from scipy import stats


def test_gsl_available():
    """Test that we can check GSL availability."""
    from cssm._openmp_status import get_openmp_info

    info = get_openmp_info()
    print(f"\nGSL available: {info['gsl_available']}")
    print(f"OpenMP available: {info['openmp_available']}")
    print(f"Parallel ready: {info['parallel_ready']}")

    # This test passes regardless - it's informational
    assert isinstance(info["gsl_available"], bool)


@pytest.mark.skipif(
    not pytest.importorskip("cssm._openmp_status").is_gsl_available(),
    reason="GSL not available",
)
class TestGSLGaussian:
    """Tests for GSL Gaussian (Ziggurat) implementation."""

    def test_gaussian_mean(self):
        """Gaussian samples should have mean close to 0."""
        from cssm._c_rng import generate_gaussian_samples

        samples = generate_gaussian_samples(1_000_000, seed=42)
        mean = np.mean(samples)

        assert abs(mean) < 0.01, f"Mean {mean} too far from 0"

    def test_gaussian_variance(self):
        """Gaussian samples should have variance close to 1.0."""
        from cssm._c_rng import generate_gaussian_samples

        samples = generate_gaussian_samples(1_000_000, seed=42)
        var = np.var(samples)

        # This is the critical test - the broken ziggurat.h had variance ~0.97
        assert 0.99 < var < 1.01, f"Variance {var} not close to 1.0"

    def test_gaussian_skewness(self):
        """Gaussian samples should have skewness close to 0."""
        from cssm._c_rng import generate_gaussian_samples

        samples = generate_gaussian_samples(1_000_000, seed=42)
        skew = stats.skew(samples)

        assert abs(skew) < 0.02, f"Skewness {skew} too far from 0"

    def test_gaussian_kurtosis(self):
        """Gaussian samples should have excess kurtosis close to 0."""
        from cssm._c_rng import generate_gaussian_samples

        samples = generate_gaussian_samples(1_000_000, seed=42)
        kurt = stats.kurtosis(samples)  # Excess kurtosis (normal = 0)

        assert abs(kurt) < 0.05, f"Excess kurtosis {kurt} too far from 0"

    def test_gaussian_ks_test(self):
        """Gaussian samples should pass Kolmogorov-Smirnov test."""
        from cssm._c_rng import generate_gaussian_samples

        samples = generate_gaussian_samples(100_000, seed=42)
        stat, pvalue = stats.kstest(samples, "norm")

        # p-value should be > 0.01 (not significantly different from normal)
        assert pvalue > 0.01, f"KS test failed: stat={stat}, p={pvalue}"

    def test_gaussian_reproducibility(self):
        """Same seed should produce same samples."""
        from cssm._c_rng import generate_gaussian_samples

        samples1 = generate_gaussian_samples(1000, seed=12345)
        samples2 = generate_gaussian_samples(1000, seed=12345)

        np.testing.assert_array_equal(samples1, samples2)

    def test_gaussian_different_seeds(self):
        """Different seeds should produce different samples."""
        from cssm._c_rng import generate_gaussian_samples

        samples1 = generate_gaussian_samples(1000, seed=12345)
        samples2 = generate_gaussian_samples(1000, seed=54321)

        # Should not be equal
        assert not np.allclose(samples1, samples2)


@pytest.mark.skipif(
    not pytest.importorskip("cssm._openmp_status").is_gsl_available(),
    reason="GSL not available",
)
class TestGSLLevy:
    """Tests for GSL Levy alpha-stable implementation."""

    def test_levy_alpha2_is_gaussian(self):
        """Levy with alpha=2 should produce Gaussian with variance=2."""
        from cssm._c_rng import generate_levy_samples

        samples = generate_levy_samples(1_000_000, alpha=2.0, seed=42)
        var = np.var(samples)

        # alpha=2 gives Gaussian with variance = 2
        assert 1.95 < var < 2.05, f"Levy(alpha=2) variance {var} not close to 2"

    def test_levy_alpha1_is_cauchy(self):
        """Levy with alpha=1 should produce Cauchy (heavy tails)."""
        from cssm._c_rng import generate_levy_samples

        samples = generate_levy_samples(100_000, alpha=1.0, seed=42)

        # Cauchy has undefined variance, but median should be ~0
        median = np.median(samples)
        assert abs(median) < 0.1, f"Levy(alpha=1) median {median} not close to 0"

        # Should have heavy tails (some extreme values)
        assert np.max(np.abs(samples)) > 100, "Levy(alpha=1) should have heavy tails"


@pytest.mark.skipif(
    not pytest.importorskip("cssm._openmp_status").is_gsl_available(),
    reason="GSL not available",
)
class TestGSLUniform:
    """Tests for GSL uniform RNG."""

    def test_uniform_range(self):
        """Uniform samples should be in (0, 1)."""
        from cssm._c_rng import generate_uniform_samples

        samples = generate_uniform_samples(100_000, seed=42)

        assert np.all(samples > 0), "Uniform should be > 0"
        assert np.all(samples < 1), "Uniform should be < 1"

    def test_uniform_mean(self):
        """Uniform samples should have mean close to 0.5."""
        from cssm._c_rng import generate_uniform_samples

        samples = generate_uniform_samples(1_000_000, seed=42)
        mean = np.mean(samples)

        assert 0.499 < mean < 0.501, f"Uniform mean {mean} not close to 0.5"

    def test_uniform_variance(self):
        """Uniform samples should have variance close to 1/12."""
        from cssm._c_rng import generate_uniform_samples

        samples = generate_uniform_samples(1_000_000, seed=42)
        var = np.var(samples)
        expected_var = 1.0 / 12.0  # ~0.0833

        assert abs(var - expected_var) < 0.001, (
            f"Uniform variance {var} not close to {expected_var}"
        )


class TestParallelRequestValidation:
    """Tests for check_parallel_request function."""

    def test_single_thread_always_works(self):
        """n_threads=1 should always return 1."""
        from cssm._openmp_status import check_parallel_request

        result = check_parallel_request(1, warn=False)
        assert result == 1

    def test_parallel_request_returns_correct_value(self):
        """check_parallel_request should return appropriate value."""
        from cssm._openmp_status import check_parallel_request, get_openmp_info

        info = get_openmp_info()
        result = check_parallel_request(4, warn=False)

        if info["parallel_ready"]:
            # Both OpenMP and GSL available
            assert result == 4
        else:
            # Falls back to single-threaded
            assert result == 1

    def test_parallel_request_warns_when_unavailable(self):
        """check_parallel_request should warn when parallel not available."""
        from cssm._openmp_status import check_parallel_request, get_openmp_info
        import warnings

        info = get_openmp_info()

        if not info["parallel_ready"]:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                check_parallel_request(4, warn=True)
                assert len(w) == 1
                assert "n_threads=4" in str(w[0].message)


def test_numpy_gaussian_variance_reference():
    """Reference test: NumPy's Gaussian should have variance ~1.0."""
    rng = np.random.default_rng(42)
    samples = rng.standard_normal(1_000_000)
    var = np.var(samples)

    assert 0.999 < var < 1.001, f"NumPy Gaussian variance {var} not close to 1.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
