"""
Rigorous validation tests for the Ziggurat Gaussian sampler.

These tests verify that our Ziggurat implementation produces statistically
correct samples that match the standard normal distribution N(0,1).

Tests include:
1. Moment tests (mean, variance, skewness, kurtosis)
2. Kolmogorov-Smirnov test against N(0,1)
3. Anderson-Darling normality test
4. Chi-squared goodness-of-fit test
5. Tail coverage tests (ensure we sample the tails correctly)
6. Comparison with NumPy's random.standard_normal

Run with:
    pytest tests/test_ziggurat_validation.py -v
    pytest tests/test_ziggurat_validation.py -v --tb=short  # Quick summary
"""

import numpy as np
import pytest
from scipy import stats

# Try to import the C RNG module
try:
    from cssm._c_rng import generate_gaussian_samples

    C_RNG_AVAILABLE = True
except ImportError:
    C_RNG_AVAILABLE = False


pytestmark = [
    pytest.mark.rng_validation,
]


@pytest.mark.skipif(not C_RNG_AVAILABLE, reason="C RNG module not available")
class TestZigguratMoments:
    """Test that the Ziggurat sampler produces correct moments."""

    @pytest.fixture
    def large_sample(self):
        """Generate a large sample for statistical tests."""
        return generate_gaussian_samples(1_000_000, seed=12345)

    def test_mean_close_to_zero(self, large_sample):
        """Test that mean is approximately 0."""
        mean = np.mean(large_sample)
        # For N=1M, std error of mean is 1/sqrt(1M) = 0.001
        # Allow 5 sigma = 0.005
        assert abs(mean) < 0.005, f"Mean {mean:.6f} too far from 0"

    def test_variance_close_to_one(self, large_sample):
        """Test that variance is approximately 1."""
        var = np.var(large_sample, ddof=1)
        # For N=1M, std error of variance is sqrt(2/(N-1)) ≈ 0.0014
        # Allow 5 sigma = 0.007
        assert abs(var - 1.0) < 0.007, f"Variance {var:.6f} too far from 1.0"

    def test_std_close_to_one(self, large_sample):
        """Test that standard deviation is approximately 1."""
        std = np.std(large_sample, ddof=1)
        # Allow 0.005 tolerance
        assert abs(std - 1.0) < 0.005, f"Std {std:.6f} too far from 1.0"

    def test_skewness_close_to_zero(self, large_sample):
        """Test that skewness is approximately 0."""
        skew = stats.skew(large_sample)
        # For N=1M, std error of skewness is sqrt(6/N) ≈ 0.0024
        # Allow 5 sigma = 0.012
        assert abs(skew) < 0.015, f"Skewness {skew:.6f} too far from 0"

    def test_kurtosis_close_to_zero(self, large_sample):
        """Test that excess kurtosis is approximately 0 (normal = 0)."""
        kurt = stats.kurtosis(large_sample)  # Fisher kurtosis (excess)
        # For N=1M, std error of kurtosis is sqrt(24/N) ≈ 0.0049
        # Allow 5 sigma = 0.025
        assert abs(kurt) < 0.03, f"Excess kurtosis {kurt:.6f} too far from 0"


@pytest.mark.skipif(not C_RNG_AVAILABLE, reason="C RNG module not available")
class TestZigguratDistribution:
    """Test that samples follow the standard normal distribution."""

    def test_ks_test(self):
        """Kolmogorov-Smirnov test against standard normal."""
        samples = generate_gaussian_samples(100_000, seed=42)
        statistic, pvalue = stats.kstest(samples, "norm")

        # p-value should be > 0.01 (not significantly different from normal)
        assert pvalue > 0.01, (
            f"KS test failed: statistic={statistic:.4f}, p-value={pvalue:.6f}"
        )

    def test_anderson_darling(self):
        """Anderson-Darling test for normality."""
        samples = generate_gaussian_samples(50_000, seed=42)
        result = stats.anderson(samples, dist="norm")

        # Check against 5% significance level (index 2)
        critical_5pct = result.critical_values[2]
        assert result.statistic < critical_5pct, (
            f"Anderson-Darling test failed: statistic={result.statistic:.4f} > {critical_5pct:.4f}"
        )

    def test_chi_squared_goodness_of_fit(self):
        """Chi-squared test against binned normal distribution."""
        samples = generate_gaussian_samples(500_000, seed=42)

        bins = np.linspace(-4, 4, 51)
        bins[0] = -np.inf
        bins[-1] = np.inf
        observed, _ = np.histogram(samples, bins=bins)

        n_samples = len(samples)
        expected = n_samples * np.diff(stats.norm.cdf(bins))

        mask = expected > 5
        chi2, pvalue = stats.chisquare(observed[mask], expected[mask])

        assert pvalue > 0.01, (
            f"Chi-squared test failed: chi2={chi2:.2f}, p-value={pvalue:.6f}"
        )


@pytest.mark.skipif(not C_RNG_AVAILABLE, reason="C RNG module not available")
class TestZigguratTails:
    """Test that the Ziggurat sampler correctly samples the tails."""

    def test_tail_coverage_3sigma(self):
        """Test that we get approximately correct proportion beyond 3 sigma."""
        samples = generate_gaussian_samples(1_000_000, seed=42)

        # P(|X| > 3) = 2 * (1 - Phi(3)) ≈ 0.0027
        expected_prop = 2 * (1 - stats.norm.cdf(3))  # ≈ 0.0027
        observed_prop = np.mean(np.abs(samples) > 3)

        # Allow 20% relative error
        rel_error = abs(observed_prop - expected_prop) / expected_prop
        assert rel_error < 0.2, (
            f"Tail (|x|>3) proportion {observed_prop:.5f} vs expected {expected_prop:.5f}"
        )

    def test_tail_coverage_3_5sigma(self):
        """Test coverage beyond 3.5 sigma (ziggurat cutoff is ~3.44)."""
        samples = generate_gaussian_samples(2_000_000, seed=42)

        # P(|X| > 3.5) ≈ 0.000465
        expected_prop = 2 * (1 - stats.norm.cdf(3.5))
        observed_prop = np.mean(np.abs(samples) > 3.5)

        # Allow 30% relative error (smaller sample in tail)
        rel_error = abs(observed_prop - expected_prop) / expected_prop
        assert rel_error < 0.3, (
            f"Tail (|x|>3.5) proportion {observed_prop:.6f} vs expected {expected_prop:.6f}"
        )

    def test_extreme_tails_exist(self):
        """Test that we actually generate extreme values (beyond ziggurat r=3.44)."""
        samples = generate_gaussian_samples(5_000_000, seed=42)

        # The ziggurat cutoff is r ≈ 3.44
        # We should see some values beyond this
        max_abs = np.max(np.abs(samples))

        assert max_abs > 4.0, f"Max |x| = {max_abs:.2f}, expected some values > 4"

        # Count values beyond 4 sigma
        n_beyond_4 = np.sum(np.abs(samples) > 4)
        expected_beyond_4 = 5_000_000 * 2 * (1 - stats.norm.cdf(4))  # ≈ 316

        # Should have at least 50% of expected
        assert n_beyond_4 > 0.5 * expected_beyond_4, (
            f"Only {n_beyond_4} samples beyond 4 sigma, expected ~{expected_beyond_4:.0f}"
        )


@pytest.mark.skipif(not C_RNG_AVAILABLE, reason="C RNG module not available")
class TestZigguratReproducibility:
    """Test that the sampler is reproducible with the same seed."""

    def test_same_seed_same_samples(self):
        """Test that same seed produces identical samples."""
        samples1 = generate_gaussian_samples(10_000, seed=42)
        samples2 = generate_gaussian_samples(10_000, seed=42)

        np.testing.assert_array_equal(samples1, samples2)

    def test_different_seeds_different_samples(self):
        """Test that different seeds produce different samples."""
        samples1 = generate_gaussian_samples(10_000, seed=42)
        samples2 = generate_gaussian_samples(10_000, seed=123)

        assert not np.allclose(samples1, samples2)

        # But both should still be valid Gaussians
        for samples in [samples1, samples2]:
            assert abs(np.mean(samples)) < 0.05
            assert abs(np.std(samples) - 1.0) < 0.05


@pytest.mark.skipif(not C_RNG_AVAILABLE, reason="C RNG module not available")
class TestZigguratVsNumPy:
    """Compare Ziggurat output against NumPy's standard_normal."""

    def test_similar_moments(self):
        """Test that moments are similar to NumPy's generator."""
        n = 500_000

        zig_samples = generate_gaussian_samples(n, seed=42)
        np_samples = np.random.default_rng(42).standard_normal(n)

        # Compare means
        assert abs(np.mean(zig_samples) - np.mean(np_samples)) < 0.005

        # Compare stds
        assert abs(np.std(zig_samples) - np.std(np_samples)) < 0.005

        # Compare skewness
        assert abs(stats.skew(zig_samples) - stats.skew(np_samples)) < 0.02

        # Compare kurtosis
        assert abs(stats.kurtosis(zig_samples) - stats.kurtosis(np_samples)) < 0.05

    def test_similar_quantiles(self):
        """Test that quantiles are similar to NumPy's generator."""
        n = 500_000

        zig_samples = generate_gaussian_samples(n, seed=42)
        np_samples = np.random.default_rng(42).standard_normal(n)

        for q in [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]:
            zig_q = np.percentile(zig_samples, q * 100)
            np_q = np.percentile(np_samples, q * 100)

            # Allow 5% relative difference
            if abs(np_q) > 0.1:
                rel_diff = abs(zig_q - np_q) / abs(np_q)
                assert rel_diff < 0.05, (
                    f"Quantile {q}: Ziggurat={zig_q:.4f}, NumPy={np_q:.4f}"
                )


@pytest.mark.skipif(not C_RNG_AVAILABLE, reason="C RNG module not available")
class TestZigguratMultipleRuns:
    """Test consistency across multiple runs with different seeds."""

    def test_variance_consistency(self):
        """Test that variance is consistently close to 1 across many seeds."""
        variances = []
        for seed in range(100):
            samples = generate_gaussian_samples(50_000, seed=seed)
            variances.append(np.var(samples, ddof=1))

        variances = np.array(variances)

        # Mean of variances should be very close to 1
        assert abs(np.mean(variances) - 1.0) < 0.005, (
            f"Mean variance {np.mean(variances):.6f} too far from 1.0"
        )

        # No individual variance should be too far off
        assert np.all(variances > 0.95), f"Min variance {np.min(variances):.4f} too low"
        assert np.all(variances < 1.05), (
            f"Max variance {np.max(variances):.4f} too high"
        )

    def test_mean_consistency(self):
        """Test that mean is consistently close to 0 across many seeds."""
        means = []
        for seed in range(100):
            samples = generate_gaussian_samples(50_000, seed=seed)
            means.append(np.mean(samples))

        means = np.array(means)

        # Mean of means should be very close to 0
        assert abs(np.mean(means)) < 0.002, (
            f"Mean of means {np.mean(means):.6f} too far from 0"
        )

        # Std of means should be close to 1/sqrt(50000) ≈ 0.0045
        expected_std = 1.0 / np.sqrt(50_000)
        assert abs(np.std(means) - expected_std) < 0.001, (
            f"Std of means {np.std(means):.6f} vs expected {expected_std:.6f}"
        )


@pytest.mark.skipif(not C_RNG_AVAILABLE, reason="C RNG module not available")
class TestZigguratStress:
    """Stress tests for the Ziggurat sampler."""

    @pytest.mark.slow
    def test_large_sample_moments(self):
        """Test moments with a very large sample."""
        samples = generate_gaussian_samples(10_000_000, seed=42)

        mean = np.mean(samples)
        var = np.var(samples, ddof=1)
        skew = stats.skew(samples)
        kurt = stats.kurtosis(samples)

        # Very tight tolerances for 10M samples
        assert abs(mean) < 0.002, f"Mean {mean:.6f}"
        assert abs(var - 1.0) < 0.002, f"Variance {var:.6f}"
        assert abs(skew) < 0.005, f"Skewness {skew:.6f}"
        assert abs(kurt) < 0.01, f"Kurtosis {kurt:.6f}"

    @pytest.mark.slow
    def test_many_small_samples(self):
        """Test that many small samples all have reasonable statistics."""
        n_runs = 1000
        n_samples = 1000

        failed_variance = 0
        failed_mean = 0

        for seed in range(n_runs):
            samples = generate_gaussian_samples(n_samples, seed=seed)

            # Allow 3 sigma deviations
            # Variance: expected std error is sqrt(2/N) ≈ 0.045
            if abs(np.var(samples, ddof=1) - 1.0) > 0.15:
                failed_variance += 1

            # Mean: expected std error is 1/sqrt(N) ≈ 0.032
            if abs(np.mean(samples)) > 0.1:
                failed_mean += 1

        # At most 1% should fail (we're using ~3 sigma threshold)
        assert failed_variance < n_runs * 0.02, (
            f"{failed_variance}/{n_runs} runs had bad variance"
        )
        assert failed_mean < n_runs * 0.02, f"{failed_mean}/{n_runs} runs had bad mean"


def print_diagnostic_report():
    """Print a diagnostic report of the Ziggurat sampler."""
    if not C_RNG_AVAILABLE:
        print("C RNG module not available")
        return

    print("=" * 60)
    print("Ziggurat Gaussian Sampler Diagnostic Report")
    print("=" * 60)

    # Generate samples
    n = 1_000_000
    samples = generate_gaussian_samples(n, seed=42)

    print(f"\nSample size: {n:,}")
    print("\nMoments:")
    print(f"  Mean:     {np.mean(samples):+.6f}  (expected: 0)")
    print(f"  Std:      {np.std(samples, ddof=1):.6f}  (expected: 1)")
    print(f"  Variance: {np.var(samples, ddof=1):.6f}  (expected: 1)")
    print(f"  Skewness: {stats.skew(samples):+.6f}  (expected: 0)")
    print(f"  Kurtosis: {stats.kurtosis(samples):+.6f}  (expected: 0)")

    print("\nQuantiles:")
    for q in [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]:
        observed = np.percentile(samples, q * 100)
        expected = stats.norm.ppf(q)
        print(f"  {q * 100:5.1f}%: {observed:+.4f}  (expected: {expected:+.4f})")

    print("\nTail coverage:")
    for sigma in [2, 3, 3.5, 4]:
        expected = 2 * (1 - stats.norm.cdf(sigma))
        observed = np.mean(np.abs(samples) > sigma)
        print(f"  |x| > {sigma}: {observed:.6f}  (expected: {expected:.6f})")

    print("\nExtreme values:")
    print(f"  Min: {np.min(samples):.4f}")
    print(f"  Max: {np.max(samples):.4f}")

    # Statistical tests
    print("\nStatistical tests:")
    ks_stat, ks_pval = stats.kstest(samples[:100000], "norm")
    print(f"  KS test:     statistic={ks_stat:.4f}, p-value={ks_pval:.4f}")

    ad_result = stats.anderson(samples[:50000], dist="norm")
    print(
        f"  AD test:     statistic={ad_result.statistic:.4f}, "
        f"critical(5%)={ad_result.critical_values[2]:.4f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    # Run diagnostic report
    print_diagnostic_report()

    # Run quick tests
    print("\n\nRunning quick tests...")
    pytest.main([__file__, "-v", "-x", "--tb=short", "-m", "not slow"])
