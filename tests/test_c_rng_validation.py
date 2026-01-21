"""
Validation tests for C-level random number generators.

These tests verify that our custom C implementations (Ziggurat for Gaussian,
CMS for alpha-stable) produce statistically correct distributions compared
to reference implementations (NumPy/SciPy).

These tests are marked with `pytest.mark.rng_validation` so they can be
optionally run separately.

Run with:
    pytest tests/test_c_rng_validation.py -v
    pytest tests/test_c_rng_validation.py -v -m "not slow"  # Skip slow tests
"""

import numpy as np
import pytest
from scipy import stats

# Try to import the C RNG module
try:
    from cssm._c_rng import (
        generate_gaussian_samples,
        generate_uniform_samples,
        generate_levy_samples,
        py_mix_seed,
    )

    C_RNG_AVAILABLE = True
except ImportError:
    C_RNG_AVAILABLE = False

# Try to import the parallel module with CMS
try:
    from cssm._parallel_gsl import draw_random_stable

    CMS_AVAILABLE = True
except ImportError:
    CMS_AVAILABLE = False


# Custom markers for optional tests
pytestmark = [
    pytest.mark.rng_validation,
]


@pytest.mark.skipif(not C_RNG_AVAILABLE, reason="C RNG module not available")
class TestZigguratGaussian:
    """Tests for the Ziggurat Gaussian random number generator."""

    def test_gaussian_mean(self):
        """Test that Gaussian samples have approximately zero mean."""
        samples = generate_gaussian_samples(100_000, seed=42)

        # Mean should be close to 0
        assert abs(np.mean(samples)) < 0.01, (
            f"Mean {np.mean(samples):.4f} too far from 0"
        )

    def test_gaussian_std(self):
        """Test that Gaussian samples have approximately unit standard deviation."""
        samples = generate_gaussian_samples(100_000, seed=42)

        # Std should be close to 1
        assert abs(np.std(samples) - 1.0) < 0.02, (
            f"Std {np.std(samples):.4f} too far from 1"
        )

    def test_gaussian_skewness(self):
        """Test that Gaussian samples have approximately zero skewness."""
        samples = generate_gaussian_samples(100_000, seed=42)

        skewness = stats.skew(samples)
        assert abs(skewness) < 0.05, f"Skewness {skewness:.4f} too far from 0"

    def test_gaussian_kurtosis(self):
        """Test that Gaussian samples have approximately zero excess kurtosis."""
        samples = generate_gaussian_samples(100_000, seed=42)

        # Fisher kurtosis (excess kurtosis, so Gaussian = 0)
        kurt = stats.kurtosis(samples)
        assert abs(kurt) < 0.1, f"Excess kurtosis {kurt:.4f} too far from 0"

    @pytest.mark.slow
    def test_gaussian_ks_test(self):
        """Kolmogorov-Smirnov test against standard normal."""
        samples = generate_gaussian_samples(50_000, seed=42)

        # KS test against standard normal
        statistic, pvalue = stats.kstest(samples, "norm")

        # p-value should be > 0.01 (not significantly different from normal)
        assert pvalue > 0.01, (
            f"KS test failed: statistic={statistic:.4f}, p-value={pvalue:.4f}"
        )

    @pytest.mark.slow
    def test_gaussian_anderson_darling(self):
        """Anderson-Darling test for normality."""
        samples = generate_gaussian_samples(10_000, seed=42)

        result = stats.anderson(samples, dist="norm")

        # Check against 5% significance level (index 2)
        critical_5pct = result.critical_values[2]
        assert result.statistic < critical_5pct, (
            f"Anderson-Darling test failed: statistic={result.statistic:.4f} > {critical_5pct:.4f}"
        )

    def test_gaussian_multiple_seeds(self):
        """Test that different seeds produce different but valid samples."""
        samples1 = generate_gaussian_samples(10_000, seed=42)
        samples2 = generate_gaussian_samples(10_000, seed=123)

        # Different samples
        assert not np.allclose(samples1, samples2)

        # Both should be valid Gaussians
        for samples in [samples1, samples2]:
            assert abs(np.mean(samples)) < 0.05
            assert abs(np.std(samples) - 1.0) < 0.05


@pytest.mark.skipif(not C_RNG_AVAILABLE, reason="C RNG module not available")
class TestXoroshiroUniform:
    """Tests for the xoroshiro128+ uniform random number generator."""

    def test_uniform_range(self):
        """Test that uniform samples are in [0, 1)."""
        samples = generate_uniform_samples(100_000, seed=42)

        assert np.all(samples >= 0.0), "Some samples < 0"
        assert np.all(samples < 1.0), "Some samples >= 1"

    def test_uniform_mean(self):
        """Test that uniform samples have mean ≈ 0.5."""
        samples = generate_uniform_samples(100_000, seed=42)

        expected_mean = 0.5
        assert abs(np.mean(samples) - expected_mean) < 0.01, (
            f"Mean {np.mean(samples):.4f} too far from {expected_mean}"
        )

    def test_uniform_std(self):
        """Test that uniform samples have std ≈ 1/√12."""
        samples = generate_uniform_samples(100_000, seed=42)

        expected_std = 1.0 / np.sqrt(12)  # ≈ 0.2887
        assert abs(np.std(samples) - expected_std) < 0.01, (
            f"Std {np.std(samples):.4f} too far from {expected_std:.4f}"
        )

    @pytest.mark.slow
    def test_uniform_ks_test(self):
        """Kolmogorov-Smirnov test against uniform distribution."""
        samples = generate_uniform_samples(50_000, seed=42)

        statistic, pvalue = stats.kstest(samples, "uniform")

        assert pvalue > 0.01, (
            f"KS test failed: statistic={statistic:.4f}, p-value={pvalue:.4f}"
        )


@pytest.mark.skipif(not C_RNG_AVAILABLE, reason="C RNG module not available")
class TestCMSAlphaStable:
    """Tests for the Chambers-Mallows-Stuck alpha-stable generator."""

    @pytest.fixture
    def alpha_values(self):
        """Common alpha values to test."""
        return [1.2, 1.5, 1.8, 1.99]  # Various stability indices

    def test_cms_symmetric(self, alpha_values):
        """Test that symmetric alpha-stable samples are approximately symmetric."""
        for alpha in alpha_values:
            samples = generate_levy_samples(50_000, alpha, seed=42)

            # Mean should be close to 0 for symmetric stable
            # (though variance is infinite for alpha < 2)
            median = np.median(samples)
            assert abs(median) < 0.1, (
                f"alpha={alpha}: Median {median:.4f} too far from 0"
            )

    def test_cms_scale(self, alpha_values):
        """Test that samples have reasonable scale (IQR)."""
        for alpha in alpha_values:
            samples = generate_levy_samples(50_000, alpha, seed=42)

            # IQR should be positive and reasonable
            q75, q25 = np.percentile(samples, [75, 25])
            iqr = q75 - q25

            assert iqr > 0.5, f"alpha={alpha}: IQR {iqr:.4f} too small"
            assert iqr < 10.0, f"alpha={alpha}: IQR {iqr:.4f} too large"

    def test_cms_heavier_tails_for_lower_alpha(self):
        """Test that lower alpha produces heavier tails."""
        samples_12 = generate_levy_samples(50_000, alpha=1.2, seed=42)
        samples_18 = generate_levy_samples(50_000, alpha=1.8, seed=42)

        # Lower alpha should have more extreme values
        # Use 99th percentile as a measure of tail heaviness
        p99_12 = np.percentile(np.abs(samples_12), 99)
        p99_18 = np.percentile(np.abs(samples_18), 99)

        assert p99_12 > p99_18, (
            f"alpha=1.2 should have heavier tails than alpha=1.8: {p99_12:.2f} vs {p99_18:.2f}"
        )

    @pytest.mark.slow
    def test_cms_vs_scipy(self, alpha_values):
        """Compare CMS distribution against SciPy's levy_stable."""
        for alpha in alpha_values:
            cms_samples = generate_levy_samples(20_000, alpha, seed=42)
            scipy_samples = stats.levy_stable.rvs(
                alpha, 0, size=20_000, random_state=42
            )

            # Compare quantiles
            for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
                cms_q = np.percentile(cms_samples, q * 100)
                scipy_q = np.percentile(scipy_samples, q * 100)

                # Allow reasonable tolerance (these are different RNGs)
                rel_diff = abs(cms_q - scipy_q) / (abs(scipy_q) + 0.1)
                assert rel_diff < 0.3, (
                    f"alpha={alpha}, quantile={q}: CMS={cms_q:.3f}, SciPy={scipy_q:.3f}"
                )

    def test_cms_different_seeds(self, alpha_values):
        """Test that different seeds produce different samples."""
        for alpha in alpha_values:
            samples1 = generate_levy_samples(1000, alpha, seed=42)
            samples2 = generate_levy_samples(1000, alpha, seed=123)

            assert not np.allclose(samples1, samples2), (
                f"alpha={alpha}: Seeds should differ"
            )

    def test_cms_gaussian_limit(self):
        """Test that alpha=2 gives Gaussian distribution."""
        samples = generate_levy_samples(50_000, alpha=2.0, seed=42)

        # Should be approximately standard Gaussian N(0,1)
        # Note: Our implementation returns standard Gaussian for alpha=2,
        # not the S(2,0,1) stable distribution which would have variance 2
        assert abs(np.mean(samples)) < 0.05, (
            f"Mean {np.mean(samples):.4f} too far from 0"
        )
        expected_std = 1.0  # Standard Gaussian
        assert abs(np.std(samples) - expected_std) < 0.05, (
            f"Std {np.std(samples):.4f} too far from {expected_std:.4f}"
        )


@pytest.mark.skipif(not C_RNG_AVAILABLE, reason="C RNG module not available")
class TestRNGSeedMixing:
    """Tests for the seed mixing function."""

    def test_mix_seed_deterministic(self):
        """Test that seed mixing is deterministic."""
        seed1 = py_mix_seed(42, 0, 0)
        seed2 = py_mix_seed(42, 0, 0)

        assert seed1 == seed2

    def test_mix_seed_different_inputs(self):
        """Test that different inputs produce different mixed seeds."""
        seeds = set()
        for base in [42, 123]:
            for t1 in range(10):
                for t2 in range(10):
                    mixed = py_mix_seed(base, t1, t2)
                    seeds.add(mixed)

        # All combinations should produce unique seeds
        expected_unique = 2 * 10 * 10
        assert len(seeds) == expected_unique, (
            f"Expected {expected_unique} unique seeds, got {len(seeds)}"
        )


# Integration test using actual simulator
@pytest.mark.skipif(not C_RNG_AVAILABLE, reason="C RNG module not available")
class TestSimulatorRNGIntegration:
    """Integration tests verifying simulators use C RNG correctly."""

    def test_ddm_parallel_reproducible(self):
        """Test that DDM parallel produces reproducible results with same seed."""
        from cssm.ddm_models import ddm_flexbound
        from ssms.basic_simulators.boundary_functions import constant

        n_trials = 10
        v = np.array([0.5] * n_trials, dtype=np.float32)
        a = np.array([1.5] * n_trials, dtype=np.float32)
        z = np.array([0.5] * n_trials, dtype=np.float32)
        t = np.array([0.3] * n_trials, dtype=np.float32)
        deadline = np.array([10.0] * n_trials, dtype=np.float32)
        s = np.array([1.0] * n_trials, dtype=np.float32)

        boundary_params = {"a": a}

        # Run twice with same seed
        result1 = ddm_flexbound(
            v,
            a,
            z,
            t,
            deadline,
            s,
            n_samples=100,
            n_trials=n_trials,
            boundary_fun=constant,
            boundary_params=boundary_params,
            random_state=42,
            n_threads=4,
        )
        result2 = ddm_flexbound(
            v,
            a,
            z,
            t,
            deadline,
            s,
            n_samples=100,
            n_trials=n_trials,
            boundary_fun=constant,
            boundary_params=boundary_params,
            random_state=42,
            n_threads=4,
        )

        # Results should be identical
        np.testing.assert_array_equal(result1["rts"], result2["rts"])
        np.testing.assert_array_equal(result1["choices"], result2["choices"])

    def test_ddm_different_seeds_differ(self):
        """Test that different seeds produce different results."""
        from cssm.ddm_models import ddm_flexbound
        from ssms.basic_simulators.boundary_functions import constant

        n_trials = 10
        v = np.array([0.5] * n_trials, dtype=np.float32)
        a = np.array([1.5] * n_trials, dtype=np.float32)
        z = np.array([0.5] * n_trials, dtype=np.float32)
        t = np.array([0.3] * n_trials, dtype=np.float32)
        deadline = np.array([10.0] * n_trials, dtype=np.float32)
        s = np.array([1.0] * n_trials, dtype=np.float32)

        boundary_params = {"a": a}

        result1 = ddm_flexbound(
            v,
            a,
            z,
            t,
            deadline,
            s,
            n_samples=100,
            n_trials=n_trials,
            boundary_fun=constant,
            boundary_params=boundary_params,
            random_state=42,
            n_threads=4,
        )
        result2 = ddm_flexbound(
            v,
            a,
            z,
            t,
            deadline,
            s,
            n_samples=100,
            n_trials=n_trials,
            boundary_fun=constant,
            boundary_params=boundary_params,
            random_state=123,
            n_threads=4,
        )

        # Results should be different
        assert not np.allclose(result1["rts"], result2["rts"])


if __name__ == "__main__":
    # Run quick tests
    pytest.main([__file__, "-v", "-m", "not slow"])
