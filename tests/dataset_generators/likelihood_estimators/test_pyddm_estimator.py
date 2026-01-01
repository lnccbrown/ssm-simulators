"""Tests for PyDDMLikelihoodEstimator."""

import pytest
import numpy as np
from scipy import integrate

# Import will fail if pyddm not installed - skip tests in that case
pytest.importorskip("pyddm")

from ssms.external_simulators import SSMSToPyDDMMapper
from ssms.dataset_generators.likelihood_estimators.pyddm_estimator import (
    PyDDMLikelihoodEstimator,
)
from ssms.config import model_config


@pytest.fixture
def ddm_theta():
    """Standard DDM parameters."""
    return {"v": 1.5, "a": 2.0, "z": 0.5, "t": 0.3}


@pytest.fixture
def generator_config():
    """Generator configuration."""
    return {"delta_t": 0.001, "max_t": 10.0}


@pytest.fixture
def ddm_solution(ddm_theta, generator_config):
    """Create a solved PyDDM model for testing."""
    pyddm_model = SSMSToPyDDMMapper.build_pyddm_model(
        model_config=model_config["ddm"],
        theta=ddm_theta,
        generator_config=generator_config,
    )
    solution = pyddm_model.solve()
    t_domain = pyddm_model.t_domain()
    return solution, t_domain


class TestPyDDMLikelihoodEstimatorInitialization:
    """Test estimator initialization."""

    def test_initialization_basic(self, ddm_solution):
        """Test basic initialization."""
        solution, t_domain = ddm_solution

        estimator = PyDDMLikelihoodEstimator(
            pyddm_solution=solution,
            t_domain=t_domain,
            interpolation="cubic",
        )

        assert estimator.solution is solution
        assert np.array_equal(estimator.t_domain, t_domain)
        assert estimator.pdf_correct.shape == t_domain.shape
        assert estimator.pdf_error.shape == t_domain.shape

    def test_pdfs_extracted_correctly(self, ddm_solution):
        """Test that PDFs are extracted from solution."""
        solution, t_domain = ddm_solution

        estimator = PyDDMLikelihoodEstimator(solution, t_domain)

        # PDFs should match direct extraction
        assert np.allclose(estimator.pdf_correct, solution.pdf("correct"))
        assert np.allclose(estimator.pdf_error, solution.pdf("error"))

    def test_interpolation_types(self, ddm_solution):
        """Test different interpolation methods can be initialized."""
        solution, t_domain = ddm_solution

        # Linear interpolation
        est_linear = PyDDMLikelihoodEstimator(
            solution, t_domain, interpolation="linear"
        )
        assert est_linear._interp_correct is not None
        assert est_linear._interp_error is not None

        # Cubic interpolation
        est_cubic = PyDDMLikelihoodEstimator(solution, t_domain, interpolation="cubic")
        assert est_cubic._interp_correct is not None
        assert est_cubic._interp_error is not None

    def test_metadata_stored(self, ddm_solution):
        """Test that metadata is properly stored."""
        solution, t_domain = ddm_solution

        estimator = PyDDMLikelihoodEstimator(solution, t_domain)
        metadata = estimator.get_metadata()

        assert metadata["max_t"] == pytest.approx(t_domain[-1])
        assert metadata["dt"] == pytest.approx(t_domain[1] - t_domain[0])
        assert metadata["possible_choices"] == [-1, 1]
        assert metadata["n_choices"] == 2


class TestPyDDMLikelihoodEstimatorEvaluation:
    """Test likelihood evaluation."""

    def test_evaluate_correct_choices(self, ddm_solution):
        """Test evaluation for correct choices."""
        solution, t_domain = ddm_solution
        estimator = PyDDMLikelihoodEstimator(solution, t_domain)

        # Test at a few time points
        test_rts = np.array([0.5, 1.0, 1.5, 2.0])
        test_choices = np.ones(len(test_rts))

        log_liks = estimator.evaluate(test_rts, test_choices)

        assert log_liks.shape == test_rts.shape
        assert np.all(np.isfinite(log_liks))
        assert np.all(log_liks <= 0)  # Log probabilities should be <= 0

    def test_evaluate_error_choices(self, ddm_solution):
        """Test evaluation for error choices."""
        solution, t_domain = ddm_solution
        estimator = PyDDMLikelihoodEstimator(solution, t_domain)

        test_rts = np.array([0.5, 1.0, 1.5, 2.0])
        test_choices = -np.ones(len(test_rts))

        log_liks = estimator.evaluate(test_rts, test_choices)

        assert log_liks.shape == test_rts.shape
        assert np.all(np.isfinite(log_liks))

    def test_evaluate_mixed_choices(self, ddm_solution):
        """Test evaluation with mixed correct/error choices."""
        solution, t_domain = ddm_solution
        estimator = PyDDMLikelihoodEstimator(solution, t_domain)

        test_rts = np.array([0.5, 1.0, 1.5, 2.0])
        test_choices = np.array([1, -1, 1, -1])

        log_liks = estimator.evaluate(test_rts, test_choices)

        assert log_liks.shape == test_rts.shape
        assert np.all(np.isfinite(log_liks))

    def test_evaluate_out_of_bounds_rts(self, ddm_solution):
        """Test that out-of-bounds RTs get very low likelihood."""
        solution, t_domain = ddm_solution
        estimator = PyDDMLikelihoodEstimator(solution, t_domain)

        # RT beyond max_t
        test_rts = np.array([t_domain[-1] + 5.0])
        test_choices = np.array([1])

        log_liks = estimator.evaluate(test_rts, test_choices)

        # Should be very negative (effectively zero probability)
        assert log_liks[0] < -60

    def test_evaluate_consistency_with_pdf(self, ddm_solution):
        """Test that evaluation is consistent with direct PDF values."""
        solution, t_domain = ddm_solution
        estimator = PyDDMLikelihoodEstimator(solution, t_domain)

        # Pick some time points that are in the domain
        idx = len(t_domain) // 2
        test_rt = t_domain[idx]

        # Evaluate for correct choice
        log_lik = estimator.evaluate(np.array([test_rt]), np.array([1]))

        # Should be close to log of the PDF value (within interpolation error)
        expected_log_lik = np.log(estimator.pdf_correct[idx])
        assert log_lik[0] == pytest.approx(expected_log_lik, abs=0.1)


class TestPyDDMLikelihoodEstimatorSampling:
    """Test sampling from the estimator."""

    def test_sample_returns_correct_shapes(self, ddm_solution):
        """Test that sample returns correctly shaped arrays."""
        solution, t_domain = ddm_solution
        estimator = PyDDMLikelihoodEstimator(solution, t_domain)

        n_samples = 1000
        samples = estimator.sample(n_samples)

        assert "rts" in samples and "choices" in samples
        assert samples["rts"].shape == (n_samples,)
        assert samples["choices"].shape == (n_samples,)

    def test_sample_choices_in_valid_set(self, ddm_solution):
        """Test that sampled choices are valid."""
        solution, t_domain = ddm_solution
        estimator = PyDDMLikelihoodEstimator(solution, t_domain)

        samples = estimator.sample(1000)

        # All choices should be -1 or 1
        assert np.all(np.isin(samples["choices"], [-1, 1]))

    def test_sample_rts_positive(self, ddm_solution):
        """Test that sampled RTs are positive."""
        solution, t_domain = ddm_solution
        estimator = PyDDMLikelihoodEstimator(solution, t_domain)

        samples = estimator.sample(1000)

        assert np.all(samples["rts"] > 0)

    def test_sample_choice_proportions(self, ddm_solution, ddm_theta):
        """Test that sampled choice proportions match PDF integrals."""
        solution, t_domain = ddm_solution
        estimator = PyDDMLikelihoodEstimator(solution, t_domain)

        # Large sample for good statistics
        samples = estimator.sample(10000)

        # Calculate empirical P(correct)
        p_correct_empirical = np.mean(samples["choices"] == 1)

        # Calculate theoretical P(correct) from PDF
        p_correct_theoretical = integrate.trapezoid(estimator.pdf_correct, t_domain)

        # Should be close (within sampling error)
        assert p_correct_empirical == pytest.approx(p_correct_theoretical, abs=0.02)


class TestPyDDMLikelihoodEstimatorProtocolCompliance:
    """Test that estimator follows the protocol."""

    def test_has_required_methods(self, ddm_solution):
        """Test that estimator has all required methods."""
        solution, t_domain = ddm_solution
        estimator = PyDDMLikelihoodEstimator(solution, t_domain)

        assert hasattr(estimator, "fit")
        assert hasattr(estimator, "evaluate")
        assert hasattr(estimator, "sample")
        assert hasattr(estimator, "get_metadata")
        assert callable(estimator.fit)
        assert callable(estimator.evaluate)
        assert callable(estimator.sample)
        assert callable(estimator.get_metadata)

    def test_fit_is_noop(self, ddm_solution):
        """Test that fit() doesn't raise errors (even though it's a no-op)."""
        solution, t_domain = ddm_solution
        estimator = PyDDMLikelihoodEstimator(solution, t_domain)

        # Should not raise any errors
        estimator.fit({"dummy": "data"})
        estimator.fit(None)


class TestPyDDMLikelihoodEstimatorInterpolation:
    """Test interpolation behavior."""

    def test_linear_vs_cubic_interpolation(self, ddm_solution):
        """Test that both linear and cubic interpolation work."""
        solution, t_domain = ddm_solution

        est_linear = PyDDMLikelihoodEstimator(
            solution, t_domain, interpolation="linear"
        )
        est_cubic = PyDDMLikelihoodEstimator(solution, t_domain, interpolation="cubic")

        # Test at a point between grid points
        test_rt = (t_domain[100] + t_domain[101]) / 2
        test_choice = np.array([1])

        ll_linear = est_linear.evaluate(np.array([test_rt]), test_choice)
        ll_cubic = est_cubic.evaluate(np.array([test_rt]), test_choice)

        # Both should produce finite, reasonable log-likelihoods
        assert np.isfinite(ll_linear[0])
        assert np.isfinite(ll_cubic[0])
        assert ll_linear[0] > -100  # Not too extreme
        assert ll_cubic[0] > -100
        # With a fine grid (dt=0.001), they may be very similar
        assert abs(ll_linear[0] - ll_cubic[0]) < 5.0  # Within reasonable range

    def test_interpolation_smoothness(self, ddm_solution):
        """Test that interpolation produces smooth results."""
        solution, t_domain = ddm_solution
        estimator = PyDDMLikelihoodEstimator(solution, t_domain, interpolation="cubic")

        # Create dense time grid
        test_rts = np.linspace(0.5, 3.0, 100)
        test_choices = np.ones(len(test_rts))

        log_liks = estimator.evaluate(test_rts, test_choices)

        # Differences should be small (smooth function)
        diffs = np.diff(log_liks)
        assert np.all(np.abs(diffs) < 1.0)  # No large jumps


class TestPyDDMLikelihoodEstimatorWithOrnstein:
    """Test estimator with Ornstein model (position-dependent drift)."""

    def test_ornstein_model(self, generator_config):
        """Test that estimator works with Ornstein model."""
        ornstein_theta = {"v": 1.5, "a": 2.0, "z": 0.5, "t": 0.3, "g": 0.2}

        pyddm_model = SSMSToPyDDMMapper.build_pyddm_model(
            model_config=model_config["ornstein"],
            theta=ornstein_theta,
            generator_config=generator_config,
        )
        solution = pyddm_model.solve()
        t_domain = pyddm_model.t_domain()

        estimator = PyDDMLikelihoodEstimator(solution, t_domain)

        # Test evaluation
        test_rts = np.array([0.5, 1.0, 1.5])
        test_choices = np.array([1, -1, 1])

        log_liks = estimator.evaluate(test_rts, test_choices)

        assert log_liks.shape == test_rts.shape
        assert np.all(np.isfinite(log_liks))

        # Test sampling
        samples = estimator.sample(100)
        assert len(samples["rts"]) == 100
        assert np.all(np.isin(samples["choices"], [-1, 1]))


class TestPyDDMLikelihoodEstimatorEdgeCases:
    """Test edge cases and error handling."""

    def test_single_rt_evaluation(self, ddm_solution):
        """Test evaluation with single RT."""
        solution, t_domain = ddm_solution
        estimator = PyDDMLikelihoodEstimator(solution, t_domain)

        log_lik = estimator.evaluate(np.array([1.0]), np.array([1]))

        assert log_lik.shape == (1,)
        assert np.isfinite(log_lik[0])

    def test_empty_array_evaluation(self, ddm_solution):
        """Test evaluation with empty arrays."""
        solution, t_domain = ddm_solution
        estimator = PyDDMLikelihoodEstimator(solution, t_domain)

        log_liks = estimator.evaluate(np.array([]), np.array([]))

        assert log_liks.shape == (0,)

    def test_very_early_rts(self, ddm_solution):
        """Test evaluation at very early RTs (near 0)."""
        solution, t_domain = ddm_solution
        estimator = PyDDMLikelihoodEstimator(solution, t_domain)

        # Very early RT (but not negative)
        test_rt = t_domain[1]  # Second time point
        log_lik = estimator.evaluate(np.array([test_rt]), np.array([1]))

        assert np.isfinite(log_lik[0])
