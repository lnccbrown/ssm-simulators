"""Tests for PyDDMEstimatorBuilder."""

import pytest
import numpy as np

# Import will fail if pyddm not installed - skip tests in that case
pytest.importorskip("pyddm")

from ssms.config import model_config
from ssms.dataset_generators.estimator_builders.pyddm_builder import (
    PyDDMEstimatorBuilder,
)
from ssms.dataset_generators.likelihood_estimators.pyddm_estimator import (
    PyDDMLikelihoodEstimator,
)


@pytest.fixture
def ddm_model_config():
    """DDM model configuration."""
    return model_config["ddm"]


@pytest.fixture
def ornstein_model_config():
    """Ornstein model configuration."""
    return model_config["ornstein"]


@pytest.fixture
def race_model_config():
    """Race model configuration (incompatible)."""
    return model_config["race_3"]


@pytest.fixture
def generator_config():
    """Standard generator configuration."""
    return {
        "delta_t": 0.001,
        "max_t": 10.0,
        "pdf_interpolation": "cubic",
    }


@pytest.fixture
def ddm_theta():
    """Standard DDM parameters."""
    return {"v": 1.5, "a": 2.0, "z": 0.5, "t": 0.3}


class TestPyDDMEstimatorBuilderInitialization:
    """Test builder initialization."""

    def test_initialization_with_compatible_model(
        self, ddm_model_config, generator_config
    ):
        """Test initialization with compatible model."""
        builder = PyDDMEstimatorBuilder(generator_config, ddm_model_config)

        assert builder.generator_config == generator_config
        assert builder.model_config == ddm_model_config

    def test_initialization_with_ornstein(
        self, ornstein_model_config, generator_config
    ):
        """Test initialization with Ornstein model (position-dependent drift)."""
        builder = PyDDMEstimatorBuilder(generator_config, ornstein_model_config)

        assert builder.model_config == ornstein_model_config

    def test_initialization_with_incompatible_model_fails(
        self, race_model_config, generator_config
    ):
        """Test that incompatible model raises ValueError at construction."""
        with pytest.raises(ValueError, match="not compatible with PyDDM"):
            PyDDMEstimatorBuilder(generator_config, race_model_config)

    def test_error_message_suggests_kde(self, race_model_config, generator_config):
        """Test that error message suggests using KDE."""
        with pytest.raises(ValueError, match="Use 'estimator_type': 'kde'"):
            PyDDMEstimatorBuilder(generator_config, race_model_config)


class TestPyDDMEstimatorBuilderBuild:
    """Test building estimators."""

    def test_build_creates_estimator(
        self, ddm_model_config, generator_config, ddm_theta
    ):
        """Test that build creates a PyDDMLikelihoodEstimator."""
        builder = PyDDMEstimatorBuilder(generator_config, ddm_model_config)

        estimator = builder.build(ddm_theta)

        assert isinstance(estimator, PyDDMLikelihoodEstimator)

    def test_build_without_simulations(
        self, ddm_model_config, generator_config, ddm_theta
    ):
        """Test that build works without simulations data."""
        builder = PyDDMEstimatorBuilder(generator_config, ddm_model_config)

        # Should work with simulations=None
        estimator = builder.build(ddm_theta, simulations=None)

        assert isinstance(estimator, PyDDMLikelihoodEstimator)

    def test_build_ignores_simulations(
        self, ddm_model_config, generator_config, ddm_theta
    ):
        """Test that build ignores simulations data if provided."""
        builder = PyDDMEstimatorBuilder(generator_config, ddm_model_config)

        # Provide dummy simulations (should be ignored)
        dummy_simulations = {"rts": np.array([1.0]), "choices": np.array([1])}

        estimator = builder.build(ddm_theta, simulations=dummy_simulations)

        assert isinstance(estimator, PyDDMLikelihoodEstimator)

    def test_built_estimator_can_evaluate(
        self, ddm_model_config, generator_config, ddm_theta
    ):
        """Test that built estimator can evaluate likelihoods."""
        builder = PyDDMEstimatorBuilder(generator_config, ddm_model_config)
        estimator = builder.build(ddm_theta)

        test_rts = np.array([0.5, 1.0, 1.5])
        test_choices = np.array([1, -1, 1])

        log_liks = estimator.evaluate(test_rts, test_choices)

        assert log_liks.shape == test_rts.shape
        assert np.all(np.isfinite(log_liks))

    def test_built_estimator_can_sample(
        self, ddm_model_config, generator_config, ddm_theta
    ):
        """Test that built estimator can sample."""
        builder = PyDDMEstimatorBuilder(generator_config, ddm_model_config)
        estimator = builder.build(ddm_theta)

        samples = estimator.sample(100)

        assert len(samples["rts"]) == 100
        assert len(samples["choices"]) == 100
        assert np.all(np.isin(samples["choices"], [-1, 1]))


class TestPyDDMEstimatorBuilderDifferentParameters:
    """Test builder with different parameter sets."""

    def test_different_theta_produces_different_estimators(
        self, ddm_model_config, generator_config
    ):
        """Test that different theta values produce different PDFs."""
        builder = PyDDMEstimatorBuilder(generator_config, ddm_model_config)

        theta1 = {"v": 1.0, "a": 2.0, "z": 0.5, "t": 0.3}
        theta2 = {"v": 2.0, "a": 2.0, "z": 0.5, "t": 0.3}  # Higher drift

        est1 = builder.build(theta1)
        est2 = builder.build(theta2)

        # PDFs should be different
        assert not np.allclose(est1.pdf_correct, est2.pdf_correct)

    def test_different_drift_affects_choice_probability(
        self, ddm_model_config, generator_config
    ):
        """Test that higher drift increases P(correct)."""
        from scipy import integrate

        builder = PyDDMEstimatorBuilder(generator_config, ddm_model_config)

        theta_low = {"v": 0.5, "a": 2.0, "z": 0.5, "t": 0.3}
        theta_high = {"v": 2.0, "a": 2.0, "z": 0.5, "t": 0.3}

        est_low = builder.build(theta_low)
        est_high = builder.build(theta_high)

        # Calculate P(correct) for each
        p_corr_low = integrate.trapezoid(est_low.pdf_correct, est_low.t_domain)
        p_corr_high = integrate.trapezoid(est_high.pdf_correct, est_high.t_domain)

        # Higher drift should increase P(correct)
        assert p_corr_high > p_corr_low


class TestPyDDMEstimatorBuilderInterpolation:
    """Test interpolation configuration."""

    def test_cubic_interpolation_default(self, ddm_model_config, ddm_theta):
        """Test that cubic interpolation is default."""
        config_no_interp = {"delta_t": 0.001, "max_t": 10.0}
        builder = PyDDMEstimatorBuilder(config_no_interp, ddm_model_config)

        estimator = builder.build(ddm_theta)

        # Cubic is default - verify interpolator exists and works
        assert estimator._interp_correct is not None
        test_rt = np.array([1.0])
        test_choice = np.array([1])
        ll = estimator.evaluate(test_rt, test_choice)
        assert np.isfinite(ll[0])

    def test_linear_interpolation_when_specified(self, ddm_model_config, ddm_theta):
        """Test that linear interpolation works when specified."""
        config_linear = {
            "delta_t": 0.001,
            "max_t": 10.0,
            "pdf_interpolation": "linear",
        }
        builder = PyDDMEstimatorBuilder(config_linear, ddm_model_config)

        estimator = builder.build(ddm_theta)

        # Linear interpolation specified - verify interpolator exists and works
        assert estimator._interp_correct is not None
        test_rt = np.array([1.0])
        test_choice = np.array([1])
        ll = estimator.evaluate(test_rt, test_choice)
        assert np.isfinite(ll[0])

    def test_different_interpolations_give_different_results(
        self, ddm_model_config, ddm_theta
    ):
        """Test that linear vs cubic interpolation produces different values."""
        config_linear = {
            "delta_t": 0.001,
            "max_t": 10.0,
            "pdf_interpolation": "linear",
        }
        config_cubic = {
            "delta_t": 0.001,
            "max_t": 10.0,
            "pdf_interpolation": "cubic",
        }

        builder_linear = PyDDMEstimatorBuilder(config_linear, ddm_model_config)
        builder_cubic = PyDDMEstimatorBuilder(config_cubic, ddm_model_config)

        est_linear = builder_linear.build(ddm_theta)
        est_cubic = builder_cubic.build(ddm_theta)

        # Evaluate at a point between grid points
        test_rt = 1.0005  # Slightly off-grid
        test_choice = np.array([1])

        ll_linear = est_linear.evaluate(np.array([test_rt]), test_choice)
        ll_cubic = est_cubic.evaluate(np.array([test_rt]), test_choice)

        # Should be different (but close)
        assert ll_linear[0] != ll_cubic[0]


class TestPyDDMEstimatorBuilderWithOrnstein:
    """Test builder with Ornstein model."""

    def test_ornstein_model_builds_successfully(
        self, ornstein_model_config, generator_config
    ):
        """Test that Ornstein model (position-dependent drift) works."""
        ornstein_theta = {"v": 1.5, "a": 2.0, "z": 0.5, "t": 0.3, "g": 0.2}

        builder = PyDDMEstimatorBuilder(generator_config, ornstein_model_config)
        estimator = builder.build(ornstein_theta)

        assert isinstance(estimator, PyDDMLikelihoodEstimator)

    def test_ornstein_leak_parameter_affects_pdf(
        self, ornstein_model_config, generator_config
    ):
        """Test that leak parameter affects the PDF."""
        theta_no_leak = {"v": 1.5, "a": 2.0, "z": 0.5, "t": 0.3, "g": 0.0}
        theta_with_leak = {"v": 1.5, "a": 2.0, "z": 0.5, "t": 0.3, "g": 0.5}

        builder = PyDDMEstimatorBuilder(generator_config, ornstein_model_config)

        est_no_leak = builder.build(theta_no_leak)
        est_with_leak = builder.build(theta_with_leak)

        # PDFs should be different
        assert not np.allclose(est_no_leak.pdf_correct, est_with_leak.pdf_correct)


class TestPyDDMEstimatorBuilderProtocolCompliance:
    """Test protocol compliance."""

    def test_implements_estimator_builder_protocol(
        self, ddm_model_config, generator_config, ddm_theta
    ):
        """Test that builder follows EstimatorBuilderProtocol."""
        builder = PyDDMEstimatorBuilder(generator_config, ddm_model_config)

        # Should have build method
        assert hasattr(builder, "build")
        assert callable(builder.build)

        # build should return something with the right methods
        estimator = builder.build(ddm_theta)
        assert hasattr(estimator, "fit")
        assert hasattr(estimator, "evaluate")
        assert hasattr(estimator, "sample")
        assert hasattr(estimator, "get_metadata")


class TestPyDDMEstimatorBuilderEdgeCases:
    """Test edge cases and error handling."""

    def test_very_short_max_t(self, ddm_model_config, ddm_theta):
        """Test with very short maximum time."""
        config_short = {"delta_t": 0.001, "max_t": 1.0}

        builder = PyDDMEstimatorBuilder(config_short, ddm_model_config)
        estimator = builder.build(ddm_theta)

        # Should still work
        assert isinstance(estimator, PyDDMLikelihoodEstimator)
        assert estimator.t_domain[-1] <= 1.0

    def test_very_fine_dt(self, ddm_model_config, ddm_theta):
        """Test with very fine time step."""
        config_fine = {"delta_t": 0.0001, "max_t": 2.0}

        builder = PyDDMEstimatorBuilder(config_fine, ddm_model_config)
        estimator = builder.build(ddm_theta)

        # Should still work
        assert isinstance(estimator, PyDDMLikelihoodEstimator)

    def test_extreme_parameter_values(self, ddm_model_config, generator_config):
        """Test with extreme but valid parameter values."""
        theta_extreme = {
            "v": 5.0,  # Very high drift
            "a": 0.5,  # Very low boundary
            "z": 0.9,  # Very biased starting point
            "t": 0.1,  # Short non-decision time
        }

        builder = PyDDMEstimatorBuilder(generator_config, ddm_model_config)
        estimator = builder.build(theta_extreme)

        # Should still produce valid estimator
        assert isinstance(estimator, PyDDMLikelihoodEstimator)

        # Should be able to evaluate
        log_liks = estimator.evaluate(np.array([0.5]), np.array([1]))
        assert np.isfinite(log_liks[0])


class TestPyDDMEstimatorBuilderMultipleCalls:
    """Test multiple builds from same builder."""

    def test_multiple_builds_from_same_builder(
        self, ddm_model_config, generator_config
    ):
        """Test that same builder can build multiple estimators."""
        builder = PyDDMEstimatorBuilder(generator_config, ddm_model_config)

        theta1 = {"v": 1.0, "a": 2.0, "z": 0.5, "t": 0.3}
        theta2 = {"v": 1.5, "a": 2.0, "z": 0.5, "t": 0.3}
        theta3 = {"v": 2.0, "a": 2.0, "z": 0.5, "t": 0.3}

        est1 = builder.build(theta1)
        est2 = builder.build(theta2)
        est3 = builder.build(theta3)

        # All should be valid estimators
        assert all(isinstance(e, PyDDMLikelihoodEstimator) for e in [est1, est2, est3])

        # Should be independent (different PDFs)
        assert not np.allclose(est1.pdf_correct, est2.pdf_correct)
        assert not np.allclose(est2.pdf_correct, est3.pdf_correct)
