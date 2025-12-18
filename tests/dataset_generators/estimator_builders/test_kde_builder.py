"""Tests for KDE estimator builder.

These tests verify that KDEEstimatorBuilder correctly:
1. Extracts parameters from generator_config
2. Builds fitted KDELikelihoodEstimator instances
3. Implements EstimatorBuilderProtocol
"""

import numpy as np
import pytest

from ssms.basic_simulators.simulator import simulator
from ssms.dataset_generators.estimator_builders import KDEEstimatorBuilder
from ssms.dataset_generators.likelihood_estimators import KDELikelihoodEstimator


@pytest.fixture
def ddm_simulations():
    """Generate DDM simulations for testing."""
    theta = {"v": 1.0, "a": 2.0, "z": 0.5, "t": 0.3}
    return simulator(
        theta=theta,
        model="ddm",
        n_samples=1000,
        max_t=5.0,
        delta_t=0.001,
        smooth_unif=False,
        random_state=42,
    )


@pytest.fixture
def theta():
    """Sample parameter dictionary."""
    return {"v": 1.0, "a": 2.0, "z": 0.5, "t": 0.3}


def test_kde_builder_initialization_default():
    """Test builder initializes with default displace_t=False."""
    config = {"n_samples": 1000, "max_t": 5.0}
    builder = KDEEstimatorBuilder(config)

    assert builder.generator_config == config
    assert builder.displace_t is False


def test_kde_builder_initialization_with_displace_t():
    """Test builder extracts displace_t from config."""
    config = {"kde_displace_t": True, "n_samples": 1000}
    builder = KDEEstimatorBuilder(config)

    assert builder.displace_t is True


def test_kde_builder_builds_fitted_estimator(theta, ddm_simulations):
    """Test that build() returns a fitted estimator."""
    config = {"kde_displace_t": False}
    builder = KDEEstimatorBuilder(config)

    estimator = builder.build(theta, ddm_simulations)

    # Verify it's a KDELikelihoodEstimator
    assert isinstance(estimator, KDELikelihoodEstimator)

    # Verify it's already fitted
    assert estimator._kde is not None
    assert estimator._metadata is not None


def test_kde_builder_builds_with_correct_displace_t(theta, ddm_simulations):
    """Test that builder passes displace_t to estimator correctly."""
    # Test with displace_t=True
    config_with_displace = {"kde_displace_t": True}
    builder_with_displace = KDEEstimatorBuilder(config_with_displace)
    estimator_with_displace = builder_with_displace.build(theta, ddm_simulations)

    assert estimator_with_displace.displace_t is True

    # Test with displace_t=False (default)
    config_without_displace = {}
    builder_without_displace = KDEEstimatorBuilder(config_without_displace)
    estimator_without_displace = builder_without_displace.build(theta, ddm_simulations)

    assert estimator_without_displace.displace_t is False


def test_kde_builder_requires_simulations(theta):
    """Test that build() raises error when simulations=None."""
    config = {"kde_displace_t": False}
    builder = KDEEstimatorBuilder(config)

    with pytest.raises(ValueError, match="KDE estimator requires simulations"):
        builder.build(theta, simulations=None)


def test_kde_builder_estimator_can_evaluate(theta, ddm_simulations):
    """Test that built estimator can evaluate log-likelihoods."""
    config = {"kde_displace_t": False}
    builder = KDEEstimatorBuilder(config)
    estimator = builder.build(theta, ddm_simulations)

    # Evaluate at some test points
    test_rts = np.array([0.5, 1.0, 1.5])
    test_choices = np.array([1, -1, 1])

    log_liks = estimator.evaluate(test_rts, test_choices)

    assert isinstance(log_liks, np.ndarray)
    assert log_liks.shape == (3,)
    assert np.all(log_liks <= 0)


def test_kde_builder_estimator_can_sample(theta, ddm_simulations):
    """Test that built estimator can sample."""
    config = {"kde_displace_t": False}
    builder = KDEEstimatorBuilder(config)
    estimator = builder.build(theta, ddm_simulations)

    samples = estimator.sample(n_samples=50)

    assert isinstance(samples, dict)
    assert "rts" in samples
    assert "choices" in samples
    assert samples["rts"].shape == (50,)
    assert samples["choices"].shape == (50,)


def test_kde_builder_estimator_has_metadata(theta, ddm_simulations):
    """Test that built estimator has metadata."""
    config = {"kde_displace_t": False}
    builder = KDEEstimatorBuilder(config)
    estimator = builder.build(theta, ddm_simulations)

    metadata = estimator.get_metadata()

    assert metadata == ddm_simulations["metadata"]
    assert "max_t" in metadata
    assert "model" in metadata


def test_kde_builder_multiple_builds(theta, ddm_simulations):
    """Test that builder can be reused to build multiple estimators."""
    config = {"kde_displace_t": False}
    builder = KDEEstimatorBuilder(config)

    # Build first estimator
    estimator1 = builder.build(theta, ddm_simulations)

    # Generate different simulations
    new_theta = {"v": -0.5, "a": 1.5, "z": 0.5, "t": 0.2}
    new_simulations = simulator(
        theta=new_theta,
        model="ddm",
        n_samples=1000,
        max_t=5.0,
        delta_t=0.001,
        smooth_unif=False,
        random_state=123,
    )

    # Build second estimator
    estimator2 = builder.build(new_theta, new_simulations)

    # Both should be valid but different
    assert isinstance(estimator1, KDELikelihoodEstimator)
    assert isinstance(estimator2, KDELikelihoodEstimator)
    assert estimator1 is not estimator2
    assert estimator1._metadata != estimator2._metadata


def test_kde_builder_protocol_compliance(theta, ddm_simulations):
    """Test that KDEEstimatorBuilder implements EstimatorBuilderProtocol."""
    config = {"kde_displace_t": False}
    builder = KDEEstimatorBuilder(config)

    # Verify required method exists and is callable
    assert hasattr(builder, "build")
    assert callable(builder.build)

    # Verify it works in the protocol workflow
    estimator = builder.build(theta, ddm_simulations)
    samples = estimator.sample(10)
    log_liks = estimator.evaluate(samples["rts"], samples["choices"])
    metadata = estimator.get_metadata()

    assert samples["rts"].shape == (10,)
    assert log_liks.shape == (10,)
    assert isinstance(metadata, dict)


def test_kde_builder_with_complex_config():
    """Test that builder only extracts what it needs from config."""
    # Config with many unrelated fields
    complex_config = {
        "kde_displace_t": True,
        "n_samples": 2000,
        "max_t": 10.0,
        "n_parameter_sets": 100,
        "output_folder": "/tmp/data",
        "model": "ddm",
        "some_other_field": "irrelevant",
    }

    builder = KDEEstimatorBuilder(complex_config)

    # Only displace_t should be extracted
    assert builder.displace_t is True
    assert (
        builder.generator_config == complex_config
    )  # Full config stored for reference


def test_kde_builder_config_isolation():
    """Test that modifying config after builder creation doesn't affect builder."""
    config = {"kde_displace_t": False}
    builder = KDEEstimatorBuilder(config)

    # Modify the original config
    config["kde_displace_t"] = True

    # Builder should still have the original value
    # Note: This test may fail if builder doesn't copy config
    # That's okay - we can decide if we want defensive copying
    assert builder.generator_config["kde_displace_t"] is True  # References same dict
    assert builder.displace_t is False  # But cached value unchanged
