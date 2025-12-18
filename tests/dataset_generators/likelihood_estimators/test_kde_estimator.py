"""Tests for KDE likelihood estimator.

These tests verify that KDELikelihoodEstimator correctly implements the
LikelihoodEstimatorProtocol and produces expected results.
"""

import numpy as np
import pytest

from ssms.basic_simulators.simulator import simulator
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
        random_state=42,  # Fixed seed for reproducibility
    )


def test_kde_estimator_initialization():
    """Test that KDE estimator initializes correctly."""
    estimator = KDELikelihoodEstimator(displace_t=False)

    assert estimator.displace_t is False
    assert estimator._kde is None
    assert estimator._metadata is None

    # Test with displace_t=True
    estimator_with_displace = KDELikelihoodEstimator(displace_t=True)
    assert estimator_with_displace.displace_t is True


def test_kde_estimator_requires_fit():
    """Test that methods raise error before fit() is called."""
    estimator = KDELikelihoodEstimator()

    # evaluate() should raise
    with pytest.raises(ValueError, match="Must call fit"):
        estimator.evaluate(np.array([1.0]), np.array([1]))

    # sample() should raise
    with pytest.raises(ValueError, match="Must call fit"):
        estimator.sample(10)

    # get_metadata() should raise
    with pytest.raises(ValueError, match="Must call fit"):
        estimator.get_metadata()


def test_kde_estimator_fit(ddm_simulations):
    """Test that fit() correctly builds the KDE."""
    estimator = KDELikelihoodEstimator()
    estimator.fit(ddm_simulations)

    # After fit, KDE and metadata should be set
    assert estimator._kde is not None
    assert estimator._metadata is not None
    assert estimator._metadata == ddm_simulations["metadata"]


def test_kde_estimator_evaluate(ddm_simulations):
    """Test that evaluate() returns log-likelihoods."""
    estimator = KDELikelihoodEstimator()
    estimator.fit(ddm_simulations)

    # Evaluate at some points
    test_rts = np.array([0.5, 1.0, 1.5, 2.0])
    test_choices = np.array([1, -1, 1, -1])

    log_liks = estimator.evaluate(test_rts, test_choices)

    # Check output structure
    assert isinstance(log_liks, np.ndarray)
    assert log_liks.shape == (4,)
    assert log_liks.dtype in [np.float32, np.float64]

    # Log-likelihoods should be negative (or zero for perfect matches)
    assert np.all(log_liks <= 0)


def test_kde_estimator_evaluate_at_simulated_points(ddm_simulations):
    """Test evaluating at points from the original simulations."""
    estimator = KDELikelihoodEstimator()
    estimator.fit(ddm_simulations)

    # Take first 10 simulated points
    test_rts = ddm_simulations["rts"][:10]
    test_choices = ddm_simulations["choices"][:10]

    log_liks = estimator.evaluate(test_rts, test_choices)

    # Should get reasonable log-likelihoods (not -inf)
    assert np.all(np.isfinite(log_liks))
    assert log_liks.shape == (10,)


def test_kde_estimator_sample(ddm_simulations):
    """Test that sample() generates valid samples."""
    estimator = KDELikelihoodEstimator()
    estimator.fit(ddm_simulations)

    n_samples = 100
    samples = estimator.sample(n_samples)

    # Check output structure
    assert isinstance(samples, dict)
    assert "rts" in samples
    assert "choices" in samples

    # Check shapes
    assert samples["rts"].shape == (n_samples,)
    assert samples["choices"].shape == (n_samples,)

    # Check that RTs are positive
    assert np.all(samples["rts"] > 0)

    # Check that choices are valid
    possible_choices = ddm_simulations["metadata"]["possible_choices"]
    assert np.all(np.isin(samples["choices"], possible_choices))


def test_kde_estimator_sample_reproducibility(ddm_simulations):
    """Test that sampling produces different results on repeated calls."""
    estimator = KDELikelihoodEstimator()
    estimator.fit(ddm_simulations)

    samples1 = estimator.sample(50)
    samples2 = estimator.sample(50)

    # Samples should be different (extremely unlikely to be identical)
    assert not np.allclose(samples1["rts"], samples2["rts"])


def test_kde_estimator_get_metadata(ddm_simulations):
    """Test that get_metadata() returns correct information."""
    estimator = KDELikelihoodEstimator()
    estimator.fit(ddm_simulations)

    metadata = estimator.get_metadata()

    # Check that it returns the simulation metadata
    assert metadata == ddm_simulations["metadata"]
    assert "max_t" in metadata
    assert "possible_choices" in metadata
    assert "model" in metadata


def test_kde_estimator_multiple_fits(ddm_simulations):
    """Test that calling fit() multiple times updates the KDE."""
    estimator = KDELikelihoodEstimator()

    # First fit
    estimator.fit(ddm_simulations)
    first_kde = estimator._kde

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

    # Second fit
    estimator.fit(new_simulations)
    second_kde = estimator._kde

    # KDE should be different (different object)
    assert second_kde is not first_kde

    # Metadata should be updated
    assert estimator._metadata == new_simulations["metadata"]


def test_kde_estimator_with_displace_t(ddm_simulations):
    """Test KDE estimator with displace_t option."""
    estimator = KDELikelihoodEstimator(displace_t=True)

    # Should work without error when all simulations have same t
    estimator.fit(ddm_simulations)

    # Verify it's fitted
    assert estimator._kde is not None


def test_kde_estimator_consistency_across_calls(ddm_simulations):
    """Test that evaluate() returns consistent results for same inputs."""
    estimator = KDELikelihoodEstimator()
    estimator.fit(ddm_simulations)

    test_rts = np.array([0.8, 1.2, 1.6])
    test_choices = np.array([1, -1, 1])

    # Evaluate twice with same inputs
    log_liks1 = estimator.evaluate(test_rts, test_choices)
    log_liks2 = estimator.evaluate(test_rts, test_choices)

    # Should get identical results
    assert np.allclose(log_liks1, log_liks2)


def test_kde_estimator_protocol_compliance(ddm_simulations):
    """Test that KDELikelihoodEstimator implements all protocol methods."""
    estimator = KDELikelihoodEstimator()

    # Verify all required methods exist
    assert hasattr(estimator, "fit")
    assert hasattr(estimator, "evaluate")
    assert hasattr(estimator, "sample")
    assert hasattr(estimator, "get_metadata")

    # Verify they're callable
    assert callable(estimator.fit)
    assert callable(estimator.evaluate)
    assert callable(estimator.sample)
    assert callable(estimator.get_metadata)

    # Verify full workflow works
    estimator.fit(ddm_simulations)
    samples = estimator.sample(10)
    log_liks = estimator.evaluate(samples["rts"], samples["choices"])
    metadata = estimator.get_metadata()

    assert samples["rts"].shape == (10,)
    assert log_liks.shape == (10,)
    assert isinstance(metadata, dict)


def test_kde_estimator_with_different_models():
    """Test KDE estimator with different SSM models."""
    # Test with ornstein model
    theta = {"v": 0.5, "a": 1.5, "z": 0.5, "g": -0.5, "t": 0.2}

    simulations = simulator(
        theta=theta,
        model="ornstein",
        n_samples=1000,
        max_t=5.0,
        delta_t=0.001,
        smooth_unif=False,
        random_state=42,
    )

    estimator = KDELikelihoodEstimator()
    estimator.fit(simulations)

    # Should work without error
    samples = estimator.sample(20)
    log_liks = estimator.evaluate(samples["rts"], samples["choices"])

    assert samples["rts"].shape == (20,)
    assert log_liks.shape == (20,)
    assert np.all(np.isfinite(log_liks))
