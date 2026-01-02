"""Tests for dataset generator protocols.

These tests verify that the protocol definitions are correctly structured
and can be implemented by mock classes.
"""

import numpy as np
import pytest


class MockLikelihoodEstimator:
    """Mock implementation of LikelihoodEstimatorProtocol for testing."""

    def __init__(self):
        self._fitted = False
        self._metadata = {"max_t": 20.0, "possible_choices": [-1, 1]}

    def fit(self, simulations: dict) -> None:
        """Mock fit method."""
        self._fitted = True

    def evaluate(self, rts: np.ndarray, choices: np.ndarray) -> np.ndarray:
        """Mock evaluate method."""
        if not self._fitted:
            raise ValueError("Must call fit() before evaluate()")
        return np.ones(len(rts)) * -2.0  # Mock log-likelihood

    def sample(self, n_samples: int) -> dict:
        """Mock sample method."""
        if not self._fitted:
            raise ValueError("Must call fit() before sample()")
        return {
            "rts": np.random.uniform(0, 5, n_samples),
            "choices": np.random.choice([-1, 1], n_samples),
        }

    def get_metadata(self) -> dict:
        """Mock get_metadata method."""
        return self._metadata


class MockEstimatorBuilder:
    """Mock implementation of EstimatorBuilderProtocol for testing."""

    def build(self, theta: dict, simulations: dict | None = None):
        """Mock build method."""
        estimator = MockLikelihoodEstimator()
        if simulations is not None:
            estimator.fit(simulations)
        return estimator


class MockTrainingStrategy:
    """Mock implementation of TrainingDataStrategyProtocol for testing."""

    def generate(self, theta: dict, likelihood_estimator) -> np.ndarray:
        """Mock generate method."""
        n_samples = 10
        n_params = len(theta)

        # Create output array
        out = np.zeros((n_samples, n_params + 3))

        # Fill parameter columns
        out[:, :n_params] = np.tile(list(theta.values()), (n_samples, 1))

        # Sample RTs and choices
        samples = likelihood_estimator.sample(n_samples)
        out[:, -3] = samples["rts"]
        out[:, -2] = samples["choices"]

        # Evaluate likelihoods
        out[:, -1] = likelihood_estimator.evaluate(samples["rts"], samples["choices"])

        return out


def test_likelihood_estimator_protocol_compliance():
    """Test that MockLikelihoodEstimator implements the protocol."""
    estimator = MockLikelihoodEstimator()

    # Verify it has all required methods
    assert hasattr(estimator, "fit")
    assert hasattr(estimator, "evaluate")
    assert hasattr(estimator, "sample")
    assert hasattr(estimator, "get_metadata")

    # Verify methods are callable
    assert callable(estimator.fit)
    assert callable(estimator.evaluate)
    assert callable(estimator.sample)
    assert callable(estimator.get_metadata)


def test_likelihood_estimator_workflow():
    """Test the typical workflow for a likelihood estimator."""
    estimator = MockLikelihoodEstimator()

    # Should raise error before fit
    with pytest.raises(ValueError, match="Must call fit"):
        estimator.evaluate(np.array([1.0]), np.array([1]))

    # Fit the estimator
    mock_simulations = {
        "rts": np.array([1.0, 2.0, 3.0]),
        "choices": np.array([1, -1, 1]),
        "metadata": {"max_t": 20.0},
    }
    estimator.fit(mock_simulations)

    # Now evaluate should work
    log_liks = estimator.evaluate(np.array([1.0, 2.0]), np.array([1, -1]))
    assert log_liks.shape == (2,)

    # Sample should work
    samples = estimator.sample(5)
    assert "rts" in samples
    assert "choices" in samples
    assert samples["rts"].shape == (5,)
    assert samples["choices"].shape == (5,)

    # Get metadata should work
    metadata = estimator.get_metadata()
    assert "max_t" in metadata
    assert "possible_choices" in metadata


def test_estimator_builder_protocol_compliance():
    """Test that MockEstimatorBuilder implements the protocol."""
    builder = MockEstimatorBuilder()

    # Verify it has the required method
    assert hasattr(builder, "build")
    assert callable(builder.build)


def test_estimator_builder_workflow():
    """Test the typical workflow for an estimator builder."""
    builder = MockEstimatorBuilder()

    theta = {"v": 1.0, "a": 2.0, "z": 0.5}
    simulations = {
        "rts": np.array([1.0, 2.0, 3.0]),
        "choices": np.array([1, -1, 1]),
        "metadata": {"max_t": 20.0},
    }

    # Build estimator
    estimator = builder.build(theta, simulations)

    # Verify it's a LikelihoodEstimatorProtocol
    assert hasattr(estimator, "fit")
    assert hasattr(estimator, "evaluate")
    assert hasattr(estimator, "sample")
    assert hasattr(estimator, "get_metadata")

    # Verify it's ready to use (should be fitted)
    samples = estimator.sample(5)
    assert samples["rts"].shape == (5,)


def test_training_strategy_protocol_compliance():
    """Test that MockTrainingStrategy implements the protocol."""
    strategy = MockTrainingStrategy()

    # Verify it has the required method
    assert hasattr(strategy, "generate")
    assert callable(strategy.generate)


def test_training_strategy_workflow():
    """Test the typical workflow for a training strategy."""
    # Create components
    builder = MockEstimatorBuilder()
    strategy = MockTrainingStrategy()

    theta = {"v": 1.0, "a": 2.0, "z": 0.5}
    simulations = {
        "rts": np.array([1.0, 2.0, 3.0]),
        "choices": np.array([1, -1, 1]),
        "metadata": {"max_t": 20.0},
    }

    # Build estimator
    estimator = builder.build(theta, simulations)

    # Generate training data
    training_data = strategy.generate(theta, estimator)

    # Verify output structure
    assert training_data.shape[0] == 10  # n_samples
    assert training_data.shape[1] == 6  # n_params (3) + 3

    # Verify parameter columns contain theta values
    assert np.allclose(training_data[0, :3], [1.0, 2.0, 0.5])

    # Verify RT, choice, and log-likelihood columns exist
    assert training_data[:, -3].shape == (10,)  # RTs
    assert training_data[:, -2].shape == (10,)  # choices
    assert training_data[:, -1].shape == (10,)  # log-likelihoods


def test_protocol_type_checking():
    """Test that protocols work with isinstance checks (runtime)."""
    # Note: Protocol is structural typing, so we can't use isinstance directly
    # This test verifies the mock classes have the right structure

    estimator = MockLikelihoodEstimator()
    builder = MockEstimatorBuilder()
    strategy = MockTrainingStrategy()

    # Verify they all have the expected methods
    assert all(
        hasattr(estimator, method)
        for method in ["fit", "evaluate", "sample", "get_metadata"]
    )
    assert hasattr(builder, "build")
    assert hasattr(strategy, "generate")
