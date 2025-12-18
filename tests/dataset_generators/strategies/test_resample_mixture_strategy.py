"""Tests for Resample Mixture Strategy.

These tests verify that ResampleMixtureStrategy correctly:
1. Generates training data with the correct shape and structure
2. Mixes samples from the estimator with uniform samples
3. Handles separate response channels correctly
4. Implements TrainingDataStrategyProtocol
"""

import numpy as np
import pytest

from ssms.basic_simulators.simulator import simulator
from ssms.config import model_config
from ssms.dataset_generators.estimator_builders import KDEEstimatorBuilder
from ssms.dataset_generators.strategies import ResampleMixtureStrategy


@pytest.fixture
def ddm_theta():
    """Sample DDM parameters."""
    return {"v": 1.0, "a": 2.0, "z": 0.5, "t": 0.3}


@pytest.fixture
def ddm_simulations(ddm_theta):
    """Generate DDM simulations for testing."""
    return simulator(
        theta=ddm_theta,
        model="ddm",
        n_samples=1000,
        max_t=5.0,
        delta_t=0.001,
        smooth_unif=False,
        random_state=42,
    )


@pytest.fixture
def generator_config():
    """Standard generator configuration."""
    return {
        "n_training_samples_by_parameter_set": 100,
        "kde_data_mixture_probabilities": [0.8, 0.1, 0.1],
        "separate_response_channels": False,
        "negative_rt_cutoff": -1000.0,
        "kde_displace_t": False,
    }


@pytest.fixture
def ddm_model_config():
    """DDM model configuration."""
    return model_config["ddm"]


@pytest.fixture
def fitted_estimator(generator_config, ddm_theta, ddm_simulations):
    """Build a fitted KDE estimator."""
    builder = KDEEstimatorBuilder(generator_config)
    return builder.build(ddm_theta, ddm_simulations)


def test_strategy_initialization(generator_config, ddm_model_config):
    """Test that strategy initializes correctly."""
    strategy = ResampleMixtureStrategy(generator_config, ddm_model_config)

    assert strategy.generator_config == generator_config
    assert strategy.model_config == ddm_model_config


def test_strategy_generates_correct_shape(
    generator_config, ddm_model_config, ddm_theta, fitted_estimator
):
    """Test that generate() returns array with correct shape."""
    strategy = ResampleMixtureStrategy(generator_config, ddm_model_config)

    training_data = strategy.generate(ddm_theta, fitted_estimator)

    n_samples = generator_config["n_training_samples_by_parameter_set"]
    n_params = len(ddm_theta)
    n_features = 3 + n_params  # params + RT + choice + log_lik

    assert isinstance(training_data, np.ndarray)
    assert training_data.shape == (n_samples, n_features)
    assert training_data.dtype == np.float32


def test_strategy_fills_theta_parameters(
    generator_config, ddm_model_config, ddm_theta, fitted_estimator
):
    """Test that theta parameters are correctly filled in all rows."""
    strategy = ResampleMixtureStrategy(generator_config, ddm_model_config)

    training_data = strategy.generate(ddm_theta, fitted_estimator)

    # Extract theta columns
    n_params = len(ddm_theta)
    theta_cols = training_data[:, :n_params]

    # All rows should have the same theta values
    expected_theta = np.array([ddm_theta[key] for key in ddm_model_config["params"]])
    np.testing.assert_allclose(theta_cols[0], expected_theta)
    np.testing.assert_allclose(theta_cols[-1], expected_theta)

    # All rows should be identical in theta columns
    assert np.allclose(theta_cols, theta_cols[0])


def test_strategy_mixture_proportions(
    generator_config, ddm_model_config, ddm_theta, fitted_estimator
):
    """Test that samples are mixed according to specified probabilities."""
    generator_config["n_training_samples_by_parameter_set"] = (
        1000  # Large n for clear proportions
    )
    strategy = ResampleMixtureStrategy(generator_config, ddm_model_config)

    training_data = strategy.generate(ddm_theta, fitted_estimator)

    # Check RT distribution - negative RTs indicate uniform_down samples
    rt_col = training_data[:, -3]
    n_negative = np.sum(rt_col < 0)

    # Should be approximately 10% (0.1 * 1000)
    expected_negative = int(
        1000 * generator_config["kde_data_mixture_probabilities"][2]
    )
    assert abs(n_negative - expected_negative) <= 1  # Allow for rounding


def test_strategy_negative_rt_cutoff(
    generator_config, ddm_model_config, ddm_theta, fitted_estimator
):
    """Test that negative RT samples have the correct log-likelihood cutoff."""
    strategy = ResampleMixtureStrategy(generator_config, ddm_model_config)

    training_data = strategy.generate(ddm_theta, fitted_estimator)

    # Extract negative RT samples
    rt_col = training_data[:, -3]
    lik_col = training_data[:, -1]

    negative_rt_mask = rt_col < 0
    negative_rt_liks = lik_col[negative_rt_mask]

    # All negative RT log-likelihoods should equal the cutoff
    expected_cutoff = generator_config["negative_rt_cutoff"]
    np.testing.assert_allclose(negative_rt_liks, expected_cutoff)


def test_strategy_positive_rt_range(
    generator_config, ddm_model_config, ddm_theta, fitted_estimator
):
    """Test that uniform samples' RTs are in valid range."""
    strategy = ResampleMixtureStrategy(generator_config, ddm_model_config)

    training_data = strategy.generate(ddm_theta, fitted_estimator)

    # Get uniform sample indices (KDE samples can exceed max_t due to extrapolation)
    n_kde = int(
        generator_config["n_training_samples_by_parameter_set"]
        * generator_config["kde_data_mixture_probabilities"][0]
    )
    n_unif_up = int(
        generator_config["n_training_samples_by_parameter_set"]
        * generator_config["kde_data_mixture_probabilities"][1]
    )

    rt_col = training_data[:, -3]
    uniform_rts = rt_col[n_kde : (n_kde + n_unif_up)]

    # Uniform RTs should be between 0.0001 and max_t
    metadata = fitted_estimator.get_metadata()
    max_t = min(metadata["max_t"], 100)  # Strategy clips at 100

    assert np.all(uniform_rts >= 0.0001)
    assert np.all(uniform_rts <= max_t)

    # All positive RTs should be >= 0.0001 (including KDE samples)
    positive_rts = rt_col[rt_col >= 0]
    assert np.all(positive_rts >= 0.0001)


def test_strategy_choices_are_valid(
    generator_config, ddm_model_config, ddm_theta, fitted_estimator
):
    """Test that all choices are valid."""
    strategy = ResampleMixtureStrategy(generator_config, ddm_model_config)

    training_data = strategy.generate(ddm_theta, fitted_estimator)

    choice_col = training_data[:, -2]
    metadata = fitted_estimator.get_metadata()

    # All choices should be in possible_choices
    for choice in choice_col:
        assert choice in metadata["possible_choices"]


def test_strategy_log_likelihoods_are_reasonable(
    generator_config, ddm_model_config, ddm_theta, fitted_estimator
):
    """Test that log-likelihoods are reasonable (not NaN, mostly negative)."""
    strategy = ResampleMixtureStrategy(generator_config, ddm_model_config)

    training_data = strategy.generate(ddm_theta, fitted_estimator)

    lik_col = training_data[:, -1]

    # No NaNs or infs
    assert not np.any(np.isnan(lik_col))
    assert not np.any(np.isinf(lik_col))

    # Most should be negative (except possibly negative RT cutoff)
    rt_col = training_data[:, -3]
    positive_rt_mask = rt_col >= 0
    positive_rt_liks = lik_col[positive_rt_mask]

    # Log-likelihoods for positive RTs should be <= 0
    assert np.all(positive_rt_liks <= 0)


def test_strategy_with_separate_response_channels(
    generator_config, ddm_model_config, ddm_theta, fitted_estimator
):
    """Test strategy with one-hot encoded choices."""
    generator_config["separate_response_channels"] = True
    strategy = ResampleMixtureStrategy(generator_config, ddm_model_config)

    training_data = strategy.generate(ddm_theta, fitted_estimator)

    n_samples = generator_config["n_training_samples_by_parameter_set"]
    n_params = len(ddm_theta)
    nchoices = ddm_model_config["nchoices"]
    n_features = 2 + nchoices + n_params  # params + RT + one-hot-choices + log_lik

    assert training_data.shape == (n_samples, n_features)

    # Check that one-hot encoding is valid for KDE samples
    # (Uniform samples don't use one-hot encoding, even with separate_response_channels)
    n_kde = int(
        generator_config["n_training_samples_by_parameter_set"]
        * generator_config["kde_data_mixture_probabilities"][0]
    )

    # Extract one-hot columns (between RT and log-lik) for KDE samples only
    rt_col_idx = -2 - nchoices
    one_hot_cols = training_data[:n_kde, (rt_col_idx + 1) : -1]

    # Each KDE sample row should sum to 1 (exactly one choice active)
    row_sums = one_hot_cols.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0)

    # All values should be 0 or 1
    assert np.all((one_hot_cols == 0) | (one_hot_cols == 1))


def test_strategy_rounding_adjustment(
    generator_config, ddm_model_config, ddm_theta, fitted_estimator
):
    """Test that rounding adjustments work correctly."""
    # Set probabilities that will cause rounding issues
    generator_config["n_training_samples_by_parameter_set"] = 100
    generator_config["kde_data_mixture_probabilities"] = [0.333, 0.333, 0.334]

    strategy = ResampleMixtureStrategy(generator_config, ddm_model_config)

    training_data = strategy.generate(ddm_theta, fitted_estimator)

    # Should still generate exactly 100 samples
    assert training_data.shape[0] == 100


def test_strategy_rounding_error_too_large(
    generator_config, ddm_model_config, ddm_theta, fitted_estimator
):
    """Test that strategy raises error if rounding adjustment would make n_kde negative."""
    # Create pathological case
    generator_config["n_training_samples_by_parameter_set"] = 10
    generator_config["kde_data_mixture_probabilities"] = [0.0, 0.6, 0.6]  # Sums to 1.2!

    strategy = ResampleMixtureStrategy(generator_config, ddm_model_config)

    # This should raise ValueError
    with pytest.raises(ValueError, match="Rounding error too large"):
        strategy.generate(ddm_theta, fitted_estimator)


def test_strategy_reproducibility_with_seed(
    generator_config, ddm_model_config, ddm_theta, fitted_estimator
):
    """Test that strategy generates reproducible results with seeded RNG."""
    strategy = ResampleMixtureStrategy(generator_config, ddm_model_config)

    np.random.seed(42)
    data1 = strategy.generate(ddm_theta, fitted_estimator)

    np.random.seed(42)
    data2 = strategy.generate(ddm_theta, fitted_estimator)

    np.testing.assert_allclose(data1, data2)


def test_strategy_different_with_different_seed(
    generator_config, ddm_model_config, ddm_theta, fitted_estimator
):
    """Test that strategy generates different results with different seeds."""
    strategy = ResampleMixtureStrategy(generator_config, ddm_model_config)

    np.random.seed(42)
    data1 = strategy.generate(ddm_theta, fitted_estimator)

    np.random.seed(123)
    data2 = strategy.generate(ddm_theta, fitted_estimator)

    # Should be different
    assert not np.allclose(data1, data2)


def test_strategy_protocol_compliance(
    generator_config, ddm_model_config, ddm_theta, fitted_estimator
):
    """Test that ResampleMixtureStrategy implements TrainingDataStrategyProtocol."""
    strategy = ResampleMixtureStrategy(generator_config, ddm_model_config)

    # Verify required method exists and is callable
    assert hasattr(strategy, "generate")
    assert callable(strategy.generate)

    # Verify it works in the protocol workflow
    training_data = strategy.generate(ddm_theta, fitted_estimator)

    assert isinstance(training_data, np.ndarray)
    assert (
        training_data.shape[0]
        == generator_config["n_training_samples_by_parameter_set"]
    )
    assert training_data.dtype == np.float32


def test_strategy_with_different_model(generator_config):
    """Test that strategy works with different models (e.g., ornstein)."""
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

    ornstein_model_config = model_config["ornstein"]

    builder = KDEEstimatorBuilder(generator_config)
    estimator = builder.build(theta, simulations)

    strategy = ResampleMixtureStrategy(generator_config, ornstein_model_config)
    training_data = strategy.generate(theta, estimator)

    n_samples = generator_config["n_training_samples_by_parameter_set"]
    n_params = len(theta)
    n_features = 3 + n_params

    assert training_data.shape == (n_samples, n_features)

    # Check choices are valid for ornstein
    choice_col = training_data[:, -2]
    assert set(choice_col[choice_col >= 0]) <= set(
        simulations["metadata"]["possible_choices"]
    )
