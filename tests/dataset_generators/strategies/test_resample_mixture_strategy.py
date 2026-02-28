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
    """Standard generator configuration with nested structure."""
    return {
        "training": {
            "n_training_samples_by_parameter_set": 100,
            "data_mixture_probabilities": [0.8, 0.1, 0.1],
            "separate_response_channels": False,
            "negative_rt_log_likelihood": -1000.0,
        },
        "estimator": {
            "kde_displace_t": False,
        },
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

    n_samples = generator_config["training"]["n_training_samples_by_parameter_set"]
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
    generator_config["training"]["n_training_samples_by_parameter_set"] = (
        1000  # Large n for clear proportions
    )
    strategy = ResampleMixtureStrategy(generator_config, ddm_model_config)

    training_data = strategy.generate(ddm_theta, fitted_estimator)

    # Check RT distribution - negative RTs indicate uniform_down samples
    rt_col = training_data[:, -3]
    n_negative = np.sum(rt_col < 0)

    # Should be approximately 10% (0.1 * 1000)
    expected_negative = int(
        1000 * generator_config["training"]["data_mixture_probabilities"][2]
    )
    assert abs(n_negative - expected_negative) <= 1  # Allow for rounding


def test_strategy_negative_rt_log_likelihood(
    generator_config, ddm_model_config, ddm_theta, fitted_estimator
):
    """Test that negative RT samples have the correct log-likelihood value."""
    strategy = ResampleMixtureStrategy(generator_config, ddm_model_config)

    training_data = strategy.generate(ddm_theta, fitted_estimator)

    # Extract negative RT samples
    rt_col = training_data[:, -3]
    lik_col = training_data[:, -1]

    negative_rt_mask = rt_col < 0
    negative_rt_liks = lik_col[negative_rt_mask]

    # All negative RT log-likelihoods should equal the configured value
    expected_value = generator_config["training"]["negative_rt_log_likelihood"]
    np.testing.assert_allclose(negative_rt_liks, expected_value)


def test_strategy_positive_rt_range(
    generator_config, ddm_model_config, ddm_theta, fitted_estimator
):
    """Test that RTs are in valid range after shuffling."""
    strategy = ResampleMixtureStrategy(generator_config, ddm_model_config)

    training_data = strategy.generate(ddm_theta, fitted_estimator, random_state=42)

    rt_col = training_data[:, -3]

    # After shuffling, we can't identify which samples came from which source
    # But we can verify overall properties:

    # All positive RTs should be >= 0.0001
    positive_rts = rt_col[rt_col >= 0]
    assert np.all(positive_rts >= 0.0001)

    # Negative RTs should be in the range [-1.0, 0.0001)
    negative_rts = rt_col[rt_col < 0]
    assert np.all(negative_rts >= -1.0)
    assert np.all(negative_rts < 0.0001)

    # Check that we have approximately the right number of negative RTs
    n_total = generator_config["training"]["n_training_samples_by_parameter_set"]
    expected_negative = int(
        n_total * generator_config["training"]["data_mixture_probabilities"][2]
    )
    actual_negative = len(negative_rts)
    assert abs(actual_negative - expected_negative) <= 1  # Allow for rounding


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

    # Log-likelihoods for positive RTs are typically <= 0.
    # KDE density can legitimately exceed 1 near distribution peaks,
    # producing small positive log-likelihoods. Allow a small tolerance.
    assert np.all(positive_rt_liks <= 1.0), (
        f"Unexpectedly large positive log-likelihoods found: "
        f"max={positive_rt_liks.max():.4f}"
    )


def test_strategy_with_separate_response_channels(
    generator_config, ddm_model_config, ddm_theta, fitted_estimator
):
    """Test strategy with one-hot encoded choices after shuffling."""
    generator_config["training"]["separate_response_channels"] = True
    strategy = ResampleMixtureStrategy(generator_config, ddm_model_config)

    training_data = strategy.generate(ddm_theta, fitted_estimator, random_state=42)

    n_samples = generator_config["training"]["n_training_samples_by_parameter_set"]
    n_params = len(ddm_theta)
    nchoices = ddm_model_config["nchoices"]
    n_features = 2 + nchoices + n_params  # params + RT + one-hot-choices + log_lik

    assert training_data.shape == (n_samples, n_features)

    # After shuffling, KDE samples with one-hot encoding are mixed with uniform samples
    # that use scalar choice encoding. We need to identify which samples have one-hot encoding.

    # Strategy: One-hot encoded rows will have integer values in the choice columns that sum to 1
    # Uniform samples will have non-integer choice values in column -2
    rt_col_idx = -2 - nchoices
    one_hot_cols = training_data[:, (rt_col_idx + 1) : -1]
    _choice_scalar_col = training_data[:, -2]  # For reference

    # Find rows where one-hot encoding is used (KDE samples)
    # These have choice columns that sum to approximately 1
    row_sums = one_hot_cols.sum(axis=1)
    one_hot_rows = np.abs(row_sums - 1.0) < 0.01  # Tolerance for floating point

    # For one-hot encoded rows, verify structure
    one_hot_samples = one_hot_cols[one_hot_rows]
    if len(one_hot_samples) > 0:
        # Each row should sum to exactly 1
        one_hot_sums = one_hot_samples.sum(axis=1)
        np.testing.assert_allclose(one_hot_sums, 1.0, rtol=1e-5)

        # All values should be 0 or 1
        assert np.all((one_hot_samples == 0) | (one_hot_samples == 1))


def test_strategy_rounding_adjustment(
    generator_config, ddm_model_config, ddm_theta, fitted_estimator
):
    """Test that rounding adjustments work correctly."""
    # Set probabilities that will cause rounding issues
    generator_config["training"]["n_training_samples_by_parameter_set"] = 100
    generator_config["training"]["data_mixture_probabilities"] = [0.333, 0.333, 0.334]

    strategy = ResampleMixtureStrategy(generator_config, ddm_model_config)

    training_data = strategy.generate(ddm_theta, fitted_estimator)

    # Should still generate exactly 100 samples
    assert training_data.shape[0] == 100


def test_strategy_rounding_error_too_large(
    generator_config, ddm_model_config, ddm_theta, fitted_estimator
):
    """Test that strategy raises error if rounding adjustment would make n_kde negative."""
    # Create pathological case
    generator_config["training"]["n_training_samples_by_parameter_set"] = 10
    generator_config["training"]["data_mixture_probabilities"] = [
        0.0,
        0.6,
        0.6,
    ]  # Sums to 1.2!

    strategy = ResampleMixtureStrategy(generator_config, ddm_model_config)

    # This should raise ValueError
    with pytest.raises(ValueError, match="Rounding error too large"):
        strategy.generate(ddm_theta, fitted_estimator)


def test_strategy_reproducibility_with_seed(
    generator_config, ddm_model_config, ddm_theta, fitted_estimator
):
    """Test that strategy generates reproducible results with random_state parameter."""
    strategy = ResampleMixtureStrategy(generator_config, ddm_model_config)

    data1 = strategy.generate(ddm_theta, fitted_estimator, random_state=42)
    data2 = strategy.generate(ddm_theta, fitted_estimator, random_state=42)

    np.testing.assert_allclose(data1, data2)


def test_strategy_different_with_different_seed(
    generator_config, ddm_model_config, ddm_theta, fitted_estimator
):
    """Test that strategy generates different results with different random_state values."""
    strategy = ResampleMixtureStrategy(generator_config, ddm_model_config)

    data1 = strategy.generate(ddm_theta, fitted_estimator, random_state=42)
    data2 = strategy.generate(ddm_theta, fitted_estimator, random_state=123)

    # Should be different
    assert not np.allclose(data1, data2)


def test_strategy_shuffles_output(
    generator_config, ddm_model_config, ddm_theta, fitted_estimator
):
    """Test that strategy shuffles output to avoid ordering bias."""
    # Use large sample size for clear separation of mixture components
    generator_config["training"]["n_training_samples_by_parameter_set"] = 1000
    strategy = ResampleMixtureStrategy(generator_config, ddm_model_config)

    training_data = strategy.generate(ddm_theta, fitted_estimator, random_state=42)

    # Extract RT column
    rt_col = training_data[:, -3]

    # Calculate expected counts
    n_total = 1000
    n_kde = int(
        n_total * generator_config["training"]["data_mixture_probabilities"][0]
    )  # 800
    _n_unif_up = int(
        n_total * generator_config["training"]["data_mixture_probabilities"][1]
    )  # 100
    n_unif_down = int(
        n_total * generator_config["training"]["data_mixture_probabilities"][2]
    )  # 100

    # If NOT shuffled, first n_kde samples would all be KDE samples (mostly positive RTs)
    # If shuffled, negative RTs should be distributed throughout

    # Count negative RTs in first n_kde samples
    first_batch_negative = np.sum(rt_col[:n_kde] < 0)

    # If shuffled properly, we should see some negative RTs in the first batch
    # (roughly 10% of first batch should be negative since 10% overall are negative)
    # We expect ~80 negative RTs in first 800 samples if perfectly shuffled
    # Allow for randomness: should have at least 20 (significantly > 0)
    assert first_batch_negative > 20, (
        f"Expected shuffling to mix negative RTs throughout, "
        f"but found only {first_batch_negative} in first {n_kde} samples"
    )

    # Also verify that not ALL negative RTs are at the end
    last_batch_negative = np.sum(rt_col[-n_unif_down:] < 0)
    assert last_batch_negative < n_unif_down, (
        "Expected shuffling, but all negative RTs appear at the end"
    )


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
        == generator_config["training"]["n_training_samples_by_parameter_set"]
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

    n_samples = generator_config["training"]["n_training_samples_by_parameter_set"]
    n_params = len(theta)
    n_features = 3 + n_params

    assert training_data.shape == (n_samples, n_features)

    # Check choices are valid for ornstein
    choice_col = training_data[:, -2]
    assert set(choice_col[choice_col >= 0]) <= set(
        simulations["metadata"]["possible_choices"]
    )
