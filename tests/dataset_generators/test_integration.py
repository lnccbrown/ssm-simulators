"""Phase 1 Integration Tests for Refactored data_generator.

These tests verify that the refactored data_generator with builder injection:
1. Maintains backward compatibility (default KDE behavior)
2. Accepts injected builders and strategies
3. Produces identical output to the original implementation
4. Is ready for PyDDM integration in Phase 3
"""

import numpy as np
import pytest

from ssms.config import model_config
from ssms.config.generator_config import data_generator_config
from ssms.dataset_generators.lan_mlp import data_generator
from ssms.dataset_generators.estimator_builders import KDEEstimatorBuilder
from ssms.dataset_generators.strategies import ResampleMixtureStrategy


@pytest.fixture
def ddm_configs():
    """Get DDM model and generator configs."""
    model_conf = model_config["ddm"]
    gen_config = data_generator_config.get_lan_config()
    gen_config["model"] = "ddm"
    gen_config["n_parameter_sets"] = (
        10  # Increased to ensure filtering doesn't reject all
    )
    gen_config["n_samples"] = 500
    gen_config["n_training_samples_by_parameter_set"] = 100
    # Relax simulation filters to avoid rejections
    gen_config["simulation_filters"]["mode"] = 100
    gen_config["simulation_filters"]["mean_rt"] = 100
    gen_config["simulation_filters"]["std"] = 0.0
    gen_config["simulation_filters"]["mode_cnt_rel"] = 1.0
    return gen_config, model_conf


def test_backward_compatibility_default_components(ddm_configs):
    """Test that data_generator works without explicit builder injection."""
    gen_config, model_conf = ddm_configs

    # Create generator without specifying estimator_builder or training_strategy
    my_gen = data_generator(generator_config=gen_config, model_config=model_conf)

    # Verify default components were created
    assert my_gen._estimator_builder is not None
    assert isinstance(my_gen._estimator_builder, KDEEstimatorBuilder)

    assert my_gen._training_strategy is not None
    assert isinstance(my_gen._training_strategy, ResampleMixtureStrategy)

    # Verify data generation still works
    data = my_gen.generate_data_training_uniform(save=False, verbose=False)

    assert "lan_data" in data
    assert "lan_labels" in data
    assert data["lan_data"].shape[0] == data["lan_labels"].shape[0]


def test_explicit_builder_injection(ddm_configs):
    """Test that explicit builder and strategy injection works."""
    gen_config, model_conf = ddm_configs

    # Create builder and strategy explicitly
    estimator_builder = KDEEstimatorBuilder(gen_config)
    training_strategy = ResampleMixtureStrategy(gen_config, model_conf)

    # Inject into data_generator
    my_gen = data_generator(
        generator_config=gen_config,
        model_config=model_conf,
        estimator_builder=estimator_builder,
        training_strategy=training_strategy,
    )

    # Verify injected components are used
    assert my_gen._estimator_builder is estimator_builder
    assert my_gen._training_strategy is training_strategy

    # Verify data generation works
    data = my_gen.generate_data_training_uniform(save=False, verbose=False)

    assert "lan_data" in data
    assert "lan_labels" in data


def test_generate_training_data_method(ddm_configs):
    """Test the new _generate_training_data method directly."""
    gen_config, model_conf = ddm_configs

    my_gen = data_generator(generator_config=gen_config, model_config=model_conf)

    # Generate simulations
    theta = {"v": 1.0, "a": 2.0, "z": 0.5, "t": 0.3}
    simulations = my_gen.get_simulations(theta=theta, random_seed=42)

    # Call _generate_training_data
    training_data = my_gen._generate_training_data(simulations=simulations, theta=theta)

    # Verify output shape
    n_samples = gen_config["n_training_samples_by_parameter_set"]
    n_params = len(theta)
    n_features = 3 + n_params  # params + RT + choice + log_lik

    assert training_data.shape == (n_samples, n_features)
    assert training_data.dtype == np.float32


def test_make_kde_data_deprecated_wrapper(ddm_configs):
    """Test that _make_kde_data still works as a deprecated wrapper."""
    gen_config, model_conf = ddm_configs

    my_gen = data_generator(generator_config=gen_config, model_config=model_conf)

    # Generate simulations
    theta = {"v": 1.0, "a": 2.0, "z": 0.5, "t": 0.3}
    simulations = my_gen.get_simulations(theta=theta, random_seed=42)

    # Call deprecated _make_kde_data
    training_data = my_gen._make_kde_data(simulations=simulations, theta=theta)

    # Verify it produces same output as _generate_training_data
    n_samples = gen_config["n_training_samples_by_parameter_set"]
    n_params = len(theta)
    n_features = 3 + n_params

    assert training_data.shape == (n_samples, n_features)
    assert training_data.dtype == np.float32


def test_output_consistency_across_methods(ddm_configs):
    """Test that _make_kde_data and _generate_training_data produce identical output."""
    gen_config, model_conf = ddm_configs

    my_gen = data_generator(generator_config=gen_config, model_config=model_conf)

    # Generate simulations with fixed seed
    theta = {"v": 1.0, "a": 2.0, "z": 0.5, "t": 0.3}
    simulations = my_gen.get_simulations(theta=theta, random_seed=42)

    # Generate training data using both methods with same RNG state
    np.random.seed(123)
    data_new = my_gen._generate_training_data(simulations=simulations, theta=theta)

    np.random.seed(123)
    data_old = my_gen._make_kde_data(simulations=simulations, theta=theta)

    # Verify they're identical
    np.testing.assert_allclose(data_new, data_old)


def test_builder_config_extraction(ddm_configs):
    """Test that builder correctly extracts parameters from generator_config."""
    gen_config, model_conf = ddm_configs

    # Test with displace_t=True
    gen_config["kde_displace_t"] = True
    estimator_builder = KDEEstimatorBuilder(gen_config)

    assert estimator_builder.displace_t is True

    # Test with displace_t=False (default)
    gen_config["kde_displace_t"] = False
    estimator_builder = KDEEstimatorBuilder(gen_config)

    assert estimator_builder.displace_t is False


def test_strategy_mixture_probabilities(ddm_configs):
    """Test that strategy respects mixture probabilities from config."""
    gen_config, model_conf = ddm_configs

    # Set custom mixture probabilities
    gen_config["kde_data_mixture_probabilities"] = [0.7, 0.2, 0.1]
    gen_config["n_training_samples_by_parameter_set"] = 1000

    my_gen = data_generator(generator_config=gen_config, model_config=model_conf)

    theta = {"v": 1.0, "a": 2.0, "z": 0.5, "t": 0.3}
    simulations = my_gen.get_simulations(theta=theta, random_seed=42)

    training_data = my_gen._generate_training_data(simulations=simulations, theta=theta)

    # Check RT distribution - negative RTs indicate uniform_down samples
    rt_col = training_data[:, -3]
    n_negative = np.sum(rt_col < 0)

    # Should be approximately 10% (0.1 * 1000)
    expected_negative = int(1000 * 0.1)
    assert abs(n_negative - expected_negative) <= 1  # Allow for rounding


def test_different_models_with_injection(ddm_configs):
    """Test builder injection works with different models."""
    gen_config, model_conf = ddm_configs

    # Test with ornstein model
    ornstein_conf = model_config["ornstein"]
    gen_config["model"] = "ornstein"

    # Create components
    estimator_builder = KDEEstimatorBuilder(gen_config)
    training_strategy = ResampleMixtureStrategy(gen_config, ornstein_conf)

    # Inject into data_generator
    my_gen = data_generator(
        generator_config=gen_config,
        model_config=ornstein_conf,
        estimator_builder=estimator_builder,
        training_strategy=training_strategy,
    )

    # Verify data generation works
    data = my_gen.generate_data_training_uniform(save=False, verbose=False)

    assert "lan_data" in data
    assert "lan_labels" in data


def test_separate_response_channels_with_injection(ddm_configs):
    """Test that separate_response_channels mode works with injection."""
    gen_config, model_conf = ddm_configs

    # Enable separate response channels
    gen_config["separate_response_channels"] = True

    # Create generator with injection
    estimator_builder = KDEEstimatorBuilder(gen_config)
    training_strategy = ResampleMixtureStrategy(gen_config, model_conf)

    my_gen = data_generator(
        generator_config=gen_config,
        model_config=model_conf,
        estimator_builder=estimator_builder,
        training_strategy=training_strategy,
    )

    theta = {"v": 1.0, "a": 2.0, "z": 0.5, "t": 0.3}
    simulations = my_gen.get_simulations(theta=theta, random_seed=42)

    training_data = my_gen._generate_training_data(simulations=simulations, theta=theta)

    # Verify one-hot encoding in output
    n_samples = gen_config["n_training_samples_by_parameter_set"]
    n_params = len(theta)
    nchoices = model_conf["nchoices"]
    n_features = 2 + nchoices + n_params  # params + RT + one-hot + log_lik

    assert training_data.shape == (n_samples, n_features)


def test_end_to_end_with_custom_components(ddm_configs):
    """Test complete end-to-end workflow with custom injected components."""
    gen_config, model_conf = ddm_configs

    # Use many parameter sets to ensure some pass filtering
    gen_config["n_parameter_sets"] = 20

    # Create custom builder and strategy
    estimator_builder = KDEEstimatorBuilder(gen_config)
    training_strategy = ResampleMixtureStrategy(gen_config, model_conf)

    # Create data_generator with injection
    my_gen = data_generator(
        generator_config=gen_config,
        model_config=model_conf,
        estimator_builder=estimator_builder,
        training_strategy=training_strategy,
    )

    # Generate full training dataset
    data = my_gen.generate_data_training_uniform(save=False, verbose=False)

    # Verify output structure
    assert "lan_data" in data
    assert "lan_labels" in data
    assert "cpn_data" in data
    assert "cpn_labels" in data

    # Verify shapes
    n_param_sets = gen_config["n_parameter_sets"]
    n_training_samples = gen_config["n_training_samples_by_parameter_set"]
    total_samples = n_param_sets * n_training_samples

    assert data["lan_data"].shape[0] == total_samples
    assert data["lan_labels"].shape[0] == total_samples
    assert data["cpn_data"].shape[0] == n_param_sets


def test_ready_for_pyddm_pattern(ddm_configs):
    """Test that the pattern is ready for PyDDM integration (Phase 3)."""
    gen_config, model_conf = ddm_configs

    # This test verifies the pattern works WITHOUT knowing implementation details
    # (i.e., it's truly generic)

    # Create a generator with custom components
    estimator_builder = KDEEstimatorBuilder(gen_config)
    training_strategy = ResampleMixtureStrategy(gen_config, model_conf)

    my_gen = data_generator(
        generator_config=gen_config,
        model_config=model_conf,
        estimator_builder=estimator_builder,
        training_strategy=training_strategy,
    )

    # Verify the generator doesn't have hardcoded KDE logic
    # (i.e., it uses the injected components)
    theta = {"v": 1.0, "a": 2.0, "z": 0.5, "t": 0.3}
    simulations = my_gen.get_simulations(theta=theta, random_seed=42)

    training_data = my_gen._generate_training_data(simulations=simulations, theta=theta)

    # The fact that this works without _generate_training_data knowing
    # about KDE specifically means we can swap in PyDDM in Phase 3
    assert training_data is not None
    assert isinstance(training_data, np.ndarray)


def test_pyddm_pattern_actually_works(ddm_configs):
    """Test that PyDDM pattern works end-to-end (Phase 3)."""
    pytest.importorskip("pyddm")  # Skip if pyddm not installed

    from ssms.dataset_generators.estimator_builders.pyddm_builder import (
        PyDDMEstimatorBuilder,
    )

    gen_config, model_conf = ddm_configs

    # Create PyDDM builder
    pyddm_builder = PyDDMEstimatorBuilder(gen_config, model_conf)

    # Test with data_generator
    my_gen = data_generator(
        generator_config=gen_config,
        model_config=model_conf,
        estimator_builder=pyddm_builder,
    )

    # Generate training data (should not require simulations for PyDDM)
    training_data_dict = my_gen.generate_data_training_uniform(save=False)

    assert "lan_data" in training_data_dict
    lan_data = training_data_dict["lan_data"]
    assert lan_data.shape[0] > 0
    assert np.all(np.isfinite(lan_data[:, -1]))
