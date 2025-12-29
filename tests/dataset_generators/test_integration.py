"""Phase 1 Integration Tests for Refactored DataGenerator.

These tests verify that the refactored DataGenerator with builder injection:
1. Maintains backward compatibility (default KDE behavior)
2. Accepts injected builders and strategies
3. Produces identical output to the original implementation
4. Is ready for PyDDM integration in Phase 3
"""

import numpy as np
import pytest

from ssms.config import model_config
from ssms.config.generator_config.data_generator_config import (
    get_default_generator_config,
)
from ssms.dataset_generators.lan_mlp import DataGenerator
from ssms.dataset_generators.estimator_builders import KDEEstimatorBuilder
from ssms.dataset_generators.strategies import MixtureTrainingStrategy


@pytest.fixture
def ddm_configs():
    """Get DDM model and generator configs."""
    model_conf = model_config["ddm"]

    # Get nested config (always nested now)
    gen_config = get_default_generator_config("lan")

    # Update nested config settings
    gen_config["model"] = "ddm"
    gen_config["pipeline"]["n_parameter_sets"] = (
        10  # Increased to ensure filtering doesn't reject all
    )
    gen_config["simulator"]["n_samples"] = 500
    gen_config["training"]["n_samples_per_param"] = 100

    # Relax simulation filters to avoid rejections
    gen_config["simulator"]["filters"] = {
        "mode": 100,
        "choice_cnt": 0,
        "mean_rt": 100,
        "std": 0.0,
        "mode_cnt_rel": 1.0,
    }

    return gen_config, model_conf


def test_backward_compatibility_default_components(ddm_configs):
    """Test that DataGenerator works without explicit pipeline injection."""
    gen_config, model_conf = ddm_configs

    # Create generator with config dict (auto-creates strategy)
    my_gen = DataGenerator(gen_config, model_conf)

    # Verify default pipeline was created
    assert my_gen._generation_pipeline is not None
    from ssms.dataset_generators.pipelines import SimulationPipeline

    assert isinstance(my_gen._generation_pipeline, SimulationPipeline)

    # Verify data generation still works
    data = my_gen.generate_data_training(save=False, verbose=False)

    assert "lan_data" in data
    assert "lan_labels" in data
    assert data["lan_data"].shape[0] == data["lan_labels"].shape[0]


def test_explicit_strategy_injection(ddm_configs):
    """Test that explicit pipeline injection works."""
    gen_config, model_conf = ddm_configs

    # Create custom strategy explicitly
    from ssms.dataset_generators.pipelines import SimulationPipeline
    from ssms.dataset_generators.estimator_builders import KDEEstimatorBuilder
    from ssms.dataset_generators.strategies import MixtureTrainingStrategy

    estimator_builder = KDEEstimatorBuilder
    training_strategy = MixtureTrainingStrategy

    custom_pipeline = SimulationPipeline(
        generator_config=gen_config,
        model_config=model_conf,
        estimator_builder=estimator_builder,
        training_strategy=training_strategy,
    )

    # Inject into DataGenerator (pass strategy as first arg)
    my_gen = DataGenerator(custom_pipeline, model_conf)

    # Verify injected strategy is used
    assert my_gen._generation_pipeline is custom_pipeline

    # Verify data generation works
    data = my_gen.generate_data_training(save=False, verbose=False)

    assert "lan_data" in data
    assert "lan_labels" in data


@pytest.mark.skip(
    reason="Methods get_simulations() and _generate_training_data() are deprecated/commented out"
)
def test_generate_training_data_method(ddm_configs):
    """Test the new _generate_training_data method directly."""
    gen_config, model_conf = ddm_configs

    my_gen = DataGenerator(gen_config, model_conf)

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


@pytest.mark.skip(reason="Method _make_kde_data() is deprecated/commented out")
def test_make_kde_data_deprecated_wrapper(ddm_configs):
    """Test that _make_kde_data still works as a deprecated wrapper."""
    gen_config, model_conf = ddm_configs

    my_gen = DataGenerator(gen_config, model_conf)

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


@pytest.mark.skip(
    reason="Methods get_simulations(), _make_kde_data(), and _generate_training_data() are deprecated/commented out"
)
def test_output_consistency_across_methods(ddm_configs):
    """Test that _make_kde_data and _generate_training_data produce identical output."""
    gen_config, model_conf = ddm_configs

    my_gen = DataGenerator(gen_config, model_conf)

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

    # Test with displace_t=True (using nested structure)
    gen_config["estimator"]["displace_t"] = True
    estimator_builder = KDEEstimatorBuilder(gen_config)

    assert estimator_builder.displace_t is True

    # Test with displace_t=False (default)
    gen_config["estimator"]["displace_t"] = False
    estimator_builder = KDEEstimatorBuilder(gen_config)

    assert estimator_builder.displace_t is False


@pytest.mark.skip(
    reason="Methods get_simulations() and _generate_training_data() are deprecated/commented out"
)
def test_strategy_mixture_probabilities(ddm_configs):
    """Test that strategy respects mixture probabilities from config."""
    gen_config, model_conf = ddm_configs

    # Set custom mixture probabilities
    gen_config["data_mixture_probabilities"] = [0.7, 0.2, 0.1]
    gen_config["n_training_samples_by_parameter_set"] = 1000

    my_gen = DataGenerator(gen_config, model_conf)

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
    """Test pipeline injection works with different models."""
    gen_config, model_conf = ddm_configs

    # Test with ornstein model
    ornstein_conf = model_config["ornstein"]
    gen_config["model"] = "ornstein"

    # Create custom strategy
    from ssms.dataset_generators.pipelines import SimulationPipeline

    estimator_builder = KDEEstimatorBuilder
    training_strategy = MixtureTrainingStrategy

    custom_pipeline = SimulationPipeline(
        generator_config=gen_config,
        model_config=ornstein_conf,
        estimator_builder=estimator_builder,
        training_strategy=training_strategy,
    )

    # Inject into DataGenerator
    my_gen = DataGenerator(custom_pipeline, ornstein_conf)

    # Verify data generation works
    data = my_gen.generate_data_training(save=False, verbose=False)

    assert "lan_data" in data
    assert "lan_labels" in data


def test_separate_response_channels_with_injection(ddm_configs):
    """Test that separate_response_channels mode works with pipeline injection."""
    gen_config, model_conf = ddm_configs

    # Enable separate response channels
    gen_config["separate_response_channels"] = True

    # Create generator with custom strategy
    from ssms.dataset_generators.pipelines import SimulationPipeline

    estimator_builder = KDEEstimatorBuilder
    training_strategy = MixtureTrainingStrategy

    custom_pipeline = SimulationPipeline(
        generator_config=gen_config,
        model_config=model_conf,
        estimator_builder=estimator_builder,
        training_strategy=training_strategy,
    )

    my_gen = DataGenerator(custom_pipeline, model_conf)

    # Test with actual data generation (methods commented out)
    data = my_gen.generate_data_training(save=False, verbose=False)

    assert "lan_data" in data
    assert "lan_labels" in data


def test_end_to_end_with_custom_components(ddm_configs):
    """Test complete end-to-end workflow with custom injected strategy."""
    gen_config, model_conf = ddm_configs

    # Use many parameter sets to ensure some pass filtering
    gen_config["n_parameter_sets"] = 20

    # Create custom strategy
    from ssms.dataset_generators.pipelines import SimulationPipeline

    estimator_builder = KDEEstimatorBuilder
    training_strategy = MixtureTrainingStrategy

    custom_pipeline = SimulationPipeline(
        generator_config=gen_config,
        model_config=model_conf,
        estimator_builder=estimator_builder,
        training_strategy=training_strategy,
    )

    # Create DataGenerator with injection
    my_gen = DataGenerator(custom_pipeline, model_conf)

    # Generate full training dataset
    data = my_gen.generate_data_training(save=False, verbose=False)

    # Verify output structure
    assert "lan_data" in data
    assert "lan_labels" in data
    assert "cpn_data" in data
    assert "cpn_labels" in data

    # Verify shapes (using nested config)
    n_param_sets = gen_config["pipeline"]["n_parameter_sets"]
    n_training_samples = gen_config["training"]["n_samples_per_param"]
    total_samples = n_param_sets * n_training_samples

    assert data["lan_data"].shape[0] == total_samples
    assert data["lan_labels"].shape[0] == total_samples
    assert data["cpn_data"].shape[0] == n_param_sets


def test_ready_for_pyddm_pattern(ddm_configs):
    """Test that the pattern is ready for PyDDM integration."""
    gen_config, model_conf = ddm_configs

    # This test verifies the pattern works WITHOUT knowing implementation details
    # (i.e., it's truly generic)

    # Create a generator with custom strategy
    from ssms.dataset_generators.pipelines import SimulationPipeline

    estimator_builder = KDEEstimatorBuilder
    training_strategy = MixtureTrainingStrategy

    custom_pipeline = SimulationPipeline(
        generator_config=gen_config,
        model_config=model_conf,
        estimator_builder=estimator_builder,
        training_strategy=training_strategy,
    )

    my_gen = DataGenerator(custom_pipeline, model_conf)

    # Verify the generator uses the injected strategy
    assert my_gen._generation_pipeline is custom_pipeline

    # Generate data
    data = my_gen.generate_data_training(save=False, verbose=False)

    # The fact that this works means we can swap in PyDDM strategy too
    assert "lan_data" in data
    assert isinstance(data["lan_data"], np.ndarray)


def test_pyddm_pattern_actually_works(ddm_configs):
    """Test that PyDDM strategy works end-to-end."""
    pytest.importorskip("pyddm")  # Skip if pyddm not installed

    gen_config, model_conf = ddm_configs

    # Set estimator_type to pyddm - factory will create PyDDMPipeline
    gen_config["estimator_type"] = "pyddm"

    # Test with DataGenerator (auto-creates PyDDM strategy)
    my_gen = DataGenerator(gen_config, model_conf)

    # Verify PyDDM strategy was created
    from ssms.dataset_generators.pipelines import PyDDMPipeline

    assert isinstance(my_gen._generation_pipeline, PyDDMPipeline)

    # Generate training data (should not require simulations for PyDDM)
    training_data_dict = my_gen.generate_data_training(save=False)

    assert "lan_data" in training_data_dict
    lan_data = training_data_dict["lan_data"]
    assert lan_data.shape[0] > 0
    assert np.all(np.isfinite(lan_data[:, -1]))
