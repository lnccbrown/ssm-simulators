"""Comprehensive tests for TrainingDataGenerator class.

This module tests error handling, validation, and edge cases for TrainingDataGenerator.
"""

import pytest
import numpy as np

from ssms.config import model_config
from ssms.config.generator_config.data_generator_config import (
    get_default_generator_config,
)
from ssms.dataset_generators.lan_mlp import TrainingDataGenerator
from ssms.dataset_generators.pipelines import SimulationPipeline
from ssms.dataset_generators.estimator_builders import KDEEstimatorBuilder
from ssms.dataset_generators.strategies import MixtureTrainingStrategy


class TestTrainingDataGeneratorInitialization:
    """Test TrainingDataGenerator initialization and validation."""

    def test_init_with_none_config_raises_error(self):
        """Test that passing None as config raises ValueError."""
        with pytest.raises(ValueError, match="No config specified"):
            TrainingDataGenerator(config=None)

    def test_init_with_dict_config_but_no_model_config_raises_error(self):
        """Test that dict config without model_config raises ValueError."""
        gen_config = get_default_generator_config("lan")

        with pytest.raises(
            ValueError,
            match="model_config is required when config is a dictionary",
        ):
            TrainingDataGenerator(config=gen_config, model_config=None)

    def test_init_with_flat_config_raises_error(self):
        """Test that flat generator_config structure is rejected."""
        flat_config = {
            "n_parameter_sets": 100,
            "estimator_type": "kde",
            "delta_t": 0.001,
        }

        with pytest.raises(
            ValueError,
            match="Flat generator_config structure is no longer supported",
        ):
            TrainingDataGenerator(flat_config, model_config["ddm"])

    def test_init_with_nested_config_succeeds(self):
        """Test successful initialization with nested config."""
        gen_config = get_default_generator_config("lan")
        gen_config["pipeline"]["n_cpus"] = 1  # Avoid psutil issues in tests
        gen = TrainingDataGenerator(gen_config, model_config["ddm"])

        assert gen.generator_config is not None
        assert gen.model_config is not None
        assert gen._generation_pipeline is not None

    def test_init_with_pipeline_no_model_config(self):
        """Test initialization with pipeline object without explicit model_config."""
        gen_config = get_default_generator_config("lan")
        gen_config["pipeline"]["n_parameter_sets"] = 5
        gen_config["pipeline"]["n_cpus"] = 1  # Avoid psutil issues in tests
        gen_config["simulator"]["n_samples"] = 100

        pipeline = SimulationPipeline(
            generator_config=gen_config,
            model_config=model_config["ddm"],
            estimator_builder=KDEEstimatorBuilder,
            training_strategy=MixtureTrainingStrategy,
        )

        # Should extract model_config from pipeline
        gen = TrainingDataGenerator(config=pipeline)

        assert gen.model_config == model_config["ddm"]
        assert gen._generation_pipeline is pipeline

    def test_init_with_pipeline_and_explicit_model_config(self, caplog):
        """Test that pipeline's model_config is used when both are provided."""
        gen_config = get_default_generator_config("lan")
        gen_config["pipeline"]["n_parameter_sets"] = 5
        gen_config["pipeline"]["n_cpus"] = 1  # Avoid psutil issues in tests

        # Create pipeline with ddm
        pipeline = SimulationPipeline(
            generator_config=gen_config,
            model_config=model_config["ddm"],
            estimator_builder=KDEEstimatorBuilder,
            training_strategy=MixtureTrainingStrategy,
        )

        # Try to override with ornstein (should be ignored with warning)
        gen = TrainingDataGenerator(
            config=pipeline, model_config=model_config["ornstein"]
        )

        # Should use pipeline's model_config (ddm), NOT the explicit one (ornstein)
        assert gen.model_config == model_config["ddm"]

        # Should have logged a warning
        assert "pipeline's model_config will be used" in caplog.text


class TestTrainingDataGeneratorDataGeneration:
    """Test TrainingDataGenerator data generation functionality."""

    @pytest.fixture
    def minimal_config(self):
        """Create minimal generator config for fast testing."""
        gen_config = get_default_generator_config("lan")
        gen_config["pipeline"]["n_parameter_sets"] = 10
        gen_config["pipeline"]["n_cpus"] = 1  # Avoid multiprocessing issues in tests
        gen_config["simulator"]["n_samples"] = 1000
        gen_config["training"]["n_samples_per_param"] = 100
        gen_config["simulator"]["max_t"] = 20.0
        # Relaxed filters to ensure tests pass reliably
        gen_config["simulator"]["filters"] = {
            "mode": 17,
            "choice_cnt": 0,
            "mean_rt": 100,
            "std": 0.0,
            "mode_cnt_rel": 0.75,
        }
        return gen_config

    def test_generate_data_training_basic(self, minimal_config):
        """Test basic data generation with default settings."""
        gen = TrainingDataGenerator(minimal_config, model_config["ddm"])
        data = gen.generate_data_training(save=False, verbose=False)

        # Verify output structure
        assert "lan_data" in data
        assert "lan_labels" in data
        assert isinstance(data["lan_data"], np.ndarray)
        assert isinstance(data["lan_labels"], np.ndarray)

        # Verify data has correct shape
        n_params = len(model_config["ddm"]["params"])
        assert data["lan_data"].shape[1] == n_params + 2  # params + RT + choice

    def test_generate_data_training_with_save(self, minimal_config, tmp_path):
        """Test data generation with save=True."""
        minimal_config["output"]["folder"] = str(tmp_path)

        gen = TrainingDataGenerator(minimal_config, model_config["ddm"])
        data = gen.generate_data_training(save=True, verbose=False)

        # Verify data was saved
        output_files = list(tmp_path.glob("*.pickle"))
        assert len(output_files) > 0

        # Verify data structure
        assert "lan_data" in data
        assert data["lan_data"].shape[0] > 0

    def test_generate_data_with_different_models(self, minimal_config):
        """Test data generation with different models."""
        models_to_test = ["ddm", "ornstein", "angle", "levy"]

        for model_name in models_to_test:
            gen = TrainingDataGenerator(minimal_config, model_config[model_name])
            data = gen.generate_data_training(save=False, verbose=False)

            n_params = len(model_config[model_name]["params"])
            assert data["lan_data"].shape[1] == n_params + 2

    def test_generate_data_parameter_validation(self, minimal_config):
        """Test that generated parameters are within bounds."""
        gen = TrainingDataGenerator(minimal_config, model_config["ddm"])
        data = gen.generate_data_training(save=False, verbose=False)

        # Extract parameters
        lan_data = data["lan_data"]
        param_bounds = model_config["ddm"]["param_bounds_dict"]

        # Verify each parameter is within bounds
        for i, param_name in enumerate(model_config["ddm"]["params"]):
            lower, upper = param_bounds[param_name]
            param_values = lan_data[:, i]
            assert np.all(param_values >= lower)
            assert np.all(param_values <= upper)

    def test_generate_data_rt_and_choice_validity(self, minimal_config):
        """Test that generated RTs and choices are valid."""
        gen = TrainingDataGenerator(minimal_config, model_config["ddm"])
        data = gen.generate_data_training(save=False, verbose=False)

        lan_data = data["lan_data"]
        n_params = len(model_config["ddm"]["params"])

        # RT column (note: can include negative RTs from mixture strategy)
        rt_col = n_params
        assert np.all(np.isfinite(lan_data[:, rt_col]))

        # Choice column
        n_params = len(model_config["ddm"]["params"])
        choice_col = n_params + 1
        expected_choices = model_config["ddm"]["choices"]
        assert np.all(np.isin(lan_data[:, choice_col], expected_choices))

    def test_generate_data_likelihood_validity(self, minimal_config):
        """Test that generated likelihoods are valid."""
        gen = TrainingDataGenerator(minimal_config, model_config["ddm"])
        data = gen.generate_data_training(save=False, verbose=False)

        lan_labels = data["lan_labels"]

        # Likelihoods should be finite
        assert np.all(np.isfinite(lan_labels))

        # Log-likelihoods should typically be negative
        # (allowing some numerical edge cases)
        assert np.median(lan_labels) < 0


class TestTrainingDataGeneratorPipelineIntegration:
    """Test TrainingDataGenerator integration with custom pipelines."""

    @pytest.fixture
    def fast_config(self):
        """Create fast config for testing."""
        gen_config = get_default_generator_config("lan")
        gen_config["pipeline"]["n_parameter_sets"] = 10  # Increased for reliability
        gen_config["pipeline"]["n_subruns"] = (
            1  # Must be <= n_parameter_sets to avoid empty batches
        )
        gen_config["pipeline"]["n_cpus"] = 1  # Avoid multiprocessing issues in tests
        gen_config["simulator"]["n_samples"] = 500
        gen_config["training"]["n_samples_per_param"] = 50
        gen_config["simulator"]["max_t"] = 20.0
        # Use filters validated by diagnostic test (100% success rate)
        gen_config["simulator"]["filters"] = {
            "mode": 20,
            "choice_cnt": 0,
            "mean_rt": 100,
            "std": 0.0,
            "mode_cnt_rel": 0.95,
        }
        return gen_config

    def test_custom_pipeline_injection(self, fast_config):
        """Test that custom pipeline is correctly used."""
        custom_pipeline = SimulationPipeline(
            generator_config=fast_config,
            model_config=model_config["ddm"],
            estimator_builder=KDEEstimatorBuilder,
            training_strategy=MixtureTrainingStrategy,
        )

        gen = TrainingDataGenerator(config=custom_pipeline)

        # Verify pipeline was injected
        assert gen._generation_pipeline is custom_pipeline

        # Verify it works - actually generate data
        data = gen.generate_data_training(save=False, verbose=False)
        assert "lan_data" in data
        assert data["lan_data"].shape[0] > 0

    def test_auto_pipeline_creation(self, fast_config):
        """Test that pipeline is auto-created when dict config is provided."""
        gen = TrainingDataGenerator(fast_config, model_config["ddm"])

        # Should have auto-created a pipeline
        assert gen._generation_pipeline is not None
        assert hasattr(gen._generation_pipeline, "generate_for_parameter_set")
        assert isinstance(gen._generation_pipeline, SimulationPipeline)

        # Verify it works - actually generate data
        data = gen.generate_data_training(save=False, verbose=False)
        assert "lan_data" in data
        assert data["lan_data"].shape[0] > 0

    def test_kde_vs_pyddm_pipeline_selection(self, fast_config):
        """Test that correct pipeline is selected based on estimator type."""
        pytest.importorskip("pyddm")  # Skip if pyddm not installed

        # Use deepcopy to avoid mutating the fixture
        from copy import deepcopy
        from ssms.dataset_generators.pipelines import PyDDMPipeline

        # KDE config
        kde_config = deepcopy(fast_config)
        kde_config["estimator"]["type"] = "kde"

        gen_kde = TrainingDataGenerator(kde_config, model_config["ddm"])
        assert isinstance(gen_kde._generation_pipeline, SimulationPipeline)

        # Verify KDE pipeline works
        data_kde = gen_kde.generate_data_training(save=False, verbose=False)
        assert "lan_data" in data_kde
        assert data_kde["lan_data"].shape[0] > 0

        # PyDDM config
        pyddm_config = deepcopy(fast_config)
        pyddm_config["estimator"]["type"] = "pyddm"

        gen_pyddm = TrainingDataGenerator(pyddm_config, model_config["ddm"])
        assert isinstance(gen_pyddm._generation_pipeline, PyDDMPipeline)

        # Verify PyDDM pipeline works
        data_pyddm = gen_pyddm.generate_data_training(save=False, verbose=False)
        assert "lan_data" in data_pyddm
        assert data_pyddm["lan_data"].shape[0] > 0

        # Both should produce compatible data shapes
        assert data_kde["lan_data"].shape[1] == data_pyddm["lan_data"].shape[1]
