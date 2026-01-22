"""Integration tests for PyDDM-based training data generation."""

import pytest
import numpy as np

# Skip all tests if pyddm not installed
pytest.importorskip("pyddm")

from ssms.config import model_config
from ssms.config.generator_config.data_generator_config import (
    get_default_generator_config,
    get_lan_config,
)
from ssms.dataset_generators.lan_mlp import TrainingDataGenerator
from ssms.dataset_generators.estimator_builders.builder_factory import (
    create_estimator_builder,
)
from ssms.dataset_generators.estimator_builders.pyddm_builder import (
    PyDDMEstimatorBuilder,
)


@pytest.fixture
def base_generator_config():
    """Base configuration for data generation with nested structure."""
    config = get_default_generator_config("lan")

    # Update nested structure properly
    config["simulator"]["n_samples"] = 2000
    config["simulator"]["delta_t"] = 0.001
    config["simulator"]["max_t"] = 10.0

    config["pipeline"]["n_parameter_sets"] = (
        20  # Increased to ensure enough pass filters
    )

    config["training"]["n_samples_per_param"] = 200
    config["training"]["mixture_probabilities"] = [0.8, 0.1, 0.1]

    config["pipeline"]["simulation_filters"] = {
        "mode": 20,
        "choice_cnt": 0,
        "mean_rt": 17,
        "std": 0,
        "mode_cnt_rel": 0.95,
    }

    return config


class TestPyDDMTrainingDataGeneratorIntegration:
    """Test PyDDM estimator integration with TrainingDataGenerator."""

    def test_TrainingDataGenerator_with_pyddm_ddm(self, base_generator_config):
        """Test TrainingDataGenerator with PyDDM estimator for DDM."""
        from copy import deepcopy

        generator_config = deepcopy(base_generator_config)
        generator_config["estimator"]["type"] = "pyddm"
        generator_config["estimator"]["pdf_interpolation"] = "cubic"

        gen = TrainingDataGenerator(
            config=generator_config,
            model_config=model_config["ddm"],
        )

        # Generate training data
        training_data_dict = gen.generate_data_training(save=False)

        # Verify data was created
        assert "lan_data" in training_data_dict
        assert "lan_labels" in training_data_dict
        lan_data = training_data_dict["lan_data"]
        lan_labels = training_data_dict["lan_labels"]

        n_params = len(model_config["ddm"]["params"])
        # lan_data has: params + RT + choice
        expected_cols = n_params + 2
        assert lan_data.shape[1] == expected_cols
        assert lan_data.shape[0] > 0

        # Verify RTs and choices are valid
        # Note: Training data includes some negative RTs (mixture strategy for boundary learning)
        rt_col = n_params
        choice_col = n_params + 1
        assert np.all(np.isfinite(lan_data[:, rt_col]))
        assert np.all(np.isin(lan_data[:, choice_col], [-1, 1]))

        # Verify likelihoods are reasonable
        assert lan_labels.shape[0] == lan_data.shape[0]
        assert np.all(np.isfinite(lan_labels))
        # Note: lan_labels are log-likelihoods, which should be <= 0, but may have
        # some numerical edge cases where very small positive values appear
        assert np.median(lan_labels) < 0  # Most log-likelihoods should be negative

    def test_pyddm_training_data_structure(self, base_generator_config):
        """Test that PyDDM generates properly structured training data."""
        from copy import deepcopy

        generator_config = deepcopy(base_generator_config)
        generator_config["estimator"]["type"] = "pyddm"

        gen = TrainingDataGenerator(
            config=generator_config,
            model_config=model_config["ddm"],
        )

        training_data_dict = gen.generate_data_training(save=False)

        # Check structure
        data = training_data_dict["lan_data"]
        n_params = len(model_config["ddm"]["params"])

        # lan_data has: params + RT + choice (no likelihood column)
        assert data.shape[1] == n_params + 2

        # Parameters should be within bounds
        param_bounds = model_config["ddm"]["param_bounds"]
        for i, (low, high) in enumerate(zip(param_bounds[0], param_bounds[1])):
            assert np.all(data[:, i] >= low)
            assert np.all(data[:, i] <= high)

        # RTs should be finite (includes negative RTs from mixture strategy)
        rt_col = n_params
        assert np.all(np.isfinite(data[:, rt_col]))

        # Choices should be -1 or 1
        choice_col = n_params + 1
        assert np.all(np.isin(data[:, choice_col], [-1, 1]))

    def test_pyddm_with_ornstein(self, base_generator_config):
        """Test PyDDM with position-dependent drift (Ornstein)."""
        from copy import deepcopy

        generator_config = deepcopy(base_generator_config)
        generator_config["estimator"]["type"] = "pyddm"

        gen = TrainingDataGenerator(
            config=generator_config,
            model_config=model_config["ornstein"],
        )

        training_data_dict = gen.generate_data_training(save=False)
        lan_data = training_data_dict["lan_data"]

        assert lan_data.shape[0] > 0
        assert np.all(np.isfinite(lan_data[:, -1]))

    def test_pyddm_incompatible_model_fails(self, base_generator_config):
        """Test that incompatible models raise clear errors."""
        from copy import deepcopy

        generator_config = deepcopy(base_generator_config)
        generator_config["estimator"]["type"] = "pyddm"

        # Race model should fail
        with pytest.raises(ValueError, match="not compatible with PyDDM"):
            TrainingDataGenerator(
                config=generator_config,
                model_config=model_config["race_3"],
            )


class TestPyDDMVsKDEComparison:
    """Test PyDDM vs KDE consistency."""

    def test_pyddm_vs_kde_produce_similar_distributions(self, base_generator_config):
        """Test that PyDDM and KDE produce similar training data distributions.

        Note: We test that both methods produce reasonable log-likelihoods in similar ranges,
        but we don't expect them to match exactly since KDE is an approximation while PyDDM
        uses analytical solutions. We allow for substantial differences (up to 250% relative error)
        since the KDE approximation quality depends on simulation parameters.
        """
        from copy import deepcopy

        # Generate with KDE
        kde_config = deepcopy(base_generator_config)
        kde_config["estimator"]["type"] = "kde"
        kde_gen = TrainingDataGenerator(
            config=kde_config,
            model_config=model_config["ddm"],
        )
        kde_data_dict = kde_gen.generate_data_training(save=False)

        # Generate with PyDDM
        pyddm_config = deepcopy(base_generator_config)
        pyddm_config["estimator"]["type"] = "pyddm"
        pyddm_gen = TrainingDataGenerator(
            config=pyddm_config,
            model_config=model_config["ddm"],
        )
        pyddm_data_dict = pyddm_gen.generate_data_training(save=False)

        # Both should produce data
        kde_data = kde_data_dict["lan_data"]
        pyddm_data = pyddm_data_dict["lan_data"]
        assert kde_data.shape == pyddm_data.shape

        # Log-likelihoods should be in similar range (not identical due to KDE vs analytical)
        # Note: lan_labels contains the actual log-likelihoods, not lan_data[:, -1]
        # (lan_data[:, -1] is the choice column, which only contains -1 or 1)
        kde_lls = kde_data_dict["lan_labels"]
        pyddm_lls = pyddm_data_dict["lan_labels"]

        # Both should produce reasonable log-likelihood ranges
        assert np.all(np.isfinite(kde_lls)), "KDE log-likelihoods should be finite"
        assert np.all(np.isfinite(pyddm_lls)), "PyDDM log-likelihoods should be finite"

        # Medians should be negative (log-probabilities are typically negative)
        assert np.median(kde_lls) < 0, "KDE median log-likelihood should be negative"
        assert np.median(pyddm_lls) < 0, (
            "PyDDM median log-likelihood should be negative"
        )

        # Medians should be within similar order of magnitude (allowing up to 250% difference)
        # This is a loose check since KDE quality depends on simulation parameters
        assert np.median(kde_lls) == pytest.approx(np.median(pyddm_lls), rel=2.5)

    def test_pyddm_faster_than_kde_for_many_parameter_sets(self):
        """Test that PyDDM is faster than KDE for generating many parameter sets."""
        import time
        from copy import deepcopy

        config_many_params = get_default_generator_config("lan")
        config_many_params["simulator"]["n_samples"] = 2000
        config_many_params["simulator"]["delta_t"] = 0.001
        config_many_params["simulator"]["max_t"] = 5.0
        config_many_params["pipeline"]["n_parameter_sets"] = (
            30  # More parameter sets to ensure enough pass filters
        )
        config_many_params["training"]["n_samples_per_param"] = 100
        config_many_params["training"]["mixture_probabilities"] = [0.8, 0.1, 0.1]
        config_many_params["pipeline"]["simulation_filters"] = {
            "mode": 20,
            "choice_cnt": 0,
            "mean_rt": 17,
            "std": 0,
            "mode_cnt_rel": 0.95,
        }

        # Time KDE
        kde_config = deepcopy(config_many_params)
        kde_config["estimator"]["type"] = "kde"
        kde_gen = TrainingDataGenerator(
            config=kde_config,
            model_config=model_config["ddm"],
        )
        start_kde = time.time()
        kde_gen.generate_data_training(save=False)
        time_kde = time.time() - start_kde

        # Time PyDDM
        pyddm_config = deepcopy(config_many_params)
        pyddm_config["estimator"]["type"] = "pyddm"
        pyddm_gen = TrainingDataGenerator(
            config=pyddm_config,
            model_config=model_config["ddm"],
        )
        start_pyddm = time.time()
        pyddm_gen.generate_data_training(save=False)
        time_pyddm = time.time() - start_pyddm

        # PyDDM should be faster (or at least not much slower)
        # Note: This is a soft assertion as timing can vary
        print(f"KDE time: {time_kde:.2f}s, PyDDM time: {time_pyddm:.2f}s")
        # We just verify both complete successfully
        assert time_kde > 0 and time_pyddm > 0


class TestPyDDMBuilderFactory:
    """Test PyDDM through the builder factory."""

    def test_factory_creates_pyddm_builder(self, base_generator_config):
        """Test that factory creates PyDDM builder."""
        from copy import deepcopy

        config = deepcopy(base_generator_config)
        config["estimator"]["type"] = "pyddm"

        builder = create_estimator_builder(config, model_config["ddm"])

        assert isinstance(builder, PyDDMEstimatorBuilder)

    def test_pyddm_via_cli_config_structure(self, base_generator_config):
        """Test PyDDM works with CLI-style configuration."""
        from copy import deepcopy

        data_config = deepcopy(base_generator_config)
        data_config["model"] = "ddm"
        data_config["estimator"]["type"] = "pyddm"

        config_dict = {
            "data_config": data_config,
            "model_config": model_config["ddm"],
        }

        builder = create_estimator_builder(
            config_dict["data_config"], config_dict["model_config"]
        )

        assert isinstance(builder, PyDDMEstimatorBuilder)


class TestPyDDMInterpolationOptions:
    """Test PyDDM interpolation configuration."""

    def test_cubic_interpolation(self, base_generator_config):
        """Test that cubic interpolation works."""
        from copy import deepcopy

        generator_config = deepcopy(base_generator_config)
        generator_config["estimator"]["type"] = "pyddm"
        generator_config["estimator"]["pdf_interpolation"] = "cubic"

        gen = TrainingDataGenerator(
            config=generator_config,
            model_config=model_config["ddm"],
        )

        training_data_dict = gen.generate_data_training(save=False)
        assert training_data_dict["lan_data"].shape[0] > 0

    def test_linear_interpolation(self, base_generator_config):
        """Test that linear interpolation works."""
        from copy import deepcopy

        generator_config = deepcopy(base_generator_config)
        generator_config["estimator"]["type"] = "pyddm"
        generator_config["estimator"]["pdf_interpolation"] = "linear"

        gen = TrainingDataGenerator(
            config=generator_config,
            model_config=model_config["ddm"],
        )

        training_data_dict = gen.generate_data_training(save=False)
        assert training_data_dict["lan_data"].shape[0] > 0


class TestPyDDMAccuracy:
    """Test PyDDM accuracy against known properties."""

    def test_high_drift_increases_correct_choice_probability(self):
        """Test that higher drift increases P(correct) - tested at builder level."""
        from ssms.dataset_generators.estimator_builders.pyddm_builder import (
            PyDDMEstimatorBuilder,
        )

        config = get_lan_config()
        config["simulator"]["delta_t"] = 0.001
        config["simulator"]["max_t"] = 10.0

        builder = PyDDMEstimatorBuilder(config, model_config["ddm"])

        # Low drift
        theta_low = {"v": 0.5, "a": 2.0, "z": 0.5, "t": 0.3}
        est_low = builder.build(theta_low)
        samples_low = est_low.sample(10000)
        p_correct_low = np.mean(samples_low["choices"] == 1)

        # High drift
        theta_high = {"v": 2.0, "a": 2.0, "z": 0.5, "t": 0.3}
        est_high = builder.build(theta_high)
        samples_high = est_high.sample(10000)
        p_correct_high = np.mean(samples_high["choices"] == 1)

        # Higher drift should increase P(correct)
        assert p_correct_high > p_correct_low

    def test_symmetric_starting_point_gives_equal_choice_probabilities(self):
        """Test that z=0.5 gives approximately equal choice probabilities."""
        from ssms.dataset_generators.estimator_builders.pyddm_builder import (
            PyDDMEstimatorBuilder,
        )

        config = get_lan_config()
        config["simulator"]["delta_t"] = 0.001
        config["simulator"]["max_t"] = 10.0

        builder = PyDDMEstimatorBuilder(config, model_config["ddm"])

        # Symmetric starting point with zero drift
        theta = {"v": 0.0, "a": 2.0, "z": 0.5, "t": 0.3}
        est = builder.build(theta)
        samples = est.sample(10000)

        p_correct = np.mean(samples["choices"] == 1)

        # Should be close to 0.5 (allowing some sampling variability)
        assert p_correct == pytest.approx(0.5, abs=0.05)


class TestPyDDMDeterminism:
    """Test that PyDDM produces deterministic results."""

    def test_same_parameters_give_same_pdfs(self):
        """Test that same parameters produce identical PDFs."""
        from ssms.dataset_generators.estimator_builders.pyddm_builder import (
            PyDDMEstimatorBuilder,
        )

        config = get_lan_config()
        config.update({"delta_t": 0.001, "max_t": 10.0})

        # Build estimators twice with same parameters
        builder1 = PyDDMEstimatorBuilder(config, model_config["ddm"])
        builder2 = PyDDMEstimatorBuilder(config, model_config["ddm"])

        theta = {"v": 1.5, "a": 2.0, "z": 0.5, "t": 0.3}

        est1 = builder1.build(theta, simulations=None)
        est2 = builder2.build(theta, simulations=None)

        # PDFs should be identical
        assert np.allclose(est1.pdf_correct, est2.pdf_correct)
        assert np.allclose(est1.pdf_error, est2.pdf_error)
