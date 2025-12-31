"""Tests for PyDDM auxiliary label computation.

This module tests that PyDDMGenerationStrategy correctly computes
auxiliary labels (choice_p, omission_p, nogo_p) analytically.
"""

import numpy as np
import pytest

try:
    import pyddm  # noqa: F401

    PYDDM_AVAILABLE = True
except ImportError:
    PYDDM_AVAILABLE = False

from ssms.config import model_config, get_lan_config
from ssms.dataset_generators.estimator_builders.builder_factory import (
    create_estimator_builder,
)
from ssms.dataset_generators.strategies.resample_mixture_strategy import (
    ResampleMixtureStrategy,
)
from ssms.dataset_generators.strategies.pyddm_strategy import (
    PyDDMGenerationStrategy,
)


@pytest.mark.skipif(not PYDDM_AVAILABLE, reason="PyDDM not installed")
class TestPyDDMAuxiliaryLabels:
    """Test auxiliary label computation in PyDDM strategy."""

    def test_auxiliary_labels_exist(self):
        """Test that all auxiliary label fields are present and not None."""
        config = get_lan_config()
        config["model"] = "ddm"
        config["estimator_type"] = "pyddm"

        estimator_builder = create_estimator_builder(config, model_config["ddm"])
        training_strategy = ResampleMixtureStrategy(config, model_config["ddm"])

        strategy = PyDDMGenerationStrategy(
            generator_config=config,
            model_config=model_config["ddm"],
            estimator_builder=estimator_builder,
            training_strategy=training_strategy,
        )

        result = strategy.generate_for_parameter_set(
            parameter_sampling_seed=42, random_seed=42
        )

        assert result["success"], f"Generation failed: {result.get('error')}"

        data = result["data"]

        # Check all fields exist
        assert "cpn_data" in data
        assert "cpn_labels" in data
        assert "cpn_no_omission_data" in data
        assert "cpn_no_omission_labels" in data
        assert "opn_data" in data
        assert "opn_labels" in data
        assert "gonogo_data" in data
        assert "gonogo_labels" in data
        assert "binned_128" in data
        assert "binned_256" in data

        # Check that auxiliary labels are not None (except binned)
        assert data["cpn_data"] is not None
        assert data["cpn_labels"] is not None
        assert data["cpn_no_omission_data"] is not None
        assert data["cpn_no_omission_labels"] is not None
        assert data["opn_data"] is not None
        assert data["opn_labels"] is not None
        assert data["gonogo_data"] is not None
        assert data["gonogo_labels"] is not None

        # Binned histograms should be None for PyDDM
        assert data["binned_128"] is None
        assert data["binned_256"] is None

    def test_auxiliary_labels_shapes(self):
        """Test that auxiliary labels have correct shapes."""
        config = get_lan_config()
        config["model"] = "ddm"
        config["estimator_type"] = "pyddm"

        estimator_builder = create_estimator_builder(config, model_config["ddm"])
        training_strategy = ResampleMixtureStrategy(config, model_config["ddm"])

        strategy = PyDDMGenerationStrategy(
            generator_config=config,
            model_config=model_config["ddm"],
            estimator_builder=estimator_builder,
            training_strategy=training_strategy,
        )

        result = strategy.generate_for_parameter_set(
            parameter_sampling_seed=42, random_seed=42
        )
        data = result["data"]

        # Check shapes (n_trials=1 for single parameter set)
        assert data["cpn_labels"].shape == (1, 2), "cpn_labels should be (1, 2)"
        assert data["cpn_no_omission_labels"].shape == (
            1,
            2,
        ), "cpn_no_omission_labels should be (1, 2)"
        assert data["opn_labels"].shape == (1, 1), "opn_labels should be (1, 1)"
        assert data["gonogo_labels"].shape == (1, 1), "gonogo_labels should be (1, 1)"

    def test_choice_probabilities_sum_to_one(self):
        """Test that choice probabilities sum to approximately 1."""
        config = get_lan_config()
        config["model"] = "ddm"
        config["estimator_type"] = "pyddm"

        estimator_builder = create_estimator_builder(config, model_config["ddm"])
        training_strategy = ResampleMixtureStrategy(config, model_config["ddm"])

        strategy = PyDDMGenerationStrategy(
            generator_config=config,
            model_config=model_config["ddm"],
            estimator_builder=estimator_builder,
            training_strategy=training_strategy,
        )

        result = strategy.generate_for_parameter_set(
            parameter_sampling_seed=42, random_seed=42
        )
        data = result["data"]

        # choice_p should sum to ~1 (may be slightly less due to P(undecided))
        choice_sum = data["cpn_labels"][0, 0] + data["cpn_labels"][0, 1]
        assert 0.9 < choice_sum <= 1.0, f"choice_p sum = {choice_sum}, expected ~1.0"

        # choice_p_no_omission should sum to exactly 1 (renormalized)
        choice_no_omit_sum = (
            data["cpn_no_omission_labels"][0, 0] + data["cpn_no_omission_labels"][0, 1]
        )
        # Use larger tolerance due to float32 precision
        assert np.isclose(choice_no_omit_sum, 1.0, atol=1e-4), (
            f"choice_p_no_omission sum = {choice_no_omit_sum}, expected 1.0"
        )

    def test_probabilities_are_valid(self):
        """Test that all probabilities are in [0, 1]."""
        config = get_lan_config()
        config["model"] = "ddm"
        config["estimator_type"] = "pyddm"

        estimator_builder = create_estimator_builder(config, model_config["ddm"])
        training_strategy = ResampleMixtureStrategy(config, model_config["ddm"])

        strategy = PyDDMGenerationStrategy(
            generator_config=config,
            model_config=model_config["ddm"],
            estimator_builder=estimator_builder,
            training_strategy=training_strategy,
        )

        result = strategy.generate_for_parameter_set(
            parameter_sampling_seed=42, random_seed=42
        )
        data = result["data"]

        # All probabilities should be in [0, 1]
        assert np.all(data["cpn_labels"] >= 0) and np.all(data["cpn_labels"] <= 1)
        assert np.all(data["cpn_no_omission_labels"] >= 0) and np.all(
            data["cpn_no_omission_labels"] <= 1
        )
        assert np.all(data["opn_labels"] >= 0) and np.all(data["opn_labels"] <= 1)
        assert np.all(data["gonogo_labels"] >= 0) and np.all(data["gonogo_labels"] <= 1)

    def test_nogo_equals_error_probability_no_deadline(self):
        """Test that nogo_p equals P(error) when there's no deadline."""
        config = get_lan_config()
        config["model"] = "ddm"
        config["estimator_type"] = "pyddm"

        estimator_builder = create_estimator_builder(config, model_config["ddm"])
        training_strategy = ResampleMixtureStrategy(config, model_config["ddm"])

        strategy = PyDDMGenerationStrategy(
            generator_config=config,
            model_config=model_config["ddm"],
            estimator_builder=estimator_builder,
            training_strategy=training_strategy,
        )

        result = strategy.generate_for_parameter_set(
            parameter_sampling_seed=42, random_seed=42
        )
        data = result["data"]

        # For models without deadline, nogo_p should equal error probability
        error_prob = data["cpn_labels"][0, 0]
        nogo_prob = data["gonogo_labels"][0, 0]

        # nogo_p = P(error) when no deadline
        assert np.isclose(nogo_prob, error_prob, atol=1e-6), (
            f"nogo_p={nogo_prob}, P(error)={error_prob}"
        )

    def test_omission_zero_without_deadline(self):
        """Test that omission_p is zero when there's no deadline."""
        config = get_lan_config()
        config["model"] = "ddm"
        config["estimator_type"] = "pyddm"

        estimator_builder = create_estimator_builder(config, model_config["ddm"])
        training_strategy = ResampleMixtureStrategy(config, model_config["ddm"])

        strategy = PyDDMGenerationStrategy(
            generator_config=config,
            model_config=model_config["ddm"],
            estimator_builder=estimator_builder,
            training_strategy=training_strategy,
        )

        result = strategy.generate_for_parameter_set(
            parameter_sampling_seed=42, random_seed=42
        )
        data = result["data"]

        # Without deadline, omission_p should be 0
        assert np.isclose(data["opn_labels"][0, 0], 0.0, atol=1e-6), (
            f"omission_p should be 0 without deadline, got {data['opn_labels'][0, 0]}"
        )

    def test_output_format_matches_simulation_strategy(self):
        """Test that output format matches SimulationBasedGenerationStrategy."""
        config = get_lan_config()
        config["model"] = "ddm"
        config["estimator_type"] = "pyddm"

        estimator_builder = create_estimator_builder(config, model_config["ddm"])
        training_strategy = ResampleMixtureStrategy(config, model_config["ddm"])

        strategy = PyDDMGenerationStrategy(
            generator_config=config,
            model_config=model_config["ddm"],
            estimator_builder=estimator_builder,
            training_strategy=training_strategy,
        )

        result = strategy.generate_for_parameter_set(
            parameter_sampling_seed=42, random_seed=42
        )
        data = result["data"]

        # Check that all expected keys exist (matching simulation strategy)
        expected_keys = {
            "lan_data",
            "lan_labels",
            "cpn_data",
            "cpn_labels",
            "cpn_no_omission_data",
            "cpn_no_omission_labels",
            "opn_data",
            "opn_labels",
            "gonogo_data",
            "gonogo_labels",
            "binned_128",
            "binned_256",
            "theta",
        }

        assert set(data.keys()) == expected_keys, (
            f"Missing keys: {expected_keys - set(data.keys())}, "
            f"Extra keys: {set(data.keys()) - expected_keys}"
        )
