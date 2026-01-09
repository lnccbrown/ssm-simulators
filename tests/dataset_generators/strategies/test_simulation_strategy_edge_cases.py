"""Tests for SimulationPipeline error handling and edge cases."""

import numpy as np
import pytest
from ssms import OMISSION_SENTINEL
from ssms.config import model_config, get_lan_config
from ssms.dataset_generators.pipelines import SimulationPipeline
from ssms.dataset_generators.estimator_builders.kde_builder import KDEEstimatorBuilder
from ssms.dataset_generators.strategies import ResampleMixtureStrategy


class TestSimulationPipelineErrorHandling:
    """Test error handling in SimulationPipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline instance for testing."""
        config = get_lan_config()

        # SimulationPipeline accepts classes and instantiates them internally
        return SimulationPipeline(
            generator_config=config,
            model_config=model_config["ddm"],
            estimator_builder=KDEEstimatorBuilder,
            training_strategy=ResampleMixtureStrategy,
        )

    def test_is_valid_simulation_with_none(self, pipeline):
        """Test _is_valid_simulation raises ValueError when simulations is None."""
        with pytest.raises(ValueError, match="No simulations provided"):
            pipeline._is_valid_simulation(None)

    def test_compute_auxiliary_labels_all_omissions(self, pipeline):
        """Test _compute_auxiliary_labels when all responses are omissions."""
        # Create simulations with all omissions (RT and choice = OMISSION_SENTINEL)
        simulations = {
            "rts": np.array([OMISSION_SENTINEL] * 5),
            "choices": np.array([OMISSION_SENTINEL] * 5),
            "metadata": {
                "possible_choices": [-1, 1],
                "max_t": 20.0,
            },
        }

        labels = pipeline._compute_auxiliary_labels(simulations)

        # All omissions should result in uniform choice probabilities (excluding omissions)
        assert labels["choice_p_no_omission"][0, 0] == 0.5
        assert labels["choice_p_no_omission"][0, 1] == 0.5

        # Omission probability should be 1.0
        assert labels["omission_p"][0, 0] == 1.0

        # Nogo probability should be 1.0 (all omitted)
        assert labels["nogo_p"][0, 0] == 1.0

    def test_compute_auxiliary_labels_single_choice(self, pipeline):
        """Test _compute_auxiliary_labels with only one choice type."""
        # All responses are choice 1 (correct)
        simulations = {
            "rts": np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
            "choices": np.array([1, 1, 1, 1, 1]),
            "metadata": {
                "possible_choices": [-1, 1],
                "max_t": 20.0,
            },
        }

        labels = pipeline._compute_auxiliary_labels(simulations)

        # Choice probabilities: all choice 1, none choice -1
        assert labels["choice_p"][0, 0] == 0.0  # P(choice=-1)
        assert labels["choice_p"][0, 1] == 1.0  # P(choice=1)

        # No omissions
        assert labels["omission_p"][0, 0] == 0.0

        # Nogo = 0 (all chose max choice = 1)
        assert labels["nogo_p"][0, 0] == 0.0

    def test_compute_auxiliary_labels_mixed_responses(self, pipeline):
        """Test _compute_auxiliary_labels with mixed choices and omissions."""
        # 3 correct (choice=1), 2 error (choice=-1), 1 omission
        simulations = {
            "rts": np.array([0.5, 0.6, 0.7, 0.8, 0.9, OMISSION_SENTINEL]),
            "choices": np.array([1, 1, 1, -1, -1, OMISSION_SENTINEL]),
            "metadata": {
                "possible_choices": [-1, 1],
                "max_t": 20.0,
            },
        }

        labels = pipeline._compute_auxiliary_labels(simulations)

        # Choice probabilities (including omissions)
        # 3/6 = 0.5 for choice=1, 2/6 ≈ 0.333 for choice=-1
        assert np.isclose(labels["choice_p"][0, 1], 3 / 6)
        assert np.isclose(labels["choice_p"][0, 0], 2 / 6)

        # Choice probabilities excluding omissions
        # 3/5 = 0.6 for choice=1, 2/5 = 0.4 for choice=-1
        assert np.isclose(labels["choice_p_no_omission"][0, 1], 3 / 5)
        assert np.isclose(labels["choice_p_no_omission"][0, 0], 2 / 5)

        # Omission probability: 1/6
        assert np.isclose(labels["omission_p"][0, 0], 1 / 6)

    def test_compute_auxiliary_labels_binned_histograms(self, pipeline):
        """Test that binned histograms are computed correctly."""
        # Mix of RTs for different choices
        simulations = {
            "rts": np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1]),
            "choices": np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1]),
            "metadata": {
                "possible_choices": [-1, 1],
                "max_t": 20.0,
            },
        }

        labels = pipeline._compute_auxiliary_labels(simulations)

        # Should have histograms
        assert "binned_128" in labels
        assert "binned_256" in labels

        # Check shapes: (n_trials=1, n_bins, n_choices)
        assert labels["binned_128"].shape == (1, 128, 2)
        assert labels["binned_256"].shape == (1, 256, 2)

        # Histograms should sum to number of samples per choice
        assert np.sum(labels["binned_128"][0, :, 0]) == 5  # 5 samples for choice=-1
        assert np.sum(labels["binned_128"][0, :, 1]) == 5  # 5 samples for choice=1

    def test_compute_auxiliary_labels_excludes_omissions_from_histograms(
        self, pipeline
    ):
        """Test that omissions (RT=OMISSION_SENTINEL) are excluded from RT histograms."""
        simulations = {
            "rts": np.array([0.5, 0.6, OMISSION_SENTINEL, OMISSION_SENTINEL]),
            "choices": np.array([1, -1, 1, -1]),
            "metadata": {
                "possible_choices": [-1, 1],
                "max_t": 20.0,
            },
        }

        labels = pipeline._compute_auxiliary_labels(simulations)

        # Histograms should only count the 2 non-omission RTs
        assert (
            np.sum(labels["binned_128"][0, :, 0]) == 1
        )  # 1 sample for choice=-1 (RT=0.6)
        assert (
            np.sum(labels["binned_128"][0, :, 1]) == 1
        )  # 1 sample for choice=1 (RT=0.5)

    def test_generate_for_parameter_set_failure_after_max_attempts(self, pipeline):
        """Test that generate_for_parameter_set returns failure after max attempts."""
        # Create a pipeline with impossible filters
        config = get_lan_config()
        # SimulationPipeline reads filters from simulator.filters
        config["simulator"]["filters"] = {
            "mode": 0.001,  # Impossible: mode must be tiny
            "choice_cnt": 1000,  # Impossible: need 1000 samples of each choice
            "mean_rt": 0.001,  # Impossible: mean must be tiny
            "std": 1000,  # Impossible: std must be huge
            "mode_cnt_rel": 0.0,  # Impossible: mode can't exist
        }

        bad_pipeline = SimulationPipeline(
            generator_config=config,
            model_config=model_config["ddm"],
            estimator_builder=KDEEstimatorBuilder,
            training_strategy=ResampleMixtureStrategy,
        )

        # Should fail after max attempts (this will take a moment)
        result = bad_pipeline.generate_for_parameter_set(
            parameter_sampling_seed=42, simulator_seed=42
        )

        assert result["success"] is False
        assert result["data"] is None

    def test_is_valid_simulation_respects_all_filters(self, pipeline):
        """Test that _is_valid_simulation applies all filters correctly."""
        # Create simulations that should pass
        valid_simulations = {
            "rts": np.linspace(0.3, 2.0, 100),  # Spread out RTs
            "choices": np.array([1] * 50 + [-1] * 50),  # Equal choices
            "metadata": {
                "possible_choices": [-1, 1],
                "max_t": 20.0,
            },
        }

        keep, stats = pipeline._is_valid_simulation(valid_simulations)

        # Should pass with reasonable parameters
        # Note: Using truthiness test (not identity) to work with both bool and np.bool_
        assert keep

        # Stats should have expected shape
        assert len(stats) == 6  # [mode, mean, std, mode_cnt_rel, tmp_n_c, n_sim]

    def test_generate_for_parameter_set_with_isolated_rng(self, pipeline):
        """Test that parameter sampling uses isolated RNG (no global state pollution)."""
        # Generate twice with same seed - should get identical parameters
        result1 = pipeline.generate_for_parameter_set(
            parameter_sampling_seed=123, simulator_seed=456
        )

        result2 = pipeline.generate_for_parameter_set(
            parameter_sampling_seed=123,
            simulator_seed=789,  # Different simulator seed, same param seed
        )

        # Parameters should match (same param seed)
        theta1 = result1["theta"]
        theta2 = result2["theta"]

        for key in theta1:
            assert np.allclose(theta1[key], theta2[key]), f"Parameter {key} differs"

    def test_generate_for_parameter_set_success_format(self, pipeline):
        """Test that successful generation returns expected format."""
        result = pipeline.generate_for_parameter_set(
            parameter_sampling_seed=42, simulator_seed=42
        )

        assert result["success"] is True
        assert "data" in result
        assert "theta" in result

        data = result["data"]

        # Check all expected keys
        expected_keys = [
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
        ]

        for key in expected_keys:
            assert key in data, f"Missing key: {key}"
