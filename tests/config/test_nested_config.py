"""Tests for nested config structure support."""

import pytest

from ssms.config.config_utils import (
    get_nested_config,
    has_nested_structure,
    convert_flat_to_nested,
)
from ssms.config.generator_config.data_generator_config import (
    get_default_generator_config,
    get_nested_generator_config,
)


class TestNestedConfigAccess:
    """Test the nested config accessor functions."""

    def test_get_nested_config_from_nested_structure(self):
        """Test accessing values from nested structure."""
        config = {
            "pipeline": {"n_parameter_sets": 100, "n_subruns": 10},
            "estimator": {"type": "kde", "bandwidth": 0.1},
        }

        assert get_nested_config(config, "pipeline", "n_parameter_sets") == 100
        assert get_nested_config(config, "estimator", "type") == "kde"

    def test_get_nested_config_returns_none_for_missing_section(self):
        """Test that missing sections return default value."""
        config = {"estimator": {"type": "kde"}}  # No pipeline section

        # Should return default for missing section
        assert get_nested_config(config, "pipeline", "n_parameter_sets") is None
        assert (
            get_nested_config(config, "pipeline", "n_parameter_sets", default=100)
            == 100
        )

    def test_get_nested_config_default_value(self):
        """Test default value when key not found."""
        config = {
            "pipeline": {"n_subruns": 10}
        }  # Has pipeline but not n_parameter_sets

        assert (
            get_nested_config(config, "pipeline", "n_parameter_sets", default=42) == 42
        )
        assert get_nested_config(config, "estimator", "type", default="kde") == "kde"

    def test_get_nested_config_only_works_with_nested(self):
        """Test that get_nested_config only works with nested structure."""
        flat_config = {
            "n_parameter_sets": 100,
            "estimator_type": "kde",
        }

        # Should NOT find values in flat config (no fallback anymore)
        assert get_nested_config(flat_config, "pipeline", "n_parameter_sets") is None
        assert get_nested_config(flat_config, "estimator", "type") is None


class TestNestedConfigDetection:
    """Test nested structure detection."""

    def test_has_nested_structure_true(self):
        """Test detection of nested structure."""
        config = {"pipeline": {}, "estimator": {}}
        assert has_nested_structure(config) is True

        config = {"pipeline": {"n_parameter_sets": 100}}
        assert has_nested_structure(config) is True

    def test_has_nested_structure_false(self):
        """Test detection of flat structure."""
        config = {"n_parameter_sets": 100, "estimator_type": "kde"}
        assert has_nested_structure(config) is False

        config = {}
        assert has_nested_structure(config) is False


class TestFlatConfigRejection:
    """Test that flat configs are properly rejected."""

    def test_has_nested_structure_detects_flat(self):
        """Test that flat structure is correctly identified."""
        flat_config = {"n_parameter_sets": 100, "estimator_type": "kde"}
        assert has_nested_structure(flat_config) is False

    def test_has_nested_structure_detects_nested(self):
        """Test that nested structure is correctly identified."""
        nested_config = {"pipeline": {"n_parameter_sets": 100}}
        assert has_nested_structure(nested_config) is True

    def test_flat_config_rejected_in_practice(self):
        """Test that flat configs are rejected by DataGenerator."""
        from ssms.config import model_config
        from ssms.dataset_generators.lan_mlp import DataGenerator

        flat_config = {"n_parameter_sets": 10, "estimator_type": "kde"}

        with pytest.raises(
            ValueError, match="Flat generator_config structure is no longer supported"
        ):
            DataGenerator(flat_config, model_config["ddm"])


class TestFlatToNestedConversion:
    """Test conversion from flat to nested structure."""

    def test_convert_pipeline_settings(self):
        """Test conversion of pipeline settings."""
        flat = {
            "n_parameter_sets": 100,
            "n_subruns": 10,
            "n_cpus": 4,
        }

        nested = convert_flat_to_nested(flat)

        assert nested["pipeline"]["n_parameter_sets"] == 100
        assert nested["pipeline"]["n_subruns"] == 10
        assert nested["pipeline"]["n_cpus"] == 4

    def test_convert_estimator_settings(self):
        """Test conversion of estimator settings."""
        flat = {
            "estimator_type": "kde",
            "kde_bandwidth": 0.1,
            "kde_displace_t": True,
        }

        nested = convert_flat_to_nested(flat)

        assert nested["estimator"]["type"] == "kde"
        assert nested["estimator"]["bandwidth"] == 0.1
        assert nested["estimator"]["displace_t"] is True

    def test_convert_training_settings(self):
        """Test conversion of training settings."""
        flat = {
            "data_mixture_probabilities": [0.8, 0.1, 0.1],
            "n_training_samples_by_parameter_set": 1000,
            "separate_response_channels": True,
        }

        nested = convert_flat_to_nested(flat)

        assert nested["training"]["mixture_probabilities"] == [0.8, 0.1, 0.1]
        assert nested["training"]["n_samples_per_param"] == 1000
        assert nested["training"]["separate_response_channels"] is True

    def test_convert_simulator_settings(self):
        """Test conversion of simulator settings."""
        flat = {
            "delta_t": 0.001,
            "max_t": 20.0,
            "n_samples": 1000,
            "smooth_unif": 0.0,
        }

        nested = convert_flat_to_nested(flat)

        assert nested["simulator"]["delta_t"] == 0.001
        assert nested["simulator"]["max_t"] == 20.0
        assert nested["simulator"]["n_samples"] == 1000
        assert nested["simulator"]["smooth_unif"] == 0.0

    def test_convert_output_settings(self):
        """Test conversion of output settings."""
        flat = {
            "output_folder": "data/training",
            "pickleprotocol": 4,
        }

        nested = convert_flat_to_nested(flat)

        assert nested["output"]["folder"] == "data/training"
        assert nested["output"]["pickle_protocol"] == 4

    def test_convert_preserves_other_keys(self):
        """Test that conversion preserves keys at top level."""
        flat = {
            "model": "ddm",
            "bin_pointwise": False,
        }

        nested = convert_flat_to_nested(flat)

        assert nested["model"] == "ddm"
        assert nested["bin_pointwise"] is False


class TestGeneratorConfigFunctions:
    """Test generator config creation functions."""

    def test_get_default_generator_config_always_nested(self):
        """Test that default config always returns nested structure."""
        config = get_default_generator_config("lan")

        # Should always have nested structure
        assert has_nested_structure(config) is True
        assert "pipeline" in config
        assert "estimator" in config
        assert "training" in config
        assert "simulator" in config
        assert "output" in config

    def test_get_default_generator_config_rejects_flat_in_datagenerator(self):
        """Test that DataGenerator rejects flat configs."""
        from ssms.config import model_config
        from ssms.dataset_generators.lan_mlp import DataGenerator

        # Create a flat config
        flat_config = {
            "n_parameter_sets": 100,
            "estimator_type": "kde",
            "delta_t": 0.001,
        }

        # Should raise ValueError with helpful message
        with pytest.raises(
            ValueError, match="Flat generator_config structure is no longer supported"
        ):
            DataGenerator(flat_config, model_config["ddm"])

    def test_get_nested_generator_config(self):
        """Test convenience function for nested config."""
        config = get_nested_generator_config("lan")

        # Should have nested structure
        assert has_nested_structure(config) is True
        assert "pipeline" in config

    def test_nested_config_values_correct(self):
        """Test that nested config has correct values."""
        nested = get_default_generator_config("lan")

        # Check key values are present
        assert nested["pipeline"]["n_parameter_sets"] == 10000
        assert nested["simulator"]["delta_t"] == 0.001
        assert nested["output"]["folder"] == "data/lan_mlp/"


class TestNestedOnlyPolicy:
    """Test that flat configs are no longer supported."""

    def test_config_always_nested(self):
        """Test that get_default_generator_config always returns nested."""
        config = get_default_generator_config("lan")

        # Should be nested
        assert has_nested_structure(config) is True
        assert "pipeline" in config
        assert config["pipeline"]["n_parameter_sets"] == 10000

    def test_flat_config_rejected_by_datagenerator(self):
        """Test that DataGenerator rejects flat configs."""
        from ssms.config import model_config
        from ssms.dataset_generators.lan_mlp import DataGenerator

        flat_config = {
            "n_parameter_sets": 100,
            "n_samples": 1000,
            "delta_t": 0.001,
            "max_t": 20.0,
            "estimator_type": "kde",
        }

        with pytest.raises(
            ValueError, match="Flat generator_config structure is no longer supported"
        ):
            DataGenerator(flat_config, model_config["ddm"])

    def test_conversion_utility_still_available(self):
        """Test that convert_flat_to_nested is available for migration."""
        flat_config = {
            "n_parameter_sets": 100,
            "estimator_type": "kde",
            "delta_t": 0.001,
        }

        nested = convert_flat_to_nested(flat_config)

        assert has_nested_structure(nested) is True
        assert nested["pipeline"]["n_parameter_sets"] == 100
        assert nested["estimator"]["type"] == "kde"
        assert nested["simulator"]["delta_t"] == 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
