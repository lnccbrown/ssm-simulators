"""Tests for config utilities.

This module tests the nested config utilities used throughout the codebase.
"""

from ssms.config.config_utils import (
    has_nested_structure,
    get_nested_config,
)


class TestHasNestedStructure:
    """Test detection of nested config structure."""

    def test_has_nested_structure_true(self):
        """Test detection of nested structure."""
        nested_config = {
            "pipeline": {"n_parameter_sets": 100},
            "estimator": {"type": "kde"},
        }

        assert has_nested_structure(nested_config) is True

    def test_has_nested_structure_false(self):
        """Test detection of flat structure (no longer supported)."""
        flat_config = {
            "n_parameter_sets": 100,
            "estimator_type": "kde",
        }

        assert has_nested_structure(flat_config) is False

    def test_has_nested_structure_partial(self):
        """Test detection when at least one nested section exists."""
        partial_config = {
            "pipeline": {"n_parameter_sets": 100},
            "estimator_type": "kde",  # Flat key (mixed structure)
        }

        assert has_nested_structure(partial_config) is True

    def test_has_nested_structure_empty(self):
        """Test with empty config."""
        empty_config = {}

        assert has_nested_structure(empty_config) is False


class TestGetNestedConfig:
    """Test nested config accessor utility."""

    def test_get_nested_config_exists(self):
        """Test getting existing nested value."""
        config = {
            "pipeline": {"n_parameter_sets": 100, "n_cpus": 4},
            "estimator": {"type": "kde"},
        }

        value = get_nested_config(config, "pipeline", "n_parameter_sets")
        assert value == 100

        value = get_nested_config(config, "estimator", "type")
        assert value == "kde"

    def test_get_nested_config_missing_key(self):
        """Test getting missing key returns default."""
        config = {
            "pipeline": {"n_parameter_sets": 100},
        }

        value = get_nested_config(config, "pipeline", "missing_key", default=42)
        assert value == 42

    def test_get_nested_config_missing_section(self):
        """Test getting from missing section returns default."""
        config = {
            "pipeline": {"n_parameter_sets": 100},
        }

        value = get_nested_config(config, "missing_section", "key", default="default")
        assert value == "default"

    def test_get_nested_config_no_default(self):
        """Test getting missing value with no default returns None."""
        config = {
            "pipeline": {"n_parameter_sets": 100},
        }

        value = get_nested_config(config, "pipeline", "missing_key")
        assert value is None

    def test_get_nested_config_section_not_dict(self):
        """Test when section exists but is not a dict."""
        config = {
            "pipeline": "not_a_dict",
        }

        value = get_nested_config(config, "pipeline", "key", default="default")
        assert value == "default"
