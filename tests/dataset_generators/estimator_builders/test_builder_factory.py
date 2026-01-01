"""Tests for estimator builder factory."""

import pytest
from ssms.config import model_config, get_lan_config
from ssms.dataset_generators.estimator_builders.builder_factory import (
    create_estimator_builder,
)
from ssms.dataset_generators.estimator_builders.kde_builder import KDEEstimatorBuilder
from ssms.dataset_generators.estimator_builders.pyddm_builder import (
    PyDDMEstimatorBuilder,
)


class TestEstimatorBuilderFactory:
    """Test the estimator builder factory function."""

    def test_create_estimator_builder_kde(self):
        """Test KDE builder creation."""
        config = get_lan_config()
        config["estimator"]["type"] = "kde"

        builder = create_estimator_builder(config, model_config["ddm"])

        assert isinstance(builder, KDEEstimatorBuilder)

    def test_create_estimator_builder_pyddm(self):
        """Test PyDDM builder creation."""
        pytest.importorskip("pyddm")  # Skip if pyddm not installed

        config = get_lan_config()
        config["estimator"]["type"] = "pyddm"

        builder = create_estimator_builder(config, model_config["ddm"])

        assert isinstance(builder, PyDDMEstimatorBuilder)

    def test_create_estimator_builder_unknown_type(self):
        """Test ValueError for invalid estimator type."""
        config = get_lan_config()
        config["estimator"]["type"] = "unknown_estimator"

        with pytest.raises(ValueError, match="Unknown estimator_type"):
            create_estimator_builder(config, model_config["ddm"])

    def test_create_estimator_builder_with_nested_config(self):
        """Verify nested config handling."""
        config = get_lan_config()
        config["estimator"]["type"] = "kde"

        # Should work with nested config
        builder = create_estimator_builder(config, model_config["ddm"])

        assert isinstance(builder, KDEEstimatorBuilder)
        # Verify builder has access to config
        assert hasattr(builder, "generator_config")

    def test_create_estimator_builder_default_type(self):
        """Test that default estimator type is KDE when not specified."""
        config = get_lan_config()
        # Remove estimator type to test default
        if "type" in config.get("estimator", {}):
            del config["estimator"]["type"]

        builder = create_estimator_builder(config, model_config["ddm"])

        assert isinstance(builder, KDEEstimatorBuilder)
