"""Tests for data generation strategy factory."""

import pytest
from ssms.config import model_config, get_lan_config
from ssms.dataset_generators.strategies.strategy_factory import (
    create_data_generation_strategy,
)
from ssms.dataset_generators.strategies.simulation_based_strategy import (
    SimulationBasedGenerationStrategy,
)
from ssms.dataset_generators.strategies.pyddm_strategy import PyDDMGenerationStrategy


class TestStrategyFactory:
    """Test the data generation strategy factory function."""

    def test_create_strategy_kde(self):
        """Test simulation-based strategy creation for KDE."""
        config = get_lan_config()
        config["estimator"]["type"] = "kde"

        strategy = create_data_generation_strategy(config, model_config["ddm"])

        assert isinstance(strategy, SimulationBasedGenerationStrategy)

    def test_create_strategy_pyddm(self):
        """Test PyDDM strategy creation."""
        pytest.importorskip("pyddm")  # Skip if pyddm not installed

        config = get_lan_config()
        config["estimator"]["type"] = "pyddm"

        strategy = create_data_generation_strategy(config, model_config["ddm"])

        assert isinstance(strategy, PyDDMGenerationStrategy)

    def test_create_strategy_unknown_type(self):
        """Test ValueError for invalid estimator type."""
        config = get_lan_config()
        config["estimator"]["type"] = "unknown_strategy"

        with pytest.raises(ValueError, match="Unknown estimator_type"):
            create_data_generation_strategy(config, model_config["ddm"])

    def test_create_strategy_with_nested_config(self):
        """Verify nested config handling."""
        config = get_lan_config()
        config["estimator"]["type"] = "kde"

        strategy = create_data_generation_strategy(config, model_config["ddm"])

        assert isinstance(strategy, SimulationBasedGenerationStrategy)
        # Verify strategy has access to config
        assert hasattr(strategy, "generator_config")
        assert hasattr(strategy, "model_config")
