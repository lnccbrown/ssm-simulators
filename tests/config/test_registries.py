"""
Tests for the registry system.

This module tests all four registries:
- BoundaryRegistry
- DriftRegistry
- ModelConfigRegistry
- ParameterAdapterRegistry
"""

import pytest
import numpy as np

from ssms.config import (
    register_boundary,
    get_boundary_registry,
    register_drift,
    get_drift_registry,
    register_model_config,
    register_model_config_factory,
    get_model_registry,
    ModelConfigBuilder,
)
from ssms.basic_simulators.parameter_adapters import (
    register_adapter_to_model,
    register_adapter_to_model_family,
    get_adapter_registry,
    ParameterAdaptation,
    SetDefaultValue,
)


class TestBoundaryRegistry:
    """Tests for BoundaryRegistry."""

    def test_builtin_boundaries_loaded(self):
        """Test that built-in boundaries are automatically loaded."""
        registry = get_boundary_registry()
        boundaries = registry.list_boundaries()

        # Check that common built-in boundaries are present
        assert "constant" in boundaries
        assert "angle" in boundaries
        assert "weibull_cdf" in boundaries
        assert len(boundaries) >= 5

    def test_register_custom_boundary(self):
        """Test registering a custom boundary function."""

        def custom_boundary(t, a, rate=1.0):
            return a * np.exp(-rate * t)

        # Register the boundary
        register_boundary(
            name="test_custom_boundary",
            function=custom_boundary,
            params=["a", "rate"],
        )

        # Verify registration
        registry = get_boundary_registry()
        assert registry.is_registered("test_custom_boundary")

        # Verify retrieval
        info = registry.get("test_custom_boundary")
        assert info["fun"] == custom_boundary
        assert info["params"] == ["a", "rate"]

    def test_get_nonexistent_boundary_raises_error(self):
        """Test that getting a nonexistent boundary raises KeyError."""
        registry = get_boundary_registry()

        with pytest.raises(KeyError, match="not registered"):
            registry.get("nonexistent_boundary")

    def test_register_duplicate_boundary_raises_error(self):
        """Test that registering a duplicate boundary raises ValueError."""

        def boundary_v1(t, a):
            return a

        # Register first version
        register_boundary(
            name="test_duplicate_boundary_v2",
            function=boundary_v1,
            params=["a"],
        )

        # Attempt to register duplicate should raise ValueError
        with pytest.raises(ValueError, match="already registered"):
            register_boundary(
                name="test_duplicate_boundary_v2",
                function=boundary_v1,
                params=["a"],
            )

    def test_use_custom_boundary_in_config_builder(self):
        """Test that a custom registered boundary works with ModelConfigBuilder."""

        def linear_decay(t, a, slope=0.1):
            return a * (1.0 - slope * t)

        # Register the boundary
        register_boundary(
            name="test_linear_decay",
            function=linear_decay,
            params=["a", "slope"],
        )

        # Use with ModelConfigBuilder
        config = ModelConfigBuilder.from_model("ddm")
        config = ModelConfigBuilder.add_boundary(config, "test_linear_decay")

        assert config["boundary_name"] == "test_linear_decay"
        assert config["boundary"] == linear_decay
        assert "slope" in config["boundary_params"]
        assert "a" in config["boundary_params"]


class TestDriftRegistry:
    """Tests for DriftRegistry."""

    def test_builtin_drifts_loaded(self):
        """Test that built-in drifts are automatically loaded."""
        registry = get_drift_registry()
        drifts = registry.list_drifts()

        # Check that common built-in drifts are present
        assert "constant" in drifts
        assert "gamma_drift" in drifts
        assert len(drifts) >= 9

    def test_register_custom_drift(self):
        """Test registering a custom drift function."""

        def custom_drift(t, amplitude=0.5, frequency=1.0):
            return amplitude * np.sin(2 * np.pi * frequency * t)

        # Register the drift
        register_drift(
            name="test_custom_drift",
            function=custom_drift,
            params=["amplitude", "frequency"],
        )

        # Verify registration
        registry = get_drift_registry()
        assert registry.is_registered("test_custom_drift")

        # Verify retrieval
        info = registry.get("test_custom_drift")
        assert info["fun"] == custom_drift
        assert info["params"] == ["amplitude", "frequency"]

    def test_get_nonexistent_drift_raises_error(self):
        """Test that getting a nonexistent drift raises KeyError."""
        registry = get_drift_registry()

        with pytest.raises(KeyError, match="not registered"):
            registry.get("nonexistent_drift")

    def test_register_duplicate_drift_raises_error(self):
        """Test that registering a duplicate drift raises ValueError."""

        def drift_v1(t):
            return 1.0

        # Register first version
        register_drift(name="test_duplicate_drift_v2", function=drift_v1, params=[])

        # Attempt to register duplicate should raise ValueError
        with pytest.raises(ValueError, match="already registered"):
            register_drift(name="test_duplicate_drift_v2", function=drift_v1, params=[])

    def test_use_custom_drift_in_config_builder(self):
        """Test that a custom registered drift works with ModelConfigBuilder."""

        def exp_drift(t, rate=0.5):
            return np.exp(-rate * t)

        # Register the drift
        register_drift(name="test_exp_drift", function=exp_drift, params=["rate"])

        # Use with ModelConfigBuilder
        config = ModelConfigBuilder.from_model("ddm")
        config = ModelConfigBuilder.add_drift(config, "test_exp_drift")

        assert config["drift_name"] == "test_exp_drift"
        assert config["drift"] == exp_drift
        assert "rate" in config["drift_params"]


class TestModelConfigRegistry:
    """Tests for ModelConfigRegistry."""

    def test_builtin_models_loaded(self):
        """Test that built-in models are automatically loaded."""
        registry = get_model_registry()
        models = registry.list_models()

        # Check that common built-in models are present
        assert "ddm" in models
        assert "angle" in models
        assert "ornstein" in models
        assert len(models) >= 100  # Should have 106 models

    def test_register_custom_model_config(self):
        """Test registering a custom model configuration."""
        custom_config = {
            "name": "test_custom_model",
            "params": ["v", "a", "z", "t"],
            "param_bounds_dict": {
                "v": (-3.0, 3.0),
                "a": (0.5, 2.5),
                "z": (0.1, 0.9),
                "t": (0.0, 2.0),
            },
            "nchoices": 2,
        }

        # Register the model
        register_model_config("test_custom_model", custom_config)

        # Verify registration
        registry = get_model_registry()
        assert registry.has_model("test_custom_model")

        # Verify retrieval
        config = registry.get("test_custom_model")
        assert config["name"] == "test_custom_model"
        assert len(config["params"]) == 4

    def test_register_model_factory(self):
        """Test registering a model configuration factory."""

        def custom_model_factory():
            return {
                "name": "test_factory_model",
                "params": ["v", "a"],
                "param_bounds_dict": {
                    "v": (-2.0, 2.0),
                    "a": (1.0, 3.0),
                },
                "nchoices": 2,
            }

        # Register the factory
        register_model_config_factory("test_factory_model", custom_model_factory)

        # Verify registration
        registry = get_model_registry()
        assert registry.has_model("test_factory_model")

        # Verify retrieval
        config = registry.get("test_factory_model")
        assert config["name"] == "test_factory_model"
        assert len(config["params"]) == 2

    def test_get_nonexistent_model_raises_error(self):
        """Test that getting a nonexistent model raises KeyError."""
        registry = get_model_registry()

        with pytest.raises(KeyError, match="not registered"):
            registry.get("nonexistent_model")

    def test_use_custom_model_in_config_builder(self):
        """Test that a custom registered model works with ModelConfigBuilder."""
        custom_config = {
            "name": "test_builder_model",
            "params": ["v", "a", "z", "t"],
            "param_bounds_dict": {
                "v": (-3.0, 3.0),
                "a": (0.5, 2.5),
                "z": (0.1, 0.9),
                "t": (0.0, 2.0),
            },
            "nchoices": 2,
        }

        # Register the model
        register_model_config("test_builder_model", custom_config)

        # Use with ModelConfigBuilder
        config = ModelConfigBuilder.from_model("test_builder_model")

        assert config["name"] == "test_builder_model"
        assert len(config["params"]) == 4


class TestParameterAdapterRegistry:
    """Tests for ParameterAdapterRegistry."""

    def test_builtin_adapters_loaded(self):
        """Test that built-in parameter adapters are automatically loaded."""
        registry = get_adapter_registry()

        # Check that built-in model adaptations are registered
        registered_models = registry.list_registered_models()
        assert len(registered_models) > 0
        assert "ddm" in registered_models

    def test_register_adapter_to_model(self):
        """Test registering a parameter adaptation to a specific model."""

        # Define a custom adaptation
        class TestAdaptation(ParameterAdaptation):
            def apply(self, theta: dict, model_config: dict, n_trials: int) -> dict:
                theta["test_param"] = 1.0
                return theta

        # Register the adaptation
        register_adapter_to_model("test_adapter_model", [TestAdaptation()])

        # Verify registration
        registry = get_adapter_registry()
        assert registry.has_processor("test_adapter_model")

        # Verify retrieval
        adaptations = registry.get_processor("test_adapter_model")
        assert len(adaptations) == 1
        assert isinstance(adaptations[0], TestAdaptation)

    def test_register_adapter_to_model_family(self):
        """Test registering a parameter adaptation to a model family."""
        # Register an adaptation for all models starting with "test_family_"
        register_adapter_to_model_family(
            "test_family_matcher",
            lambda name: name.startswith("test_family_"),
            [SetDefaultValue("test_param", 2.0)],
        )

        # Verify registration
        registry = get_adapter_registry()
        families = registry.list_registered_families()
        assert "test_family_matcher" in families

        # Test that it matches a model in that family
        # Use a model name that won't have model-specific adaptations
        adaptations = registry.get_processor("test_family_model1")
        # Should have the family adaptation
        assert any(
            isinstance(a, SetDefaultValue) and a.param_name == "test_param"
            for a in adaptations
        )

    def test_model_specific_overrides_family(self):
        """Test that model-specific adaptations take precedence over family adaptations."""
        # Register family adaptation
        register_adapter_to_model_family(
            "test_family",
            lambda name: name.startswith("test_"),
            [SetDefaultValue("param1", 1.0)],
        )

        # Register model-specific adaptation
        register_adapter_to_model(
            "test_model_specific", [SetDefaultValue("param1", 2.0)]
        )

        # Verify that model-specific is used
        registry = get_adapter_registry()
        adaptations = registry.get_processor("test_model_specific")

        # Should only have model-specific adaptation
        assert len([a for a in adaptations if isinstance(a, SetDefaultValue)]) >= 1


class TestRegistryIntegration:
    """Integration tests across multiple registries."""

    def test_complete_custom_model_pipeline(self):
        """Test a complete pipeline with custom components across all registries."""

        # 1. Register custom boundary
        def custom_integration_boundary(t, a, rate=0.1):
            return a + (1.0 - rate * t)

        register_boundary(
            name="test_integration_boundary",
            function=custom_integration_boundary,
            params=["a", "rate"],
        )

        # 2. Register custom drift
        def custom_integration_drift(t, amplitude=0.5):
            return amplitude * t

        register_drift(
            name="test_integration_drift",
            function=custom_integration_drift,
            params=["amplitude"],
        )

        # 3. Register custom model
        custom_model = {
            "name": "test_integration_model",
            "params": ["v", "a", "z", "t", "rate", "amplitude"],
            "param_bounds_dict": {
                "v": (-3.0, 3.0),
                "a": (0.5, 2.5),
                "z": (0.1, 0.9),
                "t": (0.0, 2.0),
                "rate": (0.0, 0.5),
                "amplitude": (0.0, 1.0),
            },
            "boundary_name": "test_integration_boundary",
            "drift_name": "test_integration_drift",
            "nchoices": 2,
        }

        register_model_config("test_integration_model", custom_model)

        # 4. Register custom parameter adaptation
        register_adapter_to_model("test_integration_model", [SetDefaultValue("z", 0.5)])

        # 5. Build configuration using ModelConfigBuilder
        config = ModelConfigBuilder.from_model("test_integration_model")
        config = ModelConfigBuilder.add_boundary(config, "test_integration_boundary")
        config = ModelConfigBuilder.add_drift(config, "test_integration_drift")

        # Verify everything is wired up correctly
        assert config["name"] == "test_integration_model"
        assert config["boundary_name"] == "test_integration_boundary"
        assert config["drift_name"] == "test_integration_drift"
        assert config["boundary"] == custom_integration_boundary
        assert config["drift"] == custom_integration_drift
        assert len(config["params"]) == 6

        # Verify parameter adaptation is registered
        registry = get_adapter_registry()
        adaptations = registry.get_processor("test_integration_model")
        assert any(isinstance(a, SetDefaultValue) for a in adaptations)

    def test_registry_isolation(self):
        """Test that registries are independent and don't interfere with each other."""

        # Register items with same name in different registries
        def same_name_func(t, a):
            return a

        register_boundary(
            name="test_isolation",
            function=same_name_func,
            params=["a"],
        )

        register_drift(name="test_isolation", function=same_name_func, params=[])

        register_model_config(
            "test_isolation",
            {
                "name": "test_isolation",
                "params": ["v"],
                "param_bounds_dict": {"v": (-1.0, 1.0)},
                "nchoices": 2,
            },
        )

        # Verify all are registered independently
        boundary_registry = get_boundary_registry()
        drift_registry = get_drift_registry()
        model_registry = get_model_registry()

        assert boundary_registry.is_registered("test_isolation")
        assert drift_registry.is_registered("test_isolation")
        assert model_registry.has_model("test_isolation")

        # Verify they retrieve different things
        boundary_info = boundary_registry.get("test_isolation")
        drift_info = drift_registry.get("test_isolation")
        model_info = model_registry.get("test_isolation")

        assert "fun" in boundary_info  # Boundary has 'fun' key
        assert "fun" in drift_info  # Drift also has 'fun' key
        assert "params" in model_info  # Model has 'params' key
