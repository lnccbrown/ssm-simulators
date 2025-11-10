"""
Basic tests for theta transformation system.

These tests verify the core functionality of individual transformations,
the registry system, and the modular processor.
"""

import numpy as np
import pytest

from ssms.basic_simulators.modular_theta_processor import ModularThetaProcessor
from ssms.basic_simulators.theta_transforms import (
    ColumnStackParameters,
    DeleteParameters,
    ExpandDimension,
    RenameParameter,
    SetDefaultValue,
    SetZeroArray,
    ThetaProcessorRegistry,
    TileArray,
)


class TestIndividualTransformations:
    """Test individual transformation classes."""
    
    def test_set_default_value(self):
        """Test SetDefaultValue transformation."""
        transform = SetDefaultValue("nact", 3)
        theta = {}
        result = transform.apply(theta, {}, n_trials=5)
        
        assert "nact" in result
        assert result["nact"].shape == (5,)
        assert np.all(result["nact"] == 3.0)
        assert result["nact"].dtype == np.float32
    
    def test_set_default_value_preserves_existing(self):
        """SetDefaultValue should not override existing values."""
        transform = SetDefaultValue("v", 0.5)
        theta = {"v": np.array([1.0, 2.0])}
        result = transform.apply(theta, {}, n_trials=2)
        
        # Should keep original value
        assert np.all(result["v"] == [1.0, 2.0])
    
    def test_expand_dimension(self):
        """Test ExpandDimension transformation."""
        transform = ExpandDimension(["a", "t"])
        theta = {
            "a": np.array([1.0, 2.0]),
            "t": np.array([0.3, 0.3])
        }
        result = transform.apply(theta, {}, n_trials=2)
        
        assert result["a"].shape == (2, 1)
        assert result["t"].shape == (2, 1)
    
    def test_expand_dimension_missing_param(self):
        """ExpandDimension should handle missing parameters gracefully."""
        transform = ExpandDimension(["a", "missing"])
        theta = {"a": np.array([1.0, 2.0])}
        result = transform.apply(theta, {}, n_trials=2)
        
        assert result["a"].shape == (2, 1)
        assert "missing" not in result
    
    def test_column_stack_parameters(self):
        """Test ColumnStackParameters transformation."""
        transform = ColumnStackParameters(["v0", "v1", "v2"], "v")
        theta = {
            "v0": np.array([0.5]),
            "v1": np.array([0.6]),
            "v2": np.array([0.7])
        }
        result = transform.apply(theta, {}, n_trials=1)
        
        assert "v" in result
        assert result["v"].shape == (1, 3)
        assert np.allclose(result["v"], [[0.5, 0.6, 0.7]])
        
        # Source params should be deleted by default
        assert "v0" not in result
        assert "v1" not in result
        assert "v2" not in result
    
    def test_column_stack_parameters_keep_sources(self):
        """Test ColumnStackParameters with delete_sources=False."""
        transform = ColumnStackParameters(
            ["v0", "v1"], "v", delete_sources=False
        )
        theta = {"v0": np.array([0.5]), "v1": np.array([0.6])}
        result = transform.apply(theta, {}, n_trials=1)
        
        assert "v" in result
        assert "v0" in result
        assert "v1" in result
    
    def test_rename_parameter(self):
        """Test RenameParameter transformation."""
        transform = RenameParameter("A", "z")
        theta = {"A": np.array([0.5])}
        result = transform.apply(theta, {}, n_trials=1)
        
        assert "z" in result
        assert "A" not in result
        assert result["z"][0] == 0.5
    
    def test_rename_parameter_with_transform(self):
        """Test RenameParameter with transform function."""
        transform = RenameParameter(
            "A", "z",
            transform_fn=lambda x: np.expand_dims(x, axis=1)
        )
        theta = {"A": np.array([0.5, 0.6])}
        result = transform.apply(theta, {}, n_trials=2)
        
        assert result["z"].shape == (2, 1)
    
    def test_delete_parameters(self):
        """Test DeleteParameters transformation."""
        transform = DeleteParameters(["A", "b"])
        theta = {"v": [0.5], "A": [0.5], "b": [1.0]}
        result = transform.apply(theta, {}, n_trials=1)
        
        assert "v" in result
        assert "A" not in result
        assert "b" not in result
    
    def test_set_zero_array_1d(self):
        """Test SetZeroArray with 1D array."""
        transform = SetZeroArray("v")
        theta = {}
        result = transform.apply(theta, {}, n_trials=5)
        
        assert "v" in result
        assert result["v"].shape == (5,)
        assert np.all(result["v"] == 0.0)
    
    def test_set_zero_array_2d(self):
        """Test SetZeroArray with 2D array."""
        transform = SetZeroArray("v", shape=(None, 3))
        theta = {}
        result = transform.apply(theta, {}, n_trials=5)
        
        assert "v" in result
        assert result["v"].shape == (5, 3)
        assert np.all(result["v"] == 0.0)
    
    def test_tile_array(self):
        """Test TileArray transformation."""
        transform = TileArray("z", 0.5)
        theta = {}
        result = transform.apply(theta, {}, n_trials=3)
        
        assert "z" in result
        assert result["z"].shape == (3,)
        assert np.all(result["z"] == 0.5)


class TestRegistry:
    """Test registry functionality."""
    
    def test_register_and_get_model(self):
        """Test registering and retrieving a model."""
        registry = ThetaProcessorRegistry()
        transforms = [SetDefaultValue("nact", 2)]
        
        registry.register_model("test_model", transforms)
        result = registry.get_processor("test_model")
        
        assert len(result) == 1
        assert isinstance(result[0], SetDefaultValue)
    
    def test_register_family(self):
        """Test family registration and matching."""
        registry = ThetaProcessorRegistry()
        transforms = [ExpandDimension(["a"])]
        
        registry.register_family(
            "race_2",
            lambda m: m.startswith("race_") and m.endswith("_2"),
            transforms
        )
        
        # Should match
        result = registry.get_processor("race_no_bias_2")
        assert len(result) == 1
        
        # Should not match
        result = registry.get_processor("race_3")
        assert len(result) == 0
    
    def test_exact_match_priority(self):
        """Test that exact matches take priority over family matches."""
        registry = ThetaProcessorRegistry()
        
        # Register family
        registry.register_family(
            "race_2",
            lambda m: m.startswith("race_") and m.endswith("_2"),
            [SetDefaultValue("family", 1)]
        )
        
        # Register specific model
        registry.register_model(
            "race_special_2",
            [SetDefaultValue("exact", 2)]
        )
        
        # Specific model should use exact match
        result = registry.get_processor("race_special_2")
        assert len(result) == 1
        assert result[0].param_name == "exact"
        
        # Other models should use family match
        result = registry.get_processor("race_no_bias_2")
        assert len(result) == 1
        assert result[0].param_name == "family"
    
    def test_get_processor_no_match(self):
        """Test get_processor with no matching registration."""
        registry = ThetaProcessorRegistry()
        result = registry.get_processor("unknown_model")
        
        assert result == []
    
    def test_has_processor(self):
        """Test has_processor method."""
        registry = ThetaProcessorRegistry()
        registry.register_model("test_model", [])
        
        assert registry.has_processor("test_model")
        assert not registry.has_processor("unknown_model")
    
    def test_list_registered_models(self):
        """Test listing registered models."""
        registry = ThetaProcessorRegistry()
        registry.register_model("model1", [])
        registry.register_model("model2", [])
        registry.register_family("family1", lambda m: True, [])
        
        models = registry.list_registered_models()
        assert "model1" in models
        assert "model2" in models
        assert "family1" not in models
    
    def test_list_registered_families(self):
        """Test listing registered families."""
        registry = ThetaProcessorRegistry()
        registry.register_model("model1", [])
        registry.register_family("family1", lambda m: True, [])
        registry.register_family("family2", lambda m: True, [])
        
        families = registry.list_registered_families()
        assert "family1" in families
        assert "family2" in families
        assert "model1" not in families


class TestModularThetaProcessor:
    """Test ModularThetaProcessor."""
    
    def test_processor_with_default_registry(self):
        """Test processor uses default registry."""
        processor = ModularThetaProcessor()
        assert processor.registry is not None
        
        # Should have registered models
        assert processor.registry.has_processor("lba2")
        assert processor.registry.has_processor("ddm")
    
    def test_processor_with_custom_registry(self):
        """Test processor with custom registry."""
        custom_registry = ThetaProcessorRegistry()
        custom_registry.register_model("custom_model", [
            SetDefaultValue("custom_param", 42)
        ])
        
        processor = ModularThetaProcessor(registry=custom_registry)
        assert processor.registry.has_processor("custom_model")
    
    def test_process_theta_lba2(self):
        """Test processing theta for LBA2 model."""
        processor = ModularThetaProcessor()
        theta = {
            "v0": np.array([0.5]),
            "v1": np.array([0.6]),
            "A": np.array([0.5]),
            "b": np.array([1.0])
        }
        model_config = {"name": "lba2"}
        
        result = processor.process_theta(theta, model_config, n_trials=1)
        
        # Should have nact
        assert "nact" in result
        
        # Should have stacked v
        assert "v" in result
        assert result["v"].shape == (1, 2)
        
        # Should have renamed and expanded z and a
        assert "z" in result
        assert "a" in result
        assert result["z"].shape == (1, 1)
        assert result["a"].shape == (1, 1)
        
        # A and b should be deleted
        assert "A" not in result
        assert "b" not in result
        
        # Should have t set to zero
        assert "t" in result
        assert result["t"][0] == 0.0
    
    def test_process_theta_ddm(self):
        """Test processing theta for DDM (no transformations)."""
        processor = ModularThetaProcessor()
        theta = {
            "v": np.array([0.5]),
            "a": np.array([1.0]),
            "z": np.array([0.5]),
            "t": np.array([0.3])
        }
        model_config = {"name": "ddm"}
        
        result = processor.process_theta(theta, model_config, n_trials=1)
        
        # Should be unchanged
        assert result["v"][0] == 0.5
        assert result["a"][0] == 1.0
        assert result["z"][0] == 0.5
        assert result["t"][0] == 0.3
    
    def test_process_theta_race_3(self):
        """Test processing theta for race_3 model."""
        processor = ModularThetaProcessor()
        theta = {
            "v0": np.array([0.5]),
            "v1": np.array([0.6]),
            "v2": np.array([0.7]),
            "z0": np.array([0.3]),
            "z1": np.array([0.4]),
            "z2": np.array([0.5]),
            "a": np.array([1.0]),
            "t": np.array([0.3])
        }
        model_config = {"name": "race_3"}
        
        result = processor.process_theta(theta, model_config, n_trials=1)
        
        # Should have stacked v and z
        assert "v" in result
        assert "z" in result
        assert result["v"].shape == (1, 3)
        assert result["z"].shape == (1, 3)
        
        # Should have expanded a and t
        assert result["a"].shape == (1, 1)
        assert result["t"].shape == (1, 1)


class TestTransformationPipelines:
    """Test combining multiple transformations."""
    
    def test_transformation_sequence(self):
        """Test applying transformations in sequence."""
        transforms = [
            ColumnStackParameters(["v0", "v1"], "v"),
            ExpandDimension(["a"]),
            SetDefaultValue("nact", 2)
        ]
        
        theta = {
            "v0": np.array([0.5]),
            "v1": np.array([0.6]),
            "a": np.array([1.0])
        }
        
        # Apply transforms in sequence
        for transform in transforms:
            theta = transform.apply(theta, {}, n_trials=1)
        
        assert "v" in theta
        assert theta["v"].shape == (1, 2)
        assert theta["a"].shape == (1, 1)
        assert "nact" in theta
        assert theta["nact"][0] == 2.0
    
    def test_transformation_order_matters(self):
        """Test that transformation order matters."""
        theta1 = {"A": np.array([0.5])}
        theta2 = {"A": np.array([0.5])}
        
        # Order 1: Rename then expand
        theta1 = RenameParameter("A", "z").apply(theta1, {}, 1)
        theta1 = ExpandDimension(["z"]).apply(theta1, {}, 1)
        
        # Order 2: Expand then rename (won't work as expected)
        theta2 = ExpandDimension(["A"]).apply(theta2, {}, 1)
        theta2 = RenameParameter("A", "z").apply(theta2, {}, 1)
        
        # Both should have z, but shapes may differ
        assert "z" in theta1
        assert "z" in theta2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

