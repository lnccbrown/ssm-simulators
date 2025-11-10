"""
Tests for theta processor integration with Simulator class.

These tests verify that the Simulator class correctly uses theta processors
and custom transformations.
"""

import numpy as np
import pytest

from ssms import Simulator
from ssms.basic_simulators.modular_theta_processor import ModularThetaProcessor
from ssms.basic_simulators.theta_processor import SimpleThetaProcessor
from ssms.basic_simulators.theta_transforms import (
    LambdaTransformation,
    SetDefaultValue,
)


class TestThetaProcessorIntegration:
    """Test theta processor integration with Simulator."""
    
    def test_default_uses_modular_processor(self):
        """Test that Simulator uses ModularThetaProcessor by default."""
        sim = Simulator("ddm")
        
        assert isinstance(sim.theta_processor, ModularThetaProcessor)
    
    def test_custom_theta_processor(self):
        """Test that custom theta processor can be provided."""
        custom_processor = SimpleThetaProcessor()
        sim = Simulator("ddm", theta_processor=custom_processor)
        
        assert sim.theta_processor is custom_processor
        assert isinstance(sim.theta_processor, SimpleThetaProcessor)
    
    def test_simulate_with_default_processor(self):
        """Test simulation with default ModularThetaProcessor."""
        sim = Simulator("lba2")
        
        theta = {
            'v0': np.array([0.5]),
            'v1': np.array([0.6]),
            'A': np.array([0.5]),
            'b': np.array([1.0])
        }
        
        result = sim.simulate(theta, n_samples=10, random_state=42)
        
        # Should complete successfully
        assert 'rts' in result
        assert 'choices' in result
        assert len(result['rts']) == 10
    
    def test_simulate_with_legacy_processor(self):
        """Test simulation with legacy SimpleThetaProcessor."""
        sim = Simulator("lba2", theta_processor=SimpleThetaProcessor())
        
        theta = {
            'v0': np.array([0.5]),
            'v1': np.array([0.6]),
            'A': np.array([0.5]),
            'b': np.array([1.0])
        }
        
        result = sim.simulate(theta, n_samples=10, random_state=42)
        
        # Should produce same results as default
        assert 'rts' in result
        assert 'choices' in result
        assert len(result['rts']) == 10
    
    def test_custom_transformations(self):
        """Test adding custom theta transformations."""
        custom_transform = SetDefaultValue("custom_param", 999)
        
        sim = Simulator("ddm", theta_transforms=[custom_transform])
        
        theta = {'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3}
        
        # Simulate to trigger theta processing
        result = sim.simulate(theta, n_samples=10, random_state=42)
        
        # Verify simulation worked
        assert 'rts' in result
        assert 'choices' in result
    
    def test_multiple_custom_transformations(self):
        """Test adding multiple custom transformations."""
        transforms = [
            SetDefaultValue("param1", 100),
            SetDefaultValue("param2", 200),
            LambdaTransformation(
                lambda theta, cfg, n: theta.update({"param3": 300}) or theta,
                name="set_param3"
            ),
        ]
        
        sim = Simulator("ddm", theta_transforms=transforms)
        
        theta = {'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3}
        result = sim.simulate(theta, n_samples=10, random_state=42)
        
        # Verify simulation worked
        assert 'rts' in result
        assert 'choices' in result
    
    def test_custom_transforms_only_with_modular_processor(self):
        """Test that custom transforms are only applied with ModularThetaProcessor."""
        # With SimpleThetaProcessor, custom transforms should be ignored
        custom_transform = SetDefaultValue("custom_param", 999)
        
        sim = Simulator(
            "ddm",
            theta_processor=SimpleThetaProcessor(),
            theta_transforms=[custom_transform]  # Should be ignored
        )
        
        theta = {'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3}
        result = sim.simulate(theta, n_samples=10, random_state=42)
        
        # Should still work (transforms just ignored)
        assert 'rts' in result
        assert 'choices' in result


class TestProcessorEquivalence:
    """Test that both processors produce equivalent results."""
    
    @pytest.mark.parametrize("model_name", ["ddm", "angle", "race_3"])
    def test_processor_equivalence_simple_models(self, model_name):
        """Test that ModularThetaProcessor produces same results as SimpleThetaProcessor."""
        # Generate appropriate theta for each model
        if model_name == "ddm":
            theta = {'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3}
        elif model_name == "angle":
            theta = {'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3, 'theta': 0.5}
        elif model_name == "race_3":
            theta = {'v0': 0.5, 'v1': 0.6, 'v2': 0.7,
                    'z0': 0.3, 'z1': 0.4, 'z2': 0.5,
                    'a': 1.0, 't': 0.3}
        else:
            theta = {'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3}
        
        # Simulate with both processors
        sim_modular = Simulator(model_name)  # Default ModularThetaProcessor
        sim_legacy = Simulator(model_name, theta_processor=SimpleThetaProcessor())
        
        random_state = 42
        result_modular = sim_modular.simulate(theta, n_samples=100, random_state=random_state)
        result_legacy = sim_legacy.simulate(theta, n_samples=100, random_state=random_state)
        
        # Results should be identical (same random seed)
        np.testing.assert_array_equal(result_modular['rts'], result_legacy['rts'])
        np.testing.assert_array_equal(result_modular['choices'], result_legacy['choices'])
    
    def test_lba_models_work_with_both_processors(self):
        """Test that LBA models work with both processors (may have different RNG behavior)."""
        theta = {'v0': 0.5, 'v1': 0.6, 'A': 0.5, 'b': 1.0}
        
        # Both should complete successfully
        sim_modular = Simulator("lba2")
        result_modular = sim_modular.simulate(theta, n_samples=100, random_state=42)
        
        sim_legacy = Simulator("lba2", theta_processor=SimpleThetaProcessor())
        result_legacy = sim_legacy.simulate(theta, n_samples=100, random_state=42)
        
        # Both should return valid results
        assert 'rts' in result_modular and len(result_modular['rts']) == 100
        assert 'choices' in result_modular and len(result_modular['choices']) == 100
        assert 'rts' in result_legacy and len(result_legacy['rts']) == 100
        assert 'choices' in result_legacy and len(result_legacy['choices']) == 100
        
        # Statistical properties should be similar (not identical due to LBA start point randomness)
        assert np.abs(np.mean(result_modular['rts']) - np.mean(result_legacy['rts'])) < 0.5
        assert np.abs(np.mean(result_modular['choices']) - np.mean(result_legacy['choices'])) < 0.3


class TestThetaProcessorProperty:
    """Test the theta_processor property."""
    
    def test_theta_processor_property_returns_processor(self):
        """Test that theta_processor property returns the processor."""
        sim = Simulator("ddm")
        
        processor = sim.theta_processor
        assert isinstance(processor, ModularThetaProcessor)
    
    def test_theta_processor_property_with_custom(self):
        """Test theta_processor property with custom processor."""
        custom = SimpleThetaProcessor()
        sim = Simulator("ddm", theta_processor=custom)
        
        assert sim.theta_processor is custom


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def test_existing_code_still_works(self):
        """Test that existing code without theta_processor still works."""
        # Old code that doesn't specify theta_processor
        sim = Simulator("ddm")
        result = sim.simulate(
            theta={'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3},
            n_samples=100,
            random_state=42
        )
        
        assert 'rts' in result
        assert 'choices' in result
        assert len(result['rts']) == 100
    
    def test_all_existing_constructor_params_work(self):
        """Test that all existing constructor parameters still work."""
        sim = Simulator(
            model="ddm",
            boundary="constant",
            drift="constant",
            n_params=4,
        )
        
        result = sim.simulate(
            theta={'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3},
            n_samples=10,
            random_state=42
        )
        
        assert 'rts' in result
        assert 'choices' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

