"""
Tests for theta processor integration with Simulator class.

These tests verify that the Simulator class correctly uses theta processors
and custom transformations.
"""

import numpy as np
import pytest

from ssms import Simulator
from ssms.basic_simulators.modular_parameter_simulator_adapter import (
    ModularParameterSimulatorAdapter,
)
from ssms.basic_simulators.parameter_adapters import (
    LambdaAdaptation,
    SetDefaultValue,
)


class TestThetaProcessorIntegration:
    """Test theta processor integration with Simulator."""

    def test_default_uses_modular_processor(self):
        """Test that Simulator uses ModularParameterSimulatorAdapter by default."""
        sim = Simulator("ddm")

        assert isinstance(sim.parameter_adapter, ModularParameterSimulatorAdapter)

    def test_custom_theta_processor(self):
        """Test that custom theta processor can be provided."""
        custom_processor = ModularParameterSimulatorAdapter()
        sim = Simulator("ddm", parameter_adapter=custom_processor)

        assert sim.parameter_adapter is custom_processor
        assert isinstance(sim.parameter_adapter, ModularParameterSimulatorAdapter)

    def test_simulate_with_default_processor(self):
        """Test simulation with default ModularParameterSimulatorAdapter."""
        sim = Simulator("lba2")

        theta = {
            "v0": np.array([0.5]),
            "v1": np.array([0.6]),
            "A": np.array([0.5]),
            "b": np.array([1.0]),
        }

        result = sim.simulate(theta, n_samples=10, random_state=42)

        # Should complete successfully
        assert "rts" in result
        assert "choices" in result
        assert len(result["rts"]) == 10

    def test_custom_transformations(self):
        """Test adding custom theta transformations."""
        custom_transform = SetDefaultValue("custom_param", 999)

        sim = Simulator("ddm", parameter_adaptations=[custom_transform])

        theta = {"v": 0.5, "a": 1.0, "z": 0.5, "t": 0.3}

        # Simulate to trigger theta processing
        result = sim.simulate(theta, n_samples=10, random_state=42)

        # Verify simulation worked
        assert "rts" in result
        assert "choices" in result

    def test_multiple_custom_transformations(self):
        """Test adding multiple custom transformations."""
        transforms = [
            SetDefaultValue("param1", 100),
            SetDefaultValue("param2", 200),
            LambdaAdaptation(
                lambda theta, cfg, n: theta.update({"param3": 300}) or theta,
                name="set_param3",
            ),
        ]

        sim = Simulator("ddm", parameter_adaptations=transforms)

        theta = {"v": 0.5, "a": 1.0, "z": 0.5, "t": 0.3}
        result = sim.simulate(theta, n_samples=10, random_state=42)

        # Verify simulation worked
        assert "rts" in result
        assert "choices" in result


class TestProcessorEquivalence:
    """Test that processors produce consistent results."""

    @pytest.mark.parametrize("model_name", ["ddm", "angle", "race_3"])
    def test_processor_consistency_simple_models(self, model_name):
        """Test that ModularParameterSimulatorAdapter produces consistent results."""
        # Generate appropriate theta for each model
        if model_name == "ddm":
            theta = {"v": 0.5, "a": 1.0, "z": 0.5, "t": 0.3}
        elif model_name == "angle":
            theta = {"v": 0.5, "a": 1.0, "z": 0.5, "t": 0.3, "theta": 0.5}
        elif model_name == "race_3":
            theta = {
                "v0": 0.5,
                "v1": 0.6,
                "v2": 0.7,
                "z0": 0.3,
                "z1": 0.4,
                "z2": 0.5,
                "a": 1.0,
                "t": 0.3,
            }
        else:
            theta = {"v": 0.5, "a": 1.0, "z": 0.5, "t": 0.3}

        # Simulate twice with same seed
        sim = Simulator(model_name)

        random_state = 42
        result1 = sim.simulate(theta, n_samples=100, random_state=random_state)
        result2 = sim.simulate(theta, n_samples=100, random_state=random_state)

        # Results should be identical (same random seed)
        np.testing.assert_array_equal(result1["rts"], result2["rts"])
        np.testing.assert_array_equal(result1["choices"], result2["choices"])

    def test_lba_models_work(self):
        """Test that LBA models work correctly."""
        theta = {"v0": 0.5, "v1": 0.6, "A": 0.5, "b": 1.0}

        sim = Simulator("lba2")
        result = sim.simulate(theta, n_samples=100, random_state=42)

        # Should return valid results
        assert "rts" in result and len(result["rts"]) == 100
        assert "choices" in result and len(result["choices"]) == 100


class TestThetaProcessorProperty:
    """Test the theta_processor property."""

    def test_theta_processor_property_returns_processor(self):
        """Test that theta_processor property returns the processor."""
        sim = Simulator("ddm")

        processor = sim.parameter_adapter
        assert isinstance(processor, ModularParameterSimulatorAdapter)

    def test_theta_processor_property_with_custom(self):
        """Test theta_processor property with custom processor."""
        custom = ModularParameterSimulatorAdapter()
        sim = Simulator("ddm", parameter_adapter=custom)

        assert sim.parameter_adapter is custom


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_existing_code_still_works(self):
        """Test that existing code without theta_processor still works."""
        # Old code that doesn't specify theta_processor
        sim = Simulator("ddm")
        result = sim.simulate(
            theta={"v": 0.5, "a": 1.0, "z": 0.5, "t": 0.3},
            n_samples=100,
            random_state=42,
        )

        assert "rts" in result
        assert "choices" in result
        assert len(result["rts"]) == 100

    def test_all_existing_constructor_params_work(self):
        """Test that all existing constructor parameters still work."""
        sim = Simulator(
            model="ddm",
            boundary="constant",
            drift="constant",
            n_params=4,
        )

        result = sim.simulate(
            theta={"v": 0.5, "a": 1.0, "z": 0.5, "t": 0.3},
            n_samples=10,
            random_state=42,
        )

        assert "rts" in result
        assert "choices" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
