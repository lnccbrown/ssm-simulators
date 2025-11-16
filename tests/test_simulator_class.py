"""
Tests for the Simulator class and ConfigBuilder.

This module tests the class-based simulator interface including:
- Initialization with different model specifications
- Custom boundary and drift functions
- Custom simulator functions
- Parameter validation
- ConfigBuilder utilities
"""

import numpy as np
import pytest

from ssms import Simulator
from ssms.basic_simulators.simulator import simulator
from ssms.config import ConfigBuilder


class TestSimulatorInitialization:
    """Test Simulator initialization with various inputs."""

    def test_init_with_string_model(self):
        """Test initialization with string model name."""
        sim = Simulator("ddm")
        assert sim.config["name"] == "ddm"
        assert "params" in sim.config
        assert "simulator" in sim.config

    def test_init_with_invalid_model(self):
        """Test initialization with invalid model name."""
        with pytest.raises(ValueError, match="Unknown model"):
            Simulator("invalid_model_name")

    def test_init_with_config_dict(self):
        """Test initialization with configuration dictionary."""
        config = ConfigBuilder.from_model("ddm")
        sim = Simulator(config)
        assert sim.config["name"] == "ddm"

    def test_init_with_config_overrides(self):
        """Test initialization with parameter overrides."""
        sim = Simulator("ddm", param_bounds=[[-4, 0.3, 0.1, 0], [4, 3.0, 0.9, 2.0]])
        assert sim.config["param_bounds"][0][0] == -4
        assert sim.config["param_bounds"][1][0] == 4

    def test_init_with_custom_simulator_function(self):
        """Test initialization with custom simulator function."""

        def my_sim(v, a, z, t, max_t=20, n_samples=1000, **kwargs):
            rts = np.random.exponential(1 / abs(v), n_samples) + t
            choices = np.where(np.random.random(n_samples) < z, 1, -1)
            return {
                "rts": rts,
                "choices": choices,
                "metadata": {"model": "custom", "n_samples": n_samples},
            }

        sim = Simulator(
            simulator_function=my_sim, params=["v", "a", "z", "t"], nchoices=2
        )
        assert sim.config["params"] == ["v", "a", "z", "t"]
        assert sim.config["nchoices"] == 2
        assert sim.config["simulator"] == my_sim

    def test_init_missing_required_params_custom_sim(self):
        """Test that missing required params raises error for custom simulator."""

        def my_sim(v, a, **kwargs):
            return {"rts": np.array([0.5]), "choices": np.array([1]), "metadata": {}}

        with pytest.raises(ValueError, match="must provide 'params'"):
            Simulator(simulator_function=my_sim)

        with pytest.raises(ValueError, match="must provide 'nchoices'"):
            Simulator(simulator_function=my_sim, params=["v", "a"])


class TestCustomBoundaryFunctions:
    """Test custom boundary function support."""

    def test_custom_boundary_function(self):
        """Test using a custom boundary function."""

        def my_boundary(t, theta):
            return np.sin(theta * t)

        sim = Simulator("ddm", boundary=my_boundary, boundary_params=["theta"])
        assert sim.config["boundary"] == my_boundary
        assert sim.config["boundary_params"] == ["theta"]

    def test_boundary_by_name(self):
        """Test using boundary by name."""
        sim = Simulator("ddm", boundary="angle")
        assert sim.config["boundary_name"] == "angle"
        assert "boundary" in sim.config

    def test_invalid_boundary_name(self):
        """Test that invalid boundary name raises error."""
        with pytest.raises(ValueError, match="Unknown boundary"):
            Simulator("ddm", boundary="nonexistent_boundary")

    def test_boundary_without_t_parameter(self):
        """Test that boundary function must accept 't' parameter."""

        def bad_boundary(x):  # Wrong parameter name
            return x

        with pytest.raises(ValueError, match="must accept 't' as first"):
            Simulator("ddm", boundary=bad_boundary)


class TestCustomDriftFunctions:
    """Test custom drift function support."""

    def test_custom_drift_function(self):
        """Test using a custom drift function."""

        def my_drift(t, scale):
            return scale * np.exp(-t)

        sim = Simulator("gamma_drift", drift=my_drift, drift_params=["scale"])
        assert sim.config["drift"] == my_drift
        assert sim.config["drift_params"] == ["scale"]

    def test_drift_by_name(self):
        """Test using drift by name."""
        sim = Simulator("gamma_drift", drift="constant")
        assert sim.config["drift_name"] == "constant"
        assert "drift" in sim.config

    def test_invalid_drift_name(self):
        """Test that invalid drift name raises error."""
        with pytest.raises(ValueError, match="Unknown drift"):
            Simulator("gamma_drift", drift="nonexistent_drift")

    def test_drift_without_t_parameter(self):
        """Test that drift function must accept 't' parameter."""

        def bad_drift(x):  # Wrong parameter name
            return x

        with pytest.raises(ValueError, match="must accept 't' as first"):
            Simulator("gamma_drift", drift=bad_drift)


class TestSimulation:
    """Test simulation functionality."""

    def test_basic_simulation_dict_theta(self):
        """Test basic simulation with dict theta."""
        sim = Simulator("ddm")
        results = sim.simulate(
            theta={"v": 0.5, "a": 1.0, "z": 0.5, "t": 0.3}, n_samples=100
        )

        assert "rts" in results
        assert "choices" in results
        assert "metadata" in results
        assert len(results["rts"]) == 100
        assert len(results["choices"]) == 100

    def test_basic_simulation_array_theta(self):
        """Test simulation with array theta."""
        sim = Simulator("ddm")
        results = sim.simulate(theta=np.array([0.5, 1.0, 0.5, 0.3]), n_samples=100)

        assert "rts" in results
        assert "choices" in results
        assert len(results["rts"]) == 100

    def test_simulation_multiple_parameter_sets(self):
        """Test simulation with multiple parameter sets."""
        sim = Simulator("ddm")
        results = sim.simulate(
            theta={"v": [0.5, 1.0], "a": [1.0, 1.5], "z": [0.5, 0.5], "t": [0.3, 0.3]},
            n_samples=50,
        )

        assert "rts" in results
        assert "choices" in results
        # Should have 2 parameter sets × 50 samples

    def test_simulation_with_custom_simulator(self):
        """Test simulation using custom simulator function."""

        def my_sim(v, a, z, t, max_t=20, n_samples=1000, n_trials=1, **kwargs):
            rts = np.random.exponential(1 / abs(v[0]), (n_samples, n_trials, 1)) + t[0]
            choices = np.where(np.random.random((n_samples, n_trials, 1)) < z[0], 1, -1)
            return {
                "rts": rts,
                "choices": choices,
                "metadata": {
                    "model": "custom",
                    "n_samples": n_samples,
                    "possible_choices": [1, -1],
                },
            }

        sim = Simulator(
            simulator_function=my_sim, params=["v", "a", "z", "t"], nchoices=2
        )
        results = sim.simulate(
            theta={"v": 0.5, "a": 1.0, "z": 0.5, "t": 0.3}, n_samples=50
        )

        assert "rts" in results
        assert "choices" in results
        assert results["metadata"]["model"] == "custom"

    def test_simulation_reproducibility(self):
        """Test that simulations are reproducible with random_state."""
        sim = Simulator("ddm")
        theta = {"v": 0.5, "a": 1.0, "z": 0.5, "t": 0.3}

        results1 = sim.simulate(theta, n_samples=100, random_state=42)
        results2 = sim.simulate(theta, n_samples=100, random_state=42)

        np.testing.assert_array_equal(results1["rts"], results2["rts"])
        np.testing.assert_array_equal(results1["choices"], results2["choices"])


class TestConfigBuilder:
    """Test ConfigBuilder utility class."""

    def test_from_model(self):
        """Test ConfigBuilder.from_model()."""
        config = ConfigBuilder.from_model("ddm")
        assert config["name"] == "ddm"
        assert "params" in config
        assert "simulator" in config

    def test_from_model_with_overrides(self):
        """Test ConfigBuilder.from_model() with overrides."""
        config = ConfigBuilder.from_model(
            "ddm", param_bounds=[[-4, 0.3, 0.1, 0], [4, 3.0, 0.9, 2.0]]
        )
        assert config["param_bounds"][0][0] == -4

    def test_from_model_invalid_name(self):
        """Test ConfigBuilder.from_model() with invalid name."""
        with pytest.raises(ValueError, match="Unknown model"):
            ConfigBuilder.from_model("invalid_name")

    def test_from_scratch(self):
        """Test ConfigBuilder.from_scratch()."""

        def my_sim(**kwargs):
            return {"rts": np.array([0.5]), "choices": np.array([1]), "metadata": {}}

        config = ConfigBuilder.from_scratch(
            name="my_model", params=["v", "a"], simulator_function=my_sim, nchoices=2
        )

        assert config["name"] == "my_model"
        assert config["params"] == ["v", "a"]
        assert config["n_params"] == 2
        assert config["nchoices"] == 2
        assert config["simulator"] == my_sim

    def test_minimal_config(self):
        """Test ConfigBuilder.minimal_config()."""

        def my_sim(**kwargs):
            return {"rts": np.array([0.5]), "choices": np.array([1]), "metadata": {}}

        config = ConfigBuilder.minimal_config(
            params=["v", "a"], simulator_function=my_sim
        )

        assert config["params"] == ["v", "a"]
        assert config["nchoices"] == 2
        assert config["simulator"] == my_sim

    def test_validate_config_valid(self):
        """Test ConfigBuilder.validate_config() with valid config."""

        def my_sim(**kwargs):
            return {}

        config = ConfigBuilder.minimal_config(
            params=["v", "a"], simulator_function=my_sim
        )

        is_valid, errors = ConfigBuilder.validate_config(config)
        assert is_valid
        assert len(errors) == 0

    def test_validate_config_missing_fields(self):
        """Test ConfigBuilder.validate_config() with missing fields."""
        config = {"params": ["v", "a"]}  # Missing nchoices and simulator

        is_valid, errors = ConfigBuilder.validate_config(config)
        assert not is_valid
        assert len(errors) > 0

    def test_validate_config_inconsistent_params(self):
        """Test ConfigBuilder.validate_config() with inconsistent params."""

        def my_sim(**kwargs):
            return {}

        config = {
            "params": ["v", "a"],
            "n_params": 3,  # Inconsistent!
            "nchoices": 2,
            "simulator": my_sim,
        }

        is_valid, errors = ConfigBuilder.validate_config(config)
        assert not is_valid
        assert any("Inconsistent n_params" in err for err in errors)

    def test_add_boundary(self):
        """Test ConfigBuilder.add_boundary()."""
        config = ConfigBuilder.from_model("ddm")
        config = ConfigBuilder.add_boundary(config, "angle", ["theta"])

        assert config["boundary_name"] == "angle"
        assert config["boundary_params"] == ["theta"]

    def test_add_boundary_custom_function(self):
        """Test ConfigBuilder.add_boundary() with custom function."""

        def my_boundary(t, theta):
            return np.sin(theta * t)

        config = ConfigBuilder.from_model("ddm")
        config = ConfigBuilder.add_boundary(config, my_boundary, ["theta"])

        assert config["boundary"] == my_boundary
        assert config["boundary_params"] == ["theta"]

    def test_add_drift(self):
        """Test ConfigBuilder.add_drift()."""
        config = ConfigBuilder.from_model("gamma_drift")
        config = ConfigBuilder.add_drift(config, "constant")

        assert config["drift_name"] == "constant"

    def test_add_drift_custom_function(self):
        """Test ConfigBuilder.add_drift() with custom function."""

        def my_drift(t, scale):
            return scale * np.exp(-t)

        config = ConfigBuilder.from_model("gamma_drift")
        config = ConfigBuilder.add_drift(config, my_drift, ["scale"])

        assert config["drift"] == my_drift
        assert config["drift_params"] == ["scale"]


class TestBackwardCompatibility:
    """Test that Simulator produces similar results to legacy simulator()."""

    def test_equivalence_with_legacy_simulator(self):
        """Test that Simulator gives similar results to simulator()."""
        theta = {"v": 0.5, "a": 1.0, "z": 0.5, "t": 0.3}

        # Legacy simulator
        legacy_results = simulator(
            theta=theta, model="ddm", n_samples=1000, random_state=42
        )

        # New Simulator class
        sim = Simulator("ddm")
        new_results = sim.simulate(theta, n_samples=1000, random_state=42)

        # Results should be identical (same random seed)
        np.testing.assert_array_equal(legacy_results["rts"], new_results["rts"])
        np.testing.assert_array_equal(legacy_results["choices"], new_results["choices"])


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_no_model_or_simulator(self):
        """Test that initialization fails without model or simulator."""
        with pytest.raises(ValueError, match="Must provide either"):
            Simulator()

    def test_simulator_function_missing_model_params(self):
        """Test that simulator function must accept model params."""

        def bad_sim(x, y, max_t=20, **kwargs):  # Missing declared params
            return {"rts": np.array([0.5]), "choices": np.array([1]), "metadata": {}}

        with pytest.raises(ValueError, match="missing required model parameter"):
            Simulator(
                simulator_function=bad_sim,
                params=["v", "a"],  # Declared params
                nchoices=2,
            )

    def test_config_property_returns_copy(self):
        """Test that config property returns a copy, not reference."""
        sim = Simulator("ddm")
        config1 = sim.config
        config2 = sim.config

        # Modifying one shouldn't affect the other
        config1["test_field"] = "test_value"
        assert "test_field" not in config2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
