"""Tests for parameter samplers."""

import numpy as np
import pytest
from ssms.dataset_generators.parameter_samplers import UniformParameterSampler


class TestUniformParameterSampler:
    """Test suite for UniformParameterSampler."""

    def test_basic_sampling(self):
        """Test basic uniform sampling without dependencies."""
        param_space = {
            "v": (-1.0, 1.0),
            "a": (0.5, 2.0),
        }
        sampler = UniformParameterSampler(param_space)
        samples = sampler.sample(n_samples=100)

        assert "v" in samples
        assert "a" in samples
        assert samples["v"].shape == (100,)
        assert samples["a"].shape == (100,)
        assert np.all(samples["v"] >= -1.0) and np.all(samples["v"] <= 1.0)
        assert np.all(samples["a"] >= 0.5) and np.all(samples["a"] <= 2.0)

    def test_single_sample(self):
        """Test sampling a single parameter set."""
        param_space = {"v": (-1.0, 1.0), "a": (0.5, 2.0)}
        sampler = UniformParameterSampler(param_space)
        samples = sampler.sample(n_samples=1)

        assert samples["v"].shape == (1,)
        assert samples["a"].shape == (1,)
        assert -1.0 <= samples["v"][0] <= 1.0
        assert 0.5 <= samples["a"][0] <= 2.0

    def test_parameter_dependencies(self):
        """Test sampling with parameter dependencies."""
        param_space = {
            "t": (0.25, 2.25),
            "st": (1e-3, "t"),  # st depends on t
        }
        sampler = UniformParameterSampler(param_space)
        samples = sampler.sample(n_samples=100)

        # st should be less than t for all samples
        assert np.all(samples["st"] < samples["t"])
        assert np.all(samples["st"] >= 1e-3)

    def test_multiple_dependencies(self):
        """Test sampling with multiple dependent parameters."""
        param_space = {
            "v": (-3.0, 3.0),
            "a": (0.3, 2.5),
            "t": (0.25, 2.25),
            "st": (1e-3, "t"),
            "sv": (1e-3, "a"),  # sv depends on a (always positive)
        }
        sampler = UniformParameterSampler(param_space)
        samples = sampler.sample(n_samples=50)

        assert np.all(samples["st"] < samples["t"])
        assert np.all(samples["sv"] < samples["a"])

    def test_sampling_order_consistency(self):
        """Test that dependencies are sampled in correct order."""
        # This should not raise any errors about missing dependencies
        param_space = {
            "c": (0.0, "a"),  # c depends on a
            "b": (0.0, "a"),  # b depends on a
            "a": (1.0, 5.0),  # a is independent
        }
        sampler = UniformParameterSampler(param_space)
        samples = sampler.sample(n_samples=10)

        assert np.all(samples["c"] <= samples["a"])
        assert np.all(samples["b"] <= samples["a"])

    def test_get_param_space(self):
        """Test get_param_space method."""
        param_space = {"v": (-1.0, 1.0), "a": (0.5, 2.0)}
        sampler = UniformParameterSampler(param_space)

        assert sampler.get_param_space() == param_space

    def test_with_constraints(self):
        """Test sampler with constraints applied."""
        from ssms.transforms import SwapIfLessConstraint

        param_space = {"a": (0.1, 1.1), "z": (0.0, 0.5)}
        constraints = [SwapIfLessConstraint("a", "z")]
        sampler = UniformParameterSampler(param_space, constraints)
        samples = sampler.sample(n_samples=100)

        # After swap transform, all a > z
        assert np.all(samples["a"] > samples["z"])

    def test_circular_dependency_detection(self):
        """Test that circular dependencies are detected."""
        param_space = {
            "a": (0.0, "b"),
            "b": (0.0, "c"),
            "c": (0.0, "a"),  # Circular!
        }

        with pytest.raises(ValueError, match="Circular dependency"):
            _ = UniformParameterSampler(param_space)

    def test_missing_dependency(self):
        """Test that missing dependencies are detected."""
        param_space = {
            "a": (0.0, 5.0),
            "b": (0.0, "nonexistent"),  # nonexistent parameter
        }

        with pytest.raises(ValueError, match="not defined in param_space"):
            _ = UniformParameterSampler(param_space)

    def test_dtype_consistency(self):
        """Test that sampled values have consistent dtype (float32)."""
        param_space = {"v": (-1.0, 1.0), "a": (0.5, 2.0)}
        sampler = UniformParameterSampler(param_space)
        samples = sampler.sample(n_samples=10)

        assert samples["v"].dtype == np.float32
        assert samples["a"].dtype == np.float32
