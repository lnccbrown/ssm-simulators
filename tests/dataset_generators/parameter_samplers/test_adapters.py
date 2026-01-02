"""Tests for transform adapters."""

import numpy as np

from ssms.dataset_generators.parameter_samplers.constraints.adapters import (
    FunctionConstraintAdapter,
)


class TestFunctionConstraintAdapter:
    """Test the FunctionConstraintAdapter class."""

    def test_adapter_wraps_function(self):
        """Test that adapter correctly wraps a function."""

        def my_func(theta: dict) -> dict:
            theta["modified"] = True
            return theta

        adapter = FunctionConstraintAdapter(my_func, "my_func")
        result = adapter.apply({"v": 1.0})

        assert result["v"] == 1.0
        assert result["modified"] is True

    def test_adapter_preserves_function_logic(self):
        """Test that adapter preserves the wrapped function's logic."""

        def double_v(theta: dict) -> dict:
            if "v" in theta:
                theta["v"] = theta["v"] * 2
            return theta

        adapter = FunctionConstraintAdapter(double_v, "double_v")
        result = adapter.apply({"v": np.array([1.0, 2.0, 3.0])})

        np.testing.assert_array_equal(result["v"], np.array([2.0, 4.0, 6.0]))

    def test_adapter_with_scalar(self):
        """Test adapter with scalar values."""

        def add_one(theta: dict) -> dict:
            if "a" in theta:
                theta["a"] = theta["a"] + 1
            return theta

        adapter = FunctionConstraintAdapter(add_one, "add_one")
        result = adapter.apply({"a": 5.0})

        assert result["a"] == 6.0

    def test_adapter_with_arrays(self):
        """Test adapter with array values."""

        def square(theta: dict) -> dict:
            if "v" in theta:
                theta["v"] = theta["v"] ** 2
            return theta

        adapter = FunctionConstraintAdapter(square, "square")
        result = adapter.apply({"v": np.array([2.0, 3.0, 4.0])})

        np.testing.assert_array_equal(result["v"], np.array([4.0, 9.0, 16.0]))

    def test_adapter_with_multiple_parameters(self):
        """Test adapter with multiple parameters."""

        def swap_ab(theta: dict) -> dict:
            if "a" in theta and "b" in theta:
                theta["a"], theta["b"] = theta["b"], theta["a"]
            return theta

        adapter = FunctionConstraintAdapter(swap_ab, "swap")
        result = adapter.apply({"a": 1.0, "b": 2.0})

        assert result["a"] == 2.0
        assert result["b"] == 1.0

    def test_adapter_repr(self):
        """Test that adapter has a useful string representation."""

        def my_func(theta):
            return theta

        adapter = FunctionConstraintAdapter(my_func, "test_name")
        assert "test_name" in repr(adapter)
        assert "FunctionConstraintAdapter" in repr(adapter)

    def test_adapter_default_name(self):
        """Test adapter with default name."""

        def my_func(theta):
            return theta

        adapter = FunctionConstraintAdapter(my_func)
        assert "custom" in repr(adapter)

    def test_adapter_handles_missing_parameters(self):
        """Test that adapter handles missing parameters gracefully."""

        def transform_v(theta: dict) -> dict:
            if "v" in theta:
                theta["v"] = theta["v"] * 2
            return theta

        adapter = FunctionConstraintAdapter(transform_v, "transform_v")

        # Should not raise error when 'v' is missing
        result = adapter.apply({"a": 1.0, "z": 0.5})
        assert "a" in result
        assert "v" not in result

    def test_adapter_with_complex_logic(self):
        """Test adapter with more complex transformation logic."""

        def complex_transform(theta: dict) -> dict:
            # Normalize drift rates to sum to 1
            drift_params = ["v1", "v2", "v3"]
            if all(p in theta for p in drift_params):
                total = sum(theta[p] for p in drift_params)
                if total > 0:
                    for p in drift_params:
                        theta[p] = theta[p] / total
            return theta

        adapter = FunctionConstraintAdapter(complex_transform, "normalize")
        result = adapter.apply(
            {"v1": np.array([1.0]), "v2": np.array([2.0]), "v3": np.array([3.0])}
        )

        # Check normalization
        total = result["v1"][0] + result["v2"][0] + result["v3"][0]
        assert np.isclose(total, 1.0)

    def test_adapter_preserves_theta_dict_structure(self):
        """Test that adapter preserves other parameters in theta."""

        def modify_one_param(theta: dict) -> dict:
            if "v" in theta:
                theta["v"] = theta["v"] * 2
            return theta

        adapter = FunctionConstraintAdapter(modify_one_param, "modify")
        result = adapter.apply(
            {
                "v": np.array([1.0]),
                "a": np.array([2.0]),
                "z": np.array([0.5]),
                "t": np.array([0.3]),
            }
        )

        # v should be modified
        np.testing.assert_array_equal(result["v"], np.array([2.0]))

        # Other params should be unchanged
        np.testing.assert_array_equal(result["a"], np.array([2.0]))
        np.testing.assert_array_equal(result["z"], np.array([0.5]))
        np.testing.assert_array_equal(result["t"], np.array([0.3]))
