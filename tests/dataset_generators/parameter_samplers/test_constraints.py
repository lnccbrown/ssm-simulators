"""Tests for parameter sampling constraints."""

import numpy as np
from ssms.dataset_generators.parameter_samplers.constraints import (
    SwapIfLessConstraint,
    NormalizeToSumConstraint,
)


class TestSwapIfLessConstraint:
    """Test suite for SwapIfLessConstraint."""

    def test_swap_scalars(self):
        """Test swap with scalar values."""
        transform = SwapIfLessConstraint("a", "z")

        # Case 1: a <= z, should swap
        theta = {"a": 0.5, "z": 1.0}
        result = transform.apply(theta)
        assert result["a"] == 1.0
        assert result["z"] == 0.5

        # Case 2: a > z, should not swap
        theta = {"a": 1.5, "z": 0.5}
        result = transform.apply(theta)
        assert result["a"] == 1.5
        assert result["z"] == 0.5

    def test_swap_arrays(self):
        """Test swap with array values."""
        transform = SwapIfLessConstraint("a", "z")

        theta = {
            "a": np.array([0.5, 1.5, 0.8, 2.0]),
            "z": np.array([1.0, 0.5, 1.2, 0.3]),
        }
        result = transform.apply(theta)

        # Expected: [1.0, 1.5, 1.2, 2.0] (swap when a <= z)
        # Expected: [0.5, 0.5, 0.8, 0.3] (corresponding z values)
        assert np.array_equal(result["a"], np.array([1.0, 1.5, 1.2, 2.0]))
        assert np.array_equal(result["z"], np.array([0.5, 0.5, 0.8, 0.3]))

        # Verify constraint: a > z for all elements
        assert np.all(result["a"] > result["z"])

    def test_swap_equal_values(self):
        """Test swap when values are equal."""
        transform = SwapIfLessConstraint("a", "z")

        # Scalar case
        theta = {"a": 1.0, "z": 1.0}
        result = transform.apply(theta)
        # Should swap when equal (a <= z)
        assert result["a"] == 1.0
        assert result["z"] == 1.0

        # Array case
        theta = {
            "a": np.array([1.0, 2.0, 1.5]),
            "z": np.array([1.0, 1.0, 1.5]),
        }
        result = transform.apply(theta)
        assert np.all(result["a"] >= result["z"])

    def test_swap_missing_parameters(self):
        """Test that transform handles missing parameters gracefully."""
        transform = SwapIfLessConstraint("a", "z")

        # Missing both parameters
        theta = {"v": 1.0}
        result = transform.apply(theta)
        assert result == theta  # No change

        # Missing one parameter
        theta = {"a": 1.0, "v": 2.0}
        result = transform.apply(theta)
        assert result == theta  # No change

    def test_swap_large_arrays(self):
        """Test swap with large arrays for performance."""
        transform = SwapIfLessConstraint("a", "z")

        n = 10000
        theta = {
            "a": np.random.uniform(0.1, 1.1, size=n),
            "z": np.random.uniform(0.0, 0.5, size=n),
        }
        result = transform.apply(theta)

        # All should satisfy a > z after transform
        assert np.all(result["a"] > result["z"])


class TestNormalizeToSumConstraint:
    """Test suite for NormalizeToSumConstraint."""

    def test_normalize_scalars(self):
        """Test normalization with scalar values."""
        transform = NormalizeToSumConstraint(["v1", "v2", "v3"])

        theta = {"v1": 0.2, "v2": 0.3, "v3": 0.5}
        result = transform.apply(theta)

        total = result["v1"] + result["v2"] + result["v3"]
        assert abs(total - 1.0) < 1e-6

    def test_normalize_arrays(self):
        """Test normalization with array values."""
        transform = NormalizeToSumConstraint(["vRL0", "vRL1", "vRL2"])

        theta = {
            "vRL0": np.array([0.2, 0.3, 0.5]),
            "vRL1": np.array([0.3, 0.2, 0.3]),
            "vRL2": np.array([0.5, 0.5, 0.2]),
        }
        result = transform.apply(theta)

        sums = result["vRL0"] + result["vRL1"] + result["vRL2"]
        assert np.allclose(sums, 1.0, atol=1e-6)

    def test_normalize_random_values(self):
        """Test normalization with random values that need scaling."""
        transform = NormalizeToSumConstraint(["v1", "v2", "v3"])

        np.random.seed(42)
        theta = {
            "v1": np.random.uniform(0.0, 2.0, size=100),
            "v2": np.random.uniform(0.0, 2.0, size=100),
            "v3": np.random.uniform(0.0, 2.0, size=100),
        }
        result = transform.apply(theta)

        sums = result["v1"] + result["v2"] + result["v3"]
        assert np.allclose(sums, 1.0, atol=1e-6)

    def test_normalize_near_zero(self):
        """Test normalization with very small values (edge case)."""
        transform = NormalizeToSumConstraint(["v1", "v2"])

        theta = {
            "v1": np.array([1e-10, 0.0, 1e-15]),
            "v2": np.array([1e-11, 1e-12, 0.0]),
        }
        result = transform.apply(theta)

        sums = result["v1"] + result["v2"]
        # Should still sum to 1.0 (epsilon prevents division by zero)
        assert np.allclose(sums, 1.0, atol=1e-6)

    def test_normalize_preserves_ratios(self):
        """Test that normalization preserves relative ratios."""
        transform = NormalizeToSumConstraint(["v1", "v2", "v3"])

        # Values with known ratios: 1:2:3
        theta = {"v1": 1.0, "v2": 2.0, "v3": 3.0}
        result = transform.apply(theta)

        # Ratios should be approximately preserved
        # (slight deviation due to epsilon)
        ratio_12 = result["v1"] / result["v2"]
        ratio_23 = result["v2"] / result["v3"]
        assert abs(ratio_12 - 0.5) < 0.01
        assert abs(ratio_23 - 2.0 / 3.0) < 0.01

    def test_normalize_missing_parameters(self):
        """Test that transform handles missing parameters gracefully."""
        transform = NormalizeToSumConstraint(["v1", "v2", "v3"])

        # Missing one parameter
        theta = {"v1": 0.5, "v2": 0.5}
        result = transform.apply(theta)
        assert result == theta  # No change when not all params present

    def test_normalize_two_parameters(self):
        """Test normalization with just two parameters."""
        transform = NormalizeToSumConstraint(["a", "b"])

        theta = {"a": 0.7, "b": 0.3}
        result = transform.apply(theta)

        total = result["a"] + result["b"]
        assert abs(total - 1.0) < 1e-6

    def test_normalize_many_parameters(self):
        """Test normalization with many parameters."""
        param_names = [f"v{i}" for i in range(10)]
        transform = NormalizeToSumConstraint(param_names)

        theta = {name: np.random.uniform(0, 1, size=50) for name in param_names}
        result = transform.apply(theta)

        total = sum(result[name] for name in param_names)
        assert np.allclose(total, 1.0, atol=1e-6)
