"""Tests for the transform registry."""

import pytest
import numpy as np

from ssms.dataset_generators.parameter_samplers.constraints.registry import (
    ConstraintRegistry,
    register_constraint_class,
    register_constraint_function,
    get_registry,
)


class DummyTransform:
    """Simple transform class for testing."""

    def __init__(self, param_name: str):
        self.param_name = param_name

    def apply(self, theta: dict) -> dict:
        """Identity transform for testing."""
        return theta


def dummy_function(theta: dict) -> dict:
    """Simple transform function for testing."""
    return theta


class TestConstraintRegistry:
    """Test the ConstraintRegistry class."""

    def test_register_class(self):
        """Test registering a transform class."""
        registry = ConstraintRegistry()
        registry.register_class("dummy", DummyTransform)

        assert registry.is_registered("dummy")
        assert registry.get("dummy") == DummyTransform

    def test_register_function(self):
        """Test registering a transform function."""
        registry = ConstraintRegistry()
        registry.register_function("dummy_fn", dummy_function)

        assert registry.is_registered("dummy_fn")
        assert registry.get("dummy_fn") == dummy_function

    def test_duplicate_registration_raises_error(self):
        """Test that registering the same name twice raises ValueError."""
        registry = ConstraintRegistry()
        registry.register_class("duplicate", DummyTransform)

        with pytest.raises(ValueError, match="already registered"):
            registry.register_class("duplicate", DummyTransform)

    def test_duplicate_across_types_raises_error(self):
        """Test that registering same name as function and class raises error."""
        registry = ConstraintRegistry()
        registry.register_class("name", DummyTransform)

        with pytest.raises(ValueError, match="already registered"):
            registry.register_function("name", dummy_function)

    def test_get_unregistered_transform_raises_error(self):
        """Test that getting an unregistered transform raises KeyError."""
        registry = ConstraintRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.get("nonexistent")

    def test_is_registered_returns_false_for_unregistered(self):
        """Test that is_registered returns False for unregistered names."""
        registry = ConstraintRegistry()
        assert not registry.is_registered("nonexistent")

    def test_list_constraints(self):
        """Test listing all registered transforms."""
        registry = ConstraintRegistry()
        registry.register_class("class1", DummyTransform)
        registry.register_function("func1", dummy_function)

        transforms = registry.list_constraints()
        assert "class1" in transforms
        assert "func1" in transforms
        assert len(transforms) == 2

    def test_empty_registry_lists_no_transforms(self):
        """Test that a new registry has no transforms."""
        registry = ConstraintRegistry()
        assert registry.list_constraints() == []


class TestGlobalRegistryFunctions:
    """Test the global registry convenience functions."""

    def test_register_constraint_class(self):
        """Test the global register_constraint_class function."""
        # Note: This modifies the global registry, so we need to be careful
        # Register with a unique name to avoid conflicts

        class TestTransform:
            def apply(self, theta):
                return theta

        register_constraint_class("test_global_class", TestTransform)

        registry = get_registry()
        assert registry.is_registered("test_global_class")

    def test_register_constraint_function(self):
        """Test the global register_constraint_function function."""

        def test_fn(theta):
            return theta

        register_constraint_function("test_global_fn", test_fn)

        registry = get_registry()
        assert registry.is_registered("test_global_fn")

    def test_get_registry_returns_same_instance(self):
        """Test that get_registry always returns the same instance."""
        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2


class TestRegistryIntegration:
    """Integration tests for the registry with real transforms."""

    def test_register_and_use_function_transform(self):
        """Test registering and using a real function transform."""

        def double_v(theta: dict) -> dict:
            if "v" in theta:
                theta["v"] = theta["v"] * 2
            return theta

        register_constraint_function("test_double_v", double_v)

        # Get and use the transform
        registry = get_registry()
        transform_fn = registry.get("test_double_v")

        result = transform_fn({"v": np.array([1.0, 2.0, 3.0])})
        np.testing.assert_array_equal(result["v"], np.array([2.0, 4.0, 6.0]))

    def test_register_and_use_class_transform(self):
        """Test registering and using a real class transform."""

        class ScaleTransform:
            def __init__(self, param_name: str, scale: float):
                self.param_name = param_name
                self.scale = scale

            def apply(self, theta: dict) -> dict:
                if self.param_name in theta:
                    theta[self.param_name] = theta[self.param_name] * self.scale
                return theta

        register_constraint_class("test_scale", ScaleTransform)

        # Get and instantiate the transform
        registry = get_registry()
        TransformClass = registry.get("test_scale")
        transform = TransformClass(param_name="v", scale=3.0)

        result = transform.apply({"v": np.array([1.0, 2.0])})
        np.testing.assert_array_equal(result["v"], np.array([3.0, 6.0]))

    def test_multiple_registrations_coexist(self):
        """Test that multiple registrations don't interfere."""

        def fn1(theta):
            theta["a"] = theta.get("a", 0) + 1
            return theta

        def fn2(theta):
            theta["b"] = theta.get("b", 0) + 2
            return theta

        register_constraint_function("test_fn1", fn1)
        register_constraint_function("test_fn2", fn2)

        registry = get_registry()

        # Both should be registered
        assert registry.is_registered("test_fn1")
        assert registry.is_registered("test_fn2")

        # Both should work independently
        result1 = registry.get("test_fn1")({"a": 5})
        assert result1["a"] == 6

        result2 = registry.get("test_fn2")({"b": 10})
        assert result2["b"] == 12
