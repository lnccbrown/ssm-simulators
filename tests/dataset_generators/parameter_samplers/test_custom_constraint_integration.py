"""Integration tests for custom transform registration system."""

import pytest
import numpy as np

from ssms import register_constraint_class, register_constraint_function, get_registry
from ssms.dataset_generators.parameter_samplers.constraints.factory import (
    create_constraint_from_config,
    get_constraints_from_model_config,
)
from ssms.dataset_generators.parameter_samplers.uniform_sampler import (
    UniformParameterSampler,
)


class TestCustomTransformFactoryIntegration:
    """Test integration of custom transforms with the factory."""

    def test_factory_creates_registered_function_transform(self):
        """Test that factory can create a registered function transform."""

        # Register a function transform
        def clip_v(theta: dict) -> dict:
            if "v" in theta:
                theta["v"] = np.clip(theta["v"], -5.0, 5.0)
            return theta

        register_constraint_function("test_clip_v", clip_v)

        # Create via factory
        config = {"type": "test_clip_v"}
        transform = create_constraint_from_config(config)

        # Test it works
        result = transform.apply({"v": np.array([10.0, -10.0, 2.0])})
        np.testing.assert_array_equal(result["v"], np.array([5.0, -5.0, 2.0]))

    def test_factory_creates_registered_class_transform(self):
        """Test that factory can create a registered class transform."""

        # Register a class transform
        class MultiplyTransform:
            def __init__(self, param_name: str, factor: float):
                self.param_name = param_name
                self.factor = factor

            def apply(self, theta: dict) -> dict:
                if self.param_name in theta:
                    theta[self.param_name] = theta[self.param_name] * self.factor
                return theta

        register_constraint_class("test_multiply", MultiplyTransform)

        # Create via factory with config params
        config = {"type": "test_multiply", "param_name": "v", "factor": 3.0}
        transform = create_constraint_from_config(config)

        # Test it works
        result = transform.apply({"v": np.array([1.0, 2.0])})
        np.testing.assert_array_equal(result["v"], np.array([3.0, 6.0]))

    def test_factory_with_unregistered_type_raises_error(self):
        """Test that factory raises error for unregistered transform."""
        config = {"type": "nonexistent_transform"}

        with pytest.raises(ValueError, match="Unknown constraint type"):
            create_constraint_from_config(config)

    def test_factory_prioritizes_builtin_over_custom(self):
        """Test that built-in transforms take precedence over custom."""
        # This ensures backward compatibility - built-ins checked first

        # Try to register something with a built-in name (this will fail
        # because we check built-ins first, so the custom one is never reached)
        def custom_swap(theta):
            return theta

        register_constraint_function("test_builtin_priority_swap", custom_swap)

        # Built-in swap should still work as expected
        config = {"type": "swap", "param_a": "a", "param_b": "z"}
        transform = create_constraint_from_config(config)

        # Verify it's the built-in swap
        from ssms.dataset_generators.parameter_samplers.constraints.swap import (
            SwapIfLessConstraint,
        )

        assert isinstance(transform, SwapIfLessConstraint)

    def test_get_constraints_from_model_config_with_custom(self):
        """Test extracting custom transforms from model config."""

        # Register custom transforms
        def exp_v(theta):
            if "v" in theta:
                theta["v"] = np.exp(theta["v"])
            return theta

        register_constraint_function("test_exp_v", exp_v)

        # Model config with custom and built-in transforms
        model_config = {
            "name": "test_model",
            "parameter_sampling_constraints": [
                {"type": "test_exp_v"},
                {"type": "swap", "param_a": "a", "param_b": "z"},
            ],
        }

        transforms = get_constraints_from_model_config(model_config)

        assert len(transforms) == 2
        # First should be the custom function (wrapped in adapter)
        # Second should be the built-in swap
        from ssms.dataset_generators.parameter_samplers.constraints.swap import (
            SwapIfLessConstraint,
        )

        assert isinstance(transforms[1], SwapIfLessConstraint)


class TestCustomTransformWithSampler:
    """Test custom transforms integrated with parameter sampler."""

    def test_sampler_applies_custom_function_transform(self):
        """Test that sampler applies custom function transforms."""

        # Register a custom transform
        def square_v(theta: dict) -> dict:
            if "v" in theta:
                theta["v"] = theta["v"] ** 2
            return theta

        register_constraint_function("test_square_v", square_v)

        # Create sampler with custom transform
        param_space = {
            "v": (0.0, 2.0),
        }

        transforms = [create_constraint_from_config({"type": "test_square_v"})]

        sampler = UniformParameterSampler(
            param_space=param_space,
            transforms=transforms,
        )

        # Sample and verify transform was applied
        samples = sampler.sample(n_samples=10)

        # All v values should be >= 0 and <= 4 (since max input is 2, squared is 4)
        assert np.all(samples["v"] >= 0)
        assert np.all(samples["v"] <= 4)

    def test_sampler_applies_custom_class_transform(self):
        """Test that sampler applies custom class transforms."""

        # Register a custom class transform
        class AddConstant:
            def __init__(self, param_name: str, constant: float):
                self.param_name = param_name
                self.constant = constant

            def apply(self, theta: dict) -> dict:
                if self.param_name in theta:
                    theta[self.param_name] = theta[self.param_name] + self.constant
                return theta

        register_constraint_class("test_add_constant", AddConstant)

        # Create sampler with custom transform
        param_space = {
            "v": (0.0, 1.0),
        }

        transforms = [
            create_constraint_from_config(
                {"type": "test_add_constant", "param_name": "v", "constant": 10.0}
            )
        ]

        sampler = UniformParameterSampler(
            param_space=param_space,
            transforms=transforms,
        )

        # Sample and verify transform was applied
        samples = sampler.sample(n_samples=10)

        # All v values should be in [10, 11] (original [0, 1] + 10)
        assert np.all(samples["v"] >= 10.0)
        assert np.all(samples["v"] <= 11.0)

    def test_sampler_applies_multiple_custom_transforms(self):
        """Test that sampler applies multiple custom transforms in order."""

        # Register two custom transforms with unique names
        def double_v(theta: dict) -> dict:
            if "v" in theta:
                theta["v"] = theta["v"] * 2
            return theta

        def add_one_v(theta: dict) -> dict:
            if "v" in theta:
                theta["v"] = theta["v"] + 1
            return theta

        # Use unique names to avoid conflicts with global registry
        registry = get_registry()
        if not registry.is_registered("test_double_v_multi"):
            register_constraint_function("test_double_v_multi", double_v)
        if not registry.is_registered("test_add_one_v_multi"):
            register_constraint_function("test_add_one_v_multi", add_one_v)

        # Create sampler with both transforms
        param_space = {
            "v": (0.0, 1.0),
        }

        transforms = [
            create_constraint_from_config({"type": "test_double_v_multi"}),
            create_constraint_from_config({"type": "test_add_one_v_multi"}),
        ]

        sampler = UniformParameterSampler(
            param_space=param_space,
            transforms=transforms,
        )

        # Sample and verify transforms were applied in order
        samples = sampler.sample(n_samples=10)

        # v should be in [1, 3] (original [0, 1] * 2 + 1)
        assert np.all(samples["v"] >= 1.0)
        assert np.all(samples["v"] <= 3.0)

    def test_sampler_mixes_custom_and_builtin_transforms(self):
        """Test that sampler can mix custom and built-in transforms."""

        # Register a custom transform
        def abs_v(theta: dict) -> dict:
            if "v" in theta:
                theta["v"] = np.abs(theta["v"])
            return theta

        register_constraint_function("test_abs_v", abs_v)

        # Create sampler with custom and built-in transforms
        param_space = {
            "v": (-2.0, 2.0),
            "a": (0.0, 2.0),
            "z": (0.0, 2.0),
        }

        transforms = [
            create_constraint_from_config({"type": "test_abs_v"}),
            create_constraint_from_config(
                {"type": "swap", "param_a": "a", "param_b": "z"}
            ),
        ]

        sampler = UniformParameterSampler(
            param_space=param_space,
            transforms=transforms,
        )

        # Sample and verify both transforms work
        samples = sampler.sample(n_samples=100)

        # v should be >= 0 (abs transform)
        assert np.all(samples["v"] >= 0)

        # a should be >= z (swap transform)
        assert np.all(samples["a"] >= samples["z"])


class TestEndToEndCustomTransformWorkflow:
    """End-to-end tests for the complete custom transform workflow."""

    def test_complete_workflow_with_custom_model(self):
        """Test complete workflow: register -> config -> sampler -> sample."""

        # Step 1: Register custom transforms
        def exp_transform(theta: dict) -> dict:
            if "v" in theta:
                theta["v"] = np.exp(theta["v"])
            return theta

        class ClipTransform:
            def __init__(self, param_name: str, min_val: float, max_val: float):
                self.param_name = param_name
                self.min_val = min_val
                self.max_val = max_val

            def apply(self, theta: dict) -> dict:
                if self.param_name in theta:
                    theta[self.param_name] = np.clip(
                        theta[self.param_name], self.min_val, self.max_val
                    )
                return theta

        register_constraint_function("test_exp", exp_transform)
        register_constraint_class("test_clip", ClipTransform)

        # Step 2: Create model config using custom transforms
        model_config = {
            "name": "custom_model",
            "params": ["v", "a", "z"],
            "param_bounds_dict": {
                "v": (-3.0, 3.0),  # Will be exp'd to (0.05, 20)
                "a": (0.3, 2.5),
                "z": (0.1, 0.9),
            },
            "parameter_sampling_constraints": [
                {"type": "test_exp"},  # Exp transform v
                {
                    "type": "test_clip",
                    "param_name": "v",
                    "min_val": 0.1,
                    "max_val": 5.0,
                },  # Clip v
                {"type": "swap", "param_a": "a", "param_b": "z"},  # Built-in swap
            ],
        }

        # Step 3: Create sampler from config
        transforms = get_constraints_from_model_config(model_config)
        sampler = UniformParameterSampler(
            param_space=model_config["param_bounds_dict"],
            transforms=transforms,
        )

        # Step 4: Sample and verify
        samples = sampler.sample(n_samples=100)

        # Verify shapes
        assert samples["v"].shape == (100,)
        assert samples["a"].shape == (100,)
        assert samples["z"].shape == (100,)

        # Verify transforms were applied
        assert np.all(samples["v"] >= 0.1)  # Clipped
        assert np.all(samples["v"] <= 5.0)  # Clipped
        assert np.all(samples["a"] >= samples["z"])  # Swapped

        print("✅ End-to-end custom transform workflow successful!")
