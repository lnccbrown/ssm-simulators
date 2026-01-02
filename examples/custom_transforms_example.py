"""Examples of custom transform registration.

This module demonstrates how to register and use custom parameter transforms
with the ssm-simulators package. Custom transforms allow you to apply
model-specific modifications to sampled parameters.
"""

import numpy as np
from ssms import register_transform_class, register_transform_function
from ssms.dataset_generators.lan_mlp import TrainingDataGenerator


# ============================================================================
# Example 1: Simple Function Transform
# ============================================================================


def exponential_transform_v(theta: dict) -> dict:
    """Apply exponential transformation to drift rate.

    This can be useful when you want to ensure drift rates are positive
    or when working in log-space during sampling.
    """
    if "v" in theta:
        theta["v"] = np.exp(theta["v"])
    return theta


# Register it with a descriptive name
register_transform_function(
    "exp_v", exponential_transform_v, description="Exponential transform for drift rate"
)


# ============================================================================
# Example 2: Class-Based Transform with Parameters
# ============================================================================


class LogTransform:
    """Apply log transformation to a parameter.

    Class-based transforms are useful when you need to configure
    the transform with parameters at instantiation time.
    """

    def __init__(self, param_name: str, epsilon: float = 1e-10):
        """Initialize the transform.

        Args:
            param_name: Name of the parameter to transform
            epsilon: Small value to avoid log(0)
        """
        self.param_name = param_name
        self.epsilon = epsilon

    def apply(self, theta: dict) -> dict:
        """Apply log transformation to the specified parameter."""
        if self.param_name in theta:
            theta[self.param_name] = np.log(theta[self.param_name] + self.epsilon)
        return theta


# Register the class
register_transform_class("log_transform", LogTransform)


# ============================================================================
# Example 3: Clipping Transform
# ============================================================================


class ClipTransform:
    """Clip parameter values to a specified range.

    Useful for ensuring parameters stay within valid bounds
    after other transformations.
    """

    def __init__(self, param_name: str, min_val: float, max_val: float):
        self.param_name = param_name
        self.min_val = min_val
        self.max_val = max_val

    def apply(self, theta: dict) -> dict:
        """Clip parameter to [min_val, max_val]."""
        if self.param_name in theta:
            theta[self.param_name] = np.clip(
                theta[self.param_name], self.min_val, self.max_val
            )
        return theta


register_transform_class("clip", ClipTransform)


# ============================================================================
# Example 4: Scaling Transform
# ============================================================================


def create_scale_transform(param_name: str, scale: float):
    """Factory function to create a scaling transform.

    This pattern is useful when you want to create multiple similar
    transforms programmatically.
    """

    def scale_transform(theta: dict) -> dict:
        if param_name in theta:
            theta[param_name] = theta[param_name] * scale
        return theta

    return scale_transform


# Register specific scaling transforms
register_transform_function("double_v", create_scale_transform("v", 2.0))
register_transform_function("half_a", create_scale_transform("a", 0.5))


# ============================================================================
# Example 5: Multi-Parameter Transform
# ============================================================================


class RatioConstraintTransform:
    """Ensure a ratio constraint between two parameters.

    For example, ensure that z/a stays within [0.3, 0.7].
    """

    def __init__(
        self, numerator: str, denominator: str, min_ratio: float, max_ratio: float
    ):
        self.numerator = numerator
        self.denominator = denominator
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def apply(self, theta: dict) -> dict:
        """Adjust numerator to maintain ratio constraint."""
        if self.numerator in theta and self.denominator in theta:
            ratio = theta[self.numerator] / theta[self.denominator]

            # Clip ratio to valid range
            ratio_clipped = np.clip(ratio, self.min_ratio, self.max_ratio)

            # Adjust numerator to achieve clipped ratio
            theta[self.numerator] = ratio_clipped * theta[self.denominator]

        return theta


register_transform_class("ratio_constraint", RatioConstraintTransform)


# ============================================================================
# Example 6: Using Custom Transforms in a Model Config
# ============================================================================


def example_custom_model():
    """Demonstrate using custom transforms in a model configuration."""

    # Define a custom model configuration
    my_custom_model_config = {
        "name": "custom_ddm_exp_drift",
        "params": ["v", "a", "z", "t"],
        "param_bounds": [
            (-3.0, 3.0),  # v (will be exponentiated)
            (0.3, 2.5),  # a
            (0.1, 0.9),  # z
            (0.0, 2.0),  # t
        ],
        "boundary": lambda t: 1.0,
        "n_params": 4,
        "default_params": [0.0, 1.0, 0.5, 0.5],
        "hddm_include": ["z"],
        "nchoices": 2,
        # Use custom transforms!
        "parameter_transforms": [
            # Apply exponential to drift rate (custom transform)
            {"type": "exp_v"},
            # Ensure z < a (built-in transform)
            {"type": "swap", "param_a": "a", "param_b": "z"},
            # Keep drift rate in reasonable range after exp (custom transform)
            {"type": "clip", "param_name": "v", "min_val": 0.01, "max_val": 10.0},
        ],
    }

    return my_custom_model_config


# ============================================================================
# Example 7: Complete Workflow
# ============================================================================


def main():
    """Complete example of registering transforms and generating data."""

    print("Custom Transform Registration Example")
    print("=" * 60)

    # Step 1: Register all custom transforms (already done above)
    from ssms import get_registry

    registry = get_registry()
    print(f"\n1. Registered transforms: {registry.list_transforms()}")

    # Step 2: Create a model config using custom transforms
    model_config = example_custom_model()
    print(f"\n2. Model: {model_config['name']}")
    print(f"   Parameters: {model_config['params']}")
    print(f"   Transforms: {[t['type'] for t in model_config['parameter_transforms']]}")

    # Step 3: Create a TrainingDataGenerator
    # Note: We're using a small sample size for demonstration
    generator = TrainingDataGenerator(
        model_config=model_config,
        n_samples=100,  # Small for demo
        n_trials=128,
    )

    print("\n3. TrainingDataGenerator created successfully")

    # Step 4: Generate training data
    print("\n4. Generating training data...")
    training_data = generator.generate_data_training_uniform(n_training_samples=100)

    print(f"   Generated data shape: {training_data['data'].shape}")
    print(f"   Theta shape: {training_data['data'].shape[:2]}")

    # Step 5: Inspect transformed parameters
    theta_samples = training_data["data"][:, :, 0]  # First trial, all params
    print("\n5. Parameter statistics after transforms:")
    for i, param in enumerate(model_config["params"]):
        values = theta_samples[:, i]
        print(
            f"   {param}: min={values.min():.3f}, "
            f"max={values.max():.3f}, mean={values.mean():.3f}"
        )

    print("\n✅ Custom transform workflow complete!")
    print("\nKey takeaways:")
    print("  - Register transforms before creating TrainingDataGenerator")
    print("  - Use transform names in model config 'parameter_transforms'")
    print("  - Transforms are applied automatically during sampling")
    print("  - Mix custom and built-in transforms freely")


# ============================================================================
# Example 8: Dynamic Transform Registration
# ============================================================================


def register_custom_transforms_for_study():
    """Example of registering study-specific transforms on-the-fly."""

    # Register a study-specific transform
    def enforce_speed_accuracy_tradeoff(theta: dict) -> dict:
        """Ensure inverse relationship between threshold and drift."""
        if "a" in theta and "v" in theta:
            # Higher threshold -> lower average drift
            theta["v"] = theta["v"] * (2.0 - theta["a"])
        return theta

    register_transform_function(
        "speed_accuracy_tradeoff", enforce_speed_accuracy_tradeoff
    )

    print("Registered study-specific transform: speed_accuracy_tradeoff")


if __name__ == "__main__":
    # Run the main example
    main()

    # Optionally, demonstrate dynamic registration
    print("\n" + "=" * 60)
    print("Dynamic Registration Example")
    print("=" * 60)
    register_custom_transforms_for_study()
