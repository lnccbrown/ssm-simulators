# Custom Parameter Transforms

## Overview

Custom parameter transforms allow you to register and apply model-specific transformations to sampled parameters. This feature is useful when you need to:

- Apply non-linear transformations (e.g., exponential, logarithmic)
- Enforce custom constraints between parameters
- Implement model-specific parameter adjustments
- Scale or clip parameters after sampling

The transform system is fully integrated with the existing parameter sampling infrastructure and works seamlessly with the `DataGenerator` class.

## Quick Start

```python
import numpy as np
from ssms import register_transform_function

# Register a custom transform
def exponential_v(theta: dict) -> dict:
    if 'v' in theta:
        theta['v'] = np.exp(theta['v'])
    return theta

register_transform_function("exp_v", exponential_v)

# Use it in a model config
model_config = {
    "name": "my_model",
    "params": ["v", "a", "z", "t"],
    "param_bounds": [(-3, 3), (0.3, 2.5), (0.1, 0.9), (0, 2)],
    "parameter_transforms": [
        {"type": "exp_v"}  # Your custom transform!
    ]
}
```

## Two Registration Patterns

### Pattern 1: Function-Based (Simple)

Best for simple transformations that don't need configuration.

```python
from ssms import register_transform_function

def my_transform(theta: dict) -> dict:
    """Transform theta and return modified dict."""
    # Your transformation logic here
    return theta

register_transform_function("my_transform", my_transform)
```

**Use in config:**
```python
"parameter_transforms": [
    {"type": "my_transform"}
]
```

### Pattern 2: Class-Based (Configurable)

Best for transformations that need parameters at instantiation time.

```python
from ssms import register_transform_class

class MyTransform:
    def __init__(self, param_name: str, factor: float = 1.0):
        self.param_name = param_name
        self.factor = factor

    def apply(self, theta: dict) -> dict:
        """Transform theta and return modified dict."""
        if self.param_name in theta:
            theta[self.param_name] = theta[self.param_name] * self.factor
        return theta

register_transform_class("my_transform", MyTransform)
```

**Use in config:**
```python
"parameter_transforms": [
    {"type": "my_transform", "param_name": "v", "factor": 2.0}
]
```

## Common Transform Examples

### Exponential Transform

Useful for ensuring positive parameters or working in log-space.

```python
def exp_transform(param_name):
    def transform(theta):
        if param_name in theta:
            theta[param_name] = np.exp(theta[param_name])
        return theta
    return transform

register_transform_function("exp_v", exp_transform("v"))
```

### Log Transform

For log-normal distributions or compression of parameter ranges.

```python
class LogTransform:
    def __init__(self, param_name: str, epsilon: float = 1e-10):
        self.param_name = param_name
        self.epsilon = epsilon

    def apply(self, theta: dict) -> dict:
        if self.param_name in theta:
            theta[self.param_name] = np.log(
                theta[self.param_name] + self.epsilon
            )
        return theta

register_transform_class("log_transform", LogTransform)
```

### Clipping Transform

Ensure parameters stay within bounds after other transformations.

```python
class ClipTransform:
    def __init__(self, param_name: str, min_val: float, max_val: float):
        self.param_name = param_name
        self.min_val = min_val
        self.max_val = max_val

    def apply(self, theta: dict) -> dict:
        if self.param_name in theta:
            theta[self.param_name] = np.clip(
                theta[self.param_name],
                self.min_val,
                self.max_val
            )
        return theta

register_transform_class("clip", ClipTransform)
```

### Ratio Constraint

Enforce relationships between parameters.

```python
class RatioConstraint:
    def __init__(self, numerator: str, denominator: str,
                 min_ratio: float, max_ratio: float):
        self.numerator = numerator
        self.denominator = denominator
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def apply(self, theta: dict) -> dict:
        if self.numerator in theta and self.denominator in theta:
            ratio = theta[self.numerator] / theta[self.denominator]
            ratio_clipped = np.clip(ratio, self.min_ratio, self.max_ratio)
            theta[self.numerator] = ratio_clipped * theta[self.denominator]
        return theta

register_transform_class("ratio_constraint", RatioConstraint)
```

## Working with Arrays

Transform functions must handle both scalar and array inputs, as parameters are typically sampled in batches.

```python
def safe_transform(theta: dict) -> dict:
    """Example of array-safe transformation."""
    if 'v' in theta:
        # NumPy operations work on both scalars and arrays
        theta['v'] = np.exp(theta['v'])  # ✅ Works for both
    return theta
```

**Important:** Always use NumPy operations rather than Python conditionals on arrays:

```python
# ❌ WRONG - fails with arrays
if theta['v'] < 0:
    theta['v'] = -theta['v']

# ✅ CORRECT - works with arrays
theta['v'] = np.abs(theta['v'])

# ✅ CORRECT - conditional with arrays
theta['v'] = np.where(theta['v'] < 0, -theta['v'], theta['v'])
```

## Integration with DataGenerator

Custom transforms work seamlessly with the existing `DataGenerator` workflow:

```python
from ssms import register_transform_function
from ssms.dataset_generators.lan_mlp import DataGenerator

# Step 1: Register transforms BEFORE creating DataGenerator
register_transform_function("my_transform", my_transform_fn)

# Step 2: Define model config with custom transforms
model_config = {
    "name": "my_model",
    "params": ["v", "a", "z", "t"],
    "param_bounds": [...],
    "parameter_transforms": [
        {"type": "my_transform"},  # Custom
        {"type": "swap", "param_a": "a", "param_b": "z"},  # Built-in
    ]
}

# Step 3: Create generator and generate data
generator = DataGenerator(model_config=model_config, ...)
data = generator.generate_data_training_uniform(n_training_samples=1000)

# Transforms are applied automatically during parameter sampling!
```

## Mixing Custom and Built-in Transforms

You can freely mix custom transforms with built-in transforms (`swap`, `normalize`):

```python
"parameter_transforms": [
    {"type": "exp_v"},           # Custom
    {"type": "clip", "param_name": "v", "min_val": 0.1, "max_val": 5.0},  # Custom
    {"type": "swap", "param_a": "a", "param_b": "z"},  # Built-in
    {"type": "normalize", "param_names": ["v1", "v2", "v3"]},  # Built-in
]
```

Transforms are applied in the order they appear in the list.

## Import Options

The registration functions are available at multiple import levels for convenience:

```python
# Option 1: Top-level (recommended)
from ssms import register_transform_function, register_transform_class

# Option 2: Module level
from ssms.dataset_generators.parameter_samplers import (
    register_transform_function,
    register_transform_class,
)

# Option 3: Submodule level
from ssms.dataset_generators.parameter_samplers.transforms import (
    register_transform_function,
    register_transform_class,
)
```

## Registry Management

You can inspect and manage the global transform registry:

```python
from ssms import get_registry

registry = get_registry()

# List all registered transforms (including built-ins)
print(registry.list_transforms())
# Output: ['swap', 'normalize', 'exp_v', 'clip', ...]

# Check if a transform is registered
if registry.is_registered("my_transform"):
    print("Transform is registered!")

# Get a transform by name
transform_cls = registry.get("my_transform")
```

## Best Practices

### 1. Register Early

Register all custom transforms before creating `DataGenerator` instances:

```python
# ✅ GOOD
register_transform_function("my_transform", my_fn)
generator = DataGenerator(model_config=config, ...)

# ❌ BAD - transform registered after generator creation
generator = DataGenerator(model_config=config, ...)
register_transform_function("my_transform", my_fn)  # Too late!
```

### 2. Use Descriptive Names

Choose clear, descriptive names for your transforms:

```python
# ✅ GOOD
register_transform_function("exp_drift_rate", ...)
register_transform_function("log_threshold", ...)

# ❌ BAD
register_transform_function("t1", ...)
register_transform_function("my_func", ...)
```

### 3. Handle Missing Parameters Gracefully

Check if parameters exist before transforming:

```python
def robust_transform(theta: dict) -> dict:
    if 'v' in theta:  # ✅ Check first
        theta['v'] = np.exp(theta['v'])
    return theta
```

### 4. Document Your Transforms

Add docstrings explaining what the transform does and why:

```python
def speed_accuracy_tradeoff(theta: dict) -> dict:
    """Enforce inverse relationship between threshold and drift.

    Higher thresholds lead to more conservative drift rates,
    reflecting a speed-accuracy tradeoff.
    """
    if 'a' in theta and 'v' in theta:
        theta['v'] = theta['v'] * (2.0 - theta['a'])
    return theta
```

### 5. Avoid Side Effects

Transforms should modify and return theta, not modify external state:

```python
# ✅ GOOD - pure function
def good_transform(theta: dict) -> dict:
    theta['v'] = np.exp(theta['v'])
    return theta

# ❌ BAD - side effects
counter = 0
def bad_transform(theta: dict) -> dict:
    global counter
    counter += 1  # Side effect!
    return theta
```

## Advanced: Transform Factories

For creating multiple similar transforms programmatically:

```python
def create_power_transform(param_name: str, power: float):
    """Factory that creates power transforms."""
    def transform(theta: dict) -> dict:
        if param_name in theta:
            theta[param_name] = np.power(theta[param_name], power)
        return theta
    return transform

# Register multiple power transforms
register_transform_function("square_v", create_power_transform("v", 2.0))
register_transform_function("cube_a", create_power_transform("a", 3.0))
register_transform_function("sqrt_z", create_power_transform("z", 0.5))
```

## Troubleshooting

### Error: "Transform 'X' already registered"

A transform with that name already exists. Use a different name or check for duplicate registration.

### Error: "Unknown transform type: 'X'"

The transform hasn't been registered. Ensure you call `register_transform_*()` before creating the `DataGenerator`.

### Error: "The truth value of an array with more than one element is ambiguous"

Your transform uses Python conditionals on NumPy arrays. Use NumPy operations instead:

```python
# ❌ WRONG
if theta['v'] < 0:
    theta['v'] = 0

# ✅ CORRECT
theta['v'] = np.maximum(theta['v'], 0)
```

## Complete Example

See [`examples/custom_transforms_example.py`](../examples/custom_transforms_example.py) for a comprehensive, runnable example demonstrating all features.

## Related Documentation

- [Parameter Sampler Refactor Summary](../PARAMETER_SAMPLER_REFACTOR_SUMMARY.md) - Architecture overview
- [Simulator Class Guide](simulator_class_guide.md) - Using simulators with transformed parameters
- [API Documentation](api/dataset_generators.md) - Full API reference

## Summary

Custom parameter transforms provide a flexible, powerful way to extend the parameter sampling system:

- **Two patterns**: Function-based (simple) and class-based (configurable)
- **Easy registration**: One function call before creating `DataGenerator`
- **Seamless integration**: Works with existing model configs and built-in transforms
- **Array-safe**: Handle batch sampling automatically
- **Extensible**: Register as many custom transforms as needed

For questions or issues, please open an issue on GitHub or consult the API documentation.
