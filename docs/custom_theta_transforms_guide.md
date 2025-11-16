# Creating Custom Theta Transformations

## Overview

This guide shows you how to create your own theta transformations for custom parameter processing in `ssm-simulators`.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [The ThetaTransformation Interface](#the-thetatransformation-interface)
3. [Creating Simple Transformations](#creating-simple-transformations)
4. [Creating Complex Transformations](#creating-complex-transformations)
5. [Best Practices](#best-practices)
6. [Real-World Examples](#real-world-examples)
7. [Testing Your Transformations](#testing-your-transformations)

---

## Quick Start

### Using LambdaTransformation

The easiest way to create a custom transformation:

```python
from ssms import Simulator
from ssms.basic_simulators.theta_transforms import LambdaTransformation
import numpy as np

# Create transformation
my_transform = LambdaTransformation(
    func=lambda theta, cfg, n: theta.update({"my_param": 42}) or theta,
    name="set_my_param"
)

# Use it
sim = Simulator("ddm", theta_transforms=[my_transform])
```

### Creating a Class

For reusable transformations:

```python
from ssms.basic_simulators.theta_transforms import ThetaTransformation
import numpy as np

class ScaleParameter(ThetaTransformation):
    """Scale a parameter by a constant factor."""
    
    def __init__(self, param_name: str, scale_factor: float):
        self.param_name = param_name
        self.scale_factor = scale_factor
    
    def apply(self, theta: dict, model_config: dict, n_trials: int) -> dict:
        if self.param_name in theta:
            theta[self.param_name] = theta[self.param_name] * self.scale_factor
        return theta

# Use it
transform = ScaleParameter("v", 2.0)
sim = Simulator("ddm", theta_transforms=[transform])
```

---

## The ThetaTransformation Interface

### Base Class

All transformations inherit from `ThetaTransformation`:

```python
from abc import ABC, abstractmethod

class ThetaTransformation(ABC):
    """Base class for theta transformations."""
    
    @abstractmethod
    def apply(self, theta: dict, model_config: dict, n_trials: int) -> dict:
        """Apply the transformation to theta.
        
        Parameters
        ----------
        theta : dict
            Parameter dictionary to transform
        model_config : dict
            Model configuration (contains name, params, etc.)
        n_trials : int
            Number of trials
            
        Returns
        -------
        dict
            Transformed theta dictionary
        """
        pass
```

### The `apply()` Method

Your transformation must implement `apply()` with this signature:

```python
def apply(self, theta: dict, model_config: dict, n_trials: int) -> dict:
    # Your transformation logic here
    return theta
```

**Parameters:**
- `theta`: The parameter dictionary to transform (modify in-place or return new dict)
- `model_config`: Model configuration with keys like `name`, `params`, `nchoices`, etc.
- `n_trials`: Number of trials (useful for array sizing)

**Returns:**
- Modified `theta` dictionary

---

## Creating Simple Transformations

### Pattern 1: Set Value

```python
class SetConstant(ThetaTransformation):
    """Set parameter to constant value."""
    
    def __init__(self, param_name: str, value: float):
        self.param_name = param_name
        self.value = value
    
    def apply(self, theta, model_config, n_trials):
        theta[self.param_name] = self.value
        return theta
```

### Pattern 2: Modify Existing

```python
class AddOffset(ThetaTransformation):
    """Add offset to parameter."""
    
    def __init__(self, param_name: str, offset: float):
        self.param_name = param_name
        self.offset = offset
    
    def apply(self, theta, model_config, n_trials):
        if self.param_name in theta:
            theta[self.param_name] = theta[self.param_name] + self.offset
        return theta
```

### Pattern 3: Conditional Logic

```python
class ClampParameter(ThetaTransformation):
    """Clamp parameter to range [min_val, max_val]."""
    
    def __init__(self, param_name: str, min_val: float, max_val: float):
        self.param_name = param_name
        self.min_val = min_val
        self.max_val = max_val
    
    def apply(self, theta, model_config, n_trials):
        if self.param_name in theta:
            import numpy as np
            theta[self.param_name] = np.clip(
                theta[self.param_name],
                self.min_val,
                self.max_val
            )
        return theta
```

---

## Creating Complex Transformations

### Pattern 4: Multi-Parameter Operations

```python
class NormalizeParameters(ThetaTransformation):
    """Normalize multiple parameters to sum to 1."""
    
    def __init__(self, param_names: list[str]):
        self.param_names = param_names
    
    def apply(self, theta, model_config, n_trials):
        import numpy as np
        
        # Collect values
        values = [theta.get(name, 0) for name in self.param_names]
        total = sum(values)
        
        if total > 0:
            # Normalize
            for name, value in zip(self.param_names, values):
                theta[name] = value / total
        
        return theta
```

### Pattern 5: Computed Parameters

```python
class ComputeDriftRate(ThetaTransformation):
    """Compute drift rate from base + modulator."""
    
    def __init__(self, base_param: str = "v_base", modulator_param: str = "v_mod"):
        self.base_param = base_param
        self.modulator_param = modulator_param
    
    def apply(self, theta, model_config, n_trials):
        if self.base_param in theta and self.modulator_param in theta:
            theta["v"] = theta[self.base_param] + theta[self.modulator_param]
        return theta
```

### Pattern 6: Using Model Config

```python
class SetFromConfig(ThetaTransformation):
    """Set parameter from model config."""
    
    def __init__(self, param_name: str, config_key: str):
        self.param_name = param_name
        self.config_key = config_key
    
    def apply(self, theta, model_config, n_trials):
        import numpy as np
        
        if self.config_key in model_config:
            value = model_config[self.config_key]
            theta[self.param_name] = np.tile(
                np.array([value], dtype=np.float32),
                n_trials
            )
        return theta
```

---

## Best Practices

### 1. **Keep It Simple**
Each transformation should do one thing:
```python
# Good: Single responsibility
class SetDefaultValue(ThetaTransformation):
    def apply(self, theta, model_config, n_trials):
        if self.param_name not in theta:
            theta[self.param_name] = self.default_value
        return theta

# Avoid: Multiple responsibilities
class DoEverything(ThetaTransformation):
    def apply(self, theta, model_config, n_trials):
        # Sets defaults, renames, scales, validates...
        # Too much!
```

### 2. **Handle Missing Parameters Gracefully**
```python
def apply(self, theta, model_config, n_trials):
    # Check before accessing
    if self.param_name in theta:
        theta[self.param_name] = transform(theta[self.param_name])
    else:
        # Handle gracefully or skip
        pass
    return theta
```

### 3. **Use Type Hints**
```python
from typing import Any

class MyTransform(ThetaTransformation):
    def __init__(self, param_name: str, value: float):
        self.param_name: str = param_name
        self.value: float = value
    
    def apply(
        self,
        theta: dict[str, Any],
        model_config: dict[str, Any],
        n_trials: int
    ) -> dict[str, Any]:
        # ...
        return theta
```

### 4. **Document Your Transformations**
```python
class MyTransform(ThetaTransformation):
    """Short description.
    
    Longer description explaining what it does, when to use it,
    and any important caveats.
    
    Parameters
    ----------
    param1 : type
        Description
    param2 : type
        Description
        
    Examples
    --------
    >>> transform = MyTransform("v", 2.0)
    >>> theta = {'v': np.array([1.0])}
    >>> theta = transform.apply(theta, {}, 1)
    >>> theta['v']
    array([2.0])
    """
```

### 5. **Provide `__repr__`**
Makes debugging easier:
```python
class MyTransform(ThetaTransformation):
    def __init__(self, param_name, value):
        self.param_name = param_name
        self.value = value
    
    def __repr__(self):
        return f"MyTransform(param_name='{self.param_name}', value={self.value})"
```

### 6. **Be Careful with In-Place Modifications**
```python
def apply(self, theta, model_config, n_trials):
    import numpy as np
    
    # Safe: Create new array
    theta["v"] = theta["v"] * 2
    
    # Risky: In-place modification might affect original
    theta["v"] *= 2  # This modifies the original array!
    
    return theta
```

### 7. **Handle Arrays Correctly**
```python
def apply(self, theta, model_config, n_trials):
    import numpy as np
    
    # Ensure correct dtype and shape
    theta["my_param"] = np.array(
        [self.value] * n_trials,
        dtype=np.float32
    )
    
    return theta
```

---

## Real-World Examples

### Example 1: Log-Transform Parameters

```python
class LogTransform(ThetaTransformation):
    """Apply log transform to parameter."""
    
    def __init__(self, param_name: str, base: float = np.e):
        self.param_name = param_name
        self.base = base
    
    def apply(self, theta, model_config, n_trials):
        import numpy as np
        
        if self.param_name in theta:
            theta[self.param_name] = np.log(theta[self.param_name]) / np.log(self.base)
        
        return theta

# Usage
transform = LogTransform("v", base=10)
sim = Simulator("ddm", theta_transforms=[transform])
```

### Example 2: Parameter Constraints

```python
class EnforceSumConstraint(ThetaTransformation):
    """Ensure parameters sum to target value."""
    
    def __init__(self, param_names: list[str], target_sum: float = 1.0):
        self.param_names = param_names
        self.target_sum = target_sum
    
    def apply(self, theta, model_config, n_trials):
        import numpy as np
        
        # Get values
        values = [theta.get(p, 0.0) for p in self.param_names]
        current_sum = sum(values)
        
        if current_sum > 0:
            # Scale to target
            scale = self.target_sum / current_sum
            for param_name in self.param_names:
                if param_name in theta:
                    theta[param_name] = theta[param_name] * scale
        
        return theta

# Usage: Ensure drift rates sum to 1
transform = EnforceSumConstraint(["v0", "v1", "v2"], target_sum=1.0)
```

### Example 3: Hierarchical Parameters

```python
class ExpandHierarchicalParams(ThetaTransformation):
    """Expand group-level parameter to individual-level."""
    
    def __init__(self, group_param: str, individual_params: list[str]):
        self.group_param = group_param
        self.individual_params = individual_params
    
    def apply(self, theta, model_config, n_trials):
        import numpy as np
        
        if self.group_param in theta:
            group_value = theta[self.group_param]
            
            # Copy to each individual parameter
            for param in self.individual_params:
                if param not in theta:
                    theta[param] = group_value.copy()
        
        return theta

# Usage
transform = ExpandHierarchicalParams("v_group", ["v0", "v1", "v2"])
```

### Example 4: Time-Varying Parameters

```python
class InterpolateParameter(ThetaTransformation):
    """Create time-varying parameter from start and end values."""
    
    def __init__(
        self,
        start_param: str,
        end_param: str,
        output_param: str,
        n_steps: int = 100
    ):
        self.start_param = start_param
        self.end_param = end_param
        self.output_param = output_param
        self.n_steps = n_steps
    
    def apply(self, theta, model_config, n_trials):
        import numpy as np
        
        if self.start_param in theta and self.end_param in theta:
            start_val = theta[self.start_param]
            end_val = theta[self.end_param]
            
            # Create interpolated values
            theta[self.output_param] = np.linspace(
                start_val,
                end_val,
                self.n_steps
            )
        
        return theta
```

### Example 5: Conditional Transformations

```python
class ConditionalSetValue(ThetaTransformation):
    """Set parameter based on condition."""
    
    def __init__(
        self,
        param_name: str,
        value: float,
        condition_func: Callable[[dict], bool]
    ):
        self.param_name = param_name
        self.value = value
        self.condition_func = condition_func
    
    def apply(self, theta, model_config, n_trials):
        import numpy as np
        
        if self.condition_func(theta):
            theta[self.param_name] = np.array([self.value] * n_trials)
        
        return theta

# Usage: Set bias based on other parameters
transform = ConditionalSetValue(
    param_name="z",
    value=0.5,
    condition_func=lambda theta: theta.get("v", 0) > 0  # If drift positive
)
```

---

## Testing Your Transformations

### Unit Tests

```python
import numpy as np
from your_module import MyTransform

def test_my_transform():
    # Setup
    transform = MyTransform("v", 2.0)
    theta = {"v": np.array([1.0, 1.0])}
    model_config = {"name": "test"}
    n_trials = 2
    
    # Apply
    result = transform.apply(theta, model_config, n_trials)
    
    # Assert
    np.testing.assert_array_equal(result["v"], np.array([2.0, 2.0]))
```

### Integration Tests

```python
def test_transform_with_simulator():
    from ssms import Simulator
    
    transform = MyTransform("v", 2.0)
    sim = Simulator("ddm", theta_transforms=[transform])
    
    result = sim.simulate(
        theta={'v': 1.0, 'a': 1.0, 'z': 0.5, 't': 0.3},
        n_samples=10,
        random_state=42
    )
    
    assert 'rts' in result
    assert len(result['rts']) == 10
```

### Property-Based Tests

```python
from hypothesis import given, strategies as st

@given(
    value=st.floats(min_value=0.1, max_value=10.0),
    n_trials=st.integers(min_value=1, max_value=100)
)
def test_transform_properties(value, n_trials):
    transform = MyTransform("v", value)
    theta = {"v": np.ones(n_trials)}
    
    result = transform.apply(theta, {}, n_trials)
    
    # Properties that should always hold
    assert len(result["v"]) == n_trials
    assert all(result["v"] >= 0)
```

---

## Advanced Patterns

### Composable Transformations

```python
class TransformPipeline(ThetaTransformation):
    """Apply multiple transformations in sequence."""
    
    def __init__(self, transformations: list[ThetaTransformation]):
        self.transformations = transformations
    
    def apply(self, theta, model_config, n_trials):
        for transform in self.transformations:
            theta = transform.apply(theta, model_config, n_trials)
        return theta

# Usage
pipeline = TransformPipeline([
    SetDefaultValue("v", 0.5),
    ClampParameter("v", 0.0, 2.0),
    LogTransform("v")
])
```

### Configurable Transformations

```python
class ConfigurableTransform(ThetaTransformation):
    """Transformation with configuration."""
    
    def __init__(self, config: dict):
        self.config = config
    
    def apply(self, theta, model_config, n_trials):
        if self.config.get("normalize", False):
            # Normalization logic
            pass
        
        if self.config.get("scale_factor"):
            # Scaling logic
            pass
        
        return theta
```

### Stateful Transformations

```python
class RunningAverage(ThetaTransformation):
    """Maintain running average (stateful)."""
    
    def __init__(self, param_name: str, window_size: int = 10):
        self.param_name = param_name
        self.window_size = window_size
        self.history = []
    
    def apply(self, theta, model_config, n_trials):
        import numpy as np
        
        if self.param_name in theta:
            value = theta[self.param_name]
            self.history.append(value)
            
            # Keep only recent values
            if len(self.history) > self.window_size:
                self.history = self.history[-self.window_size:]
            
            # Set to average
            theta[f"{self.param_name}_avg"] = np.mean(self.history)
        
        return theta
```

---

## Summary

Creating custom theta transformations is straightforward:

1. **Inherit from `ThetaTransformation`**
2. **Implement `apply(theta, model_config, n_trials)`**
3. **Modify and return theta**

**Key principles:**
- ✅ Keep transformations simple and focused
- ✅ Handle missing parameters gracefully
- ✅ Use type hints and documentation
- ✅ Test your transformations
- ✅ Provide good `__repr__` for debugging

**See also:**
- [Theta Processor Guide](theta_processor_guide.md) - Main documentation
- [Migration Guide](migration_guide_theta_processor.md) - Migrating from SimpleThetaProcessor

