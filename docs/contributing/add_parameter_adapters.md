# Custom Parameter Adaptations

## Overview

Parameter adaptations transform user-provided parameters into simulator-ready format. This guide shows you how to create custom adaptations for your models.

---

## Quick Start

### Using Built-in Adaptations

```python
from ssms import Simulator
from ssms.basic_simulators.parameter_adapters import SetDefaultValue

sim = Simulator(
    "ddm",
    parameter_adaptations=[
        SetDefaultValue("custom_param", 0.5)
    ]
)

result = sim.simulate(
    theta={'v': 1.0, 'a': 1.5, 'z': 0.5, 't': 0.3},
    n_samples=1000
)
# theta now includes custom_param=0.5
```

### Creating a Custom Adaptation

```python
from ssms.basic_simulators.parameter_adapters import ParameterAdaptation
import numpy as np

class ScaleParameter(ParameterAdaptation):
    """Scale a parameter by a constant factor."""

    def __init__(self, param_name: str, scale: float):
        self.param_name = param_name
        self.scale = scale

    def apply(self, theta: dict, model_config: dict, n_trials: int) -> dict:
        if self.param_name in theta:
            theta[self.param_name] = theta[self.param_name] * self.scale
        return theta

# Use it
sim = Simulator("ddm", parameter_adaptations=[ScaleParameter("v", 2.0)])
```

---

## The ParameterAdaptation Interface

All adaptations inherit from `ParameterAdaptation` and implement one method:

```python
from abc import ABC, abstractmethod

class ParameterAdaptation(ABC):
    @abstractmethod
    def apply(self, theta: dict, model_config: dict, n_trials: int) -> dict:
        """Apply adaptation to theta parameters.

        Parameters
        ----------
        theta : dict
            Parameter dictionary to transform
        model_config : dict
            Model configuration (name, params, nchoices, etc.)
        n_trials : int
            Number of trials

        Returns
        -------
        dict
            Modified theta dictionary
        """
        pass
```

**Key Points:**
- Modify `theta` in-place and return it
- Handle missing parameters gracefully (check with `if param_name in theta`)
- Use `dtype=np.float32` for new arrays
- Scalar values in `theta` are converted to arrays during preprocessing (e.g., `0.5` → `np.array([0.5])`)
  - Extract scalars with `float(np.asarray(theta[key]).flat[0])` if needed

---

## Built-in Adaptations

Common adaptations you can use immediately:

```python
from ssms.basic_simulators.parameter_adapters import (
    SetDefaultValue,      # Set parameter if not present
    ExpandDimension,      # (n,) → (n, 1)
    ColumnStackParameters,# Stack v0, v1, v2 → v
    RenameParameter,      # Rename param (e.g., A → z)
    DeleteParameters,     # Remove parameters
    LambdaAdaptation,     # Quick custom function
)
```

---

## Complete Example: Time-Varying Drift

Here's a real-world example that adds time-varying drift rates:

```python
from ssms.basic_simulators.parameter_adapters import ParameterAdaptation
import numpy as np

class TimeVaryingDrift(ParameterAdaptation):
    """Create time-varying drift from start and end values.

    Takes v_start and v_end parameters and creates a linearly
    interpolated drift rate array for use with time-varying models.

    Parameters
    ----------
    n_steps : int
        Number of time steps for interpolation
    """

    def __init__(self, n_steps: int = 100):
        self.n_steps = n_steps

    def apply(self, theta: dict, model_config: dict, n_trials: int) -> dict:
        # Only apply if both start and end are provided
        if "v_start" in theta and "v_end" in theta:
            # Extract scalar values (preprocessing converts scalars to arrays)
            v_start = float(np.asarray(theta["v_start"]).flat[0])
            v_end = float(np.asarray(theta["v_end"]).flat[0])

            # Create interpolated array
            theta["v"] = np.linspace(
                v_start,
                v_end,
                self.n_steps,
                dtype=np.float32
            )

            # Clean up temporary params
            del theta["v_start"]
            del theta["v_end"]

        return theta

# Usage
sim = Simulator(
    "ddm",
    parameter_adaptations=[TimeVaryingDrift(n_steps=100)]
)

result = sim.simulate(
    theta={'v_start': 0.5, 'v_end': 2.0, 'a': 1.5, 'z': 0.5, 't': 0.3},
    n_samples=1000
)
```

---

## Testing Your Adaptation

```python
import numpy as np

def test_time_varying_drift():
    """Test the TimeVaryingDrift adaptation."""
    adaptation = TimeVaryingDrift(n_steps=10)

    # Simulate what preprocessing does: scalars → arrays
    theta = {
        'v_start': np.array([0.0]),
        'v_end': np.array([1.0]),
        'a': np.array([1.5]),
    }

    result = adaptation.apply(theta, model_config={}, n_trials=1)

    # Check that v was created
    assert 'v' in result
    assert len(result['v']) == 10

    # Check interpolation correctness
    assert result['v'][0] == 0.0
    assert result['v'][-1] == 1.0

    # Check cleanup
    assert 'v_start' not in result
    assert 'v_end' not in result
```

---

## Best Practices

1. **Single responsibility**: Each adaptation should do one thing
2. **Check before modifying**: Use `if param_name in theta` to avoid KeyErrors
3. **Document clearly**: Explain what parameters are added/removed/modified
4. **Use type hints**: Makes your code more maintainable
5. **Provide `__repr__`**: Default implementation is usually sufficient
6. **Handle edge cases**: What if a parameter is missing? Already exists?

---

## Resources

- [Adding Models Tutorial](add_models.md): How to contribute new models
- [API Reference](../../api/): Complete API documentation
- Built-in adaptations: `ssms/basic_simulators/parameter_adapters/common.py`
