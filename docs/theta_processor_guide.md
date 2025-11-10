# Theta Processor Guide

## Overview

The **theta processor** system in `ssm-simulators` handles parameter transformations before passing them to the simulator. This guide explains how to use the modular theta processor system and customize parameter transformations for your models.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [What is Theta Processing?](#what-is-theta-processing)
3. [The Modular Theta Processor](#the-modular-theta-processor)
4. [Using Custom Transformations](#using-custom-transformations)
5. [Built-in Transformations](#built-in-transformations)
6. [Advanced Usage](#advanced-usage)
7. [API Reference](#api-reference)

---

## Quick Start

### Default Behavior (Recommended)

```python
from ssms import Simulator

# ModularThetaProcessor is used by default
sim = Simulator("lba2")

result = sim.simulate(
    theta={'v0': 0.5, 'v1': 0.6, 'A': 0.5, 'b': 1.0},
    n_samples=1000,
    random_state=42
)
```

The theta parameters are automatically transformed:
- `v0` and `v1` → stacked into `v` array
- `A` → renamed to `z` and expanded to 2D
- `b` → renamed to `a` and expanded to 2D
- `nact` → set to 2
- `t` → set to zeros array

### Adding Custom Transformations

```python
from ssms import Simulator
from ssms.basic_simulators.theta_transforms import SetDefaultValue

sim = Simulator(
    "ddm",
    theta_transforms=[
        SetDefaultValue("custom_param", 42)
    ]
)

result = sim.simulate(
    theta={'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3},
    n_samples=1000
)
# Now theta also contains custom_param=42
```

---

## What is Theta Processing?

### The Problem

Different models require different parameter formats:
- **DDM** expects: `{'v': array, 'a': array, 'z': array, 't': array}`
- **LBA** expects: `{'v': 2D array, 'a': 2D array, 'z': 2D array, 't': array, 'nact': int}`
- **Race models** expect: `{'v': 2D array, 'z': 2D array, 'a': 2D array, 't': array}`

Users typically provide parameters in a natural format:
```python
theta = {'v0': 0.5, 'v1': 0.6, 'A': 0.5, 'b': 1.0}  # User format
```

But the simulator needs:
```python
theta = {
    'v': np.array([[0.5, 0.6]]),      # Stacked drift rates
    'z': np.array([[0.5]]),           # Start point (renamed from A)
    'a': np.array([[1.0]]),           # Threshold (renamed from b)
    'nact': 2,                        # Number of accumulators
    't': np.array([0.0])              # Non-decision time
}
```

### The Solution: Theta Processors

**Theta processors** automatically transform user-provided parameters into simulator-ready format. The transformation is:
1. **Model-specific**: Each model has its own transformation rules
2. **Automatic**: Happens transparently during `simulate()`
3. **Customizable**: You can add your own transformations

---

## The Modular Theta Processor

### Architecture

The `ModularThetaProcessor` uses a **pipeline of transformations**:

```
User Theta → [Transform 1] → [Transform 2] → ... → [Transform N] → Simulator Theta
```

Each transformation is a small, single-purpose operation:
- Set default values
- Rename parameters
- Expand dimensions
- Stack arrays
- Apply mappings

### How It Works

```python
from ssms import Simulator

sim = Simulator("lba2")

# Behind the scenes:
# 1. User provides: {'v0': 0.5, 'v1': 0.6, 'A': 0.5, 'b': 1.0}
# 2. ModularThetaProcessor applies transformations:
#    - LambdaTransformation: Set nact=2
#    - ColumnStackParameters: Stack v0, v1 → v
#    - RenameParameter: A → z (with expand_dims)
#    - RenameParameter: b → a (with expand_dims)
#    - DeleteParameters: Remove A, b
#    - LambdaTransformation: Set t=zeros
# 3. Simulator receives: {'v': [[0.5, 0.6]], 'z': [[0.5]], 'a': [[1.0]], 'nact': 2, 't': [0.0]}
```

### Inspecting Transformations

```python
sim = Simulator("lba2")

# See what transformations are applied
print(sim.theta_processor.registry.describe("lba2"))
```

Output:
```
Model: lba2 (exact match)
Transformations (6):
  1. LambdaTransformation(name='set_nact_2')
  2. ColumnStackParameters(source_params=['v0', 'v1'], target_param='v', delete_sources=False)
  3. RenameParameter(old_name='A', new_name='z', transform_func=<lambda>)
  4. RenameParameter(old_name='b', new_name='a', transform_func=<lambda>)
  5. DeleteParameters(params=['A', 'b'])
  6. LambdaTransformation(name='set_zero_t')
```

---

## Using Custom Transformations

### Adding Single Transformation

```python
from ssms import Simulator
from ssms.basic_simulators.theta_transforms import SetDefaultValue

# Add default value for a parameter
sim = Simulator(
    "ddm",
    theta_transforms=[
        SetDefaultValue("sigma", 1.0)
    ]
)
```

### Adding Multiple Transformations

```python
from ssms import Simulator
from ssms.basic_simulators.theta_transforms import (
    SetDefaultValue,
    LambdaTransformation,
)

sim = Simulator(
    "ddm",
    theta_transforms=[
        SetDefaultValue("param1", 100),
        SetDefaultValue("param2", 200),
        LambdaTransformation(
            lambda theta, cfg, n: theta.update({"computed": theta["param1"] * 2}) or theta,
            name="compute_value"
        )
    ]
)
```

### Transformation Order

Custom transformations are applied **AFTER** model-default transformations:

```python
# For LBA2:
# 1. Model defaults: set nact, stack v, rename A/b, etc.
# 2. Your custom transforms: applied in order
```

This ensures model-specific logic runs first, then your customizations.

---

## Built-in Transformations

### 1. `SetDefaultValue`
Set parameter to default value if not present.

```python
from ssms.basic_simulators.theta_transforms import SetDefaultValue

# If 'bias' not in theta, set it to 0.5
transform = SetDefaultValue("bias", 0.5)
```

### 2. `ExpandDimension`
Add a dimension to parameters (1D → 2D).

```python
from ssms.basic_simulators.theta_transforms import ExpandDimension

# Expand 'a' and 'z' from shape (n,) to (n, 1)
transform = ExpandDimension(["a", "z"])
```

### 3. `ColumnStackParameters`
Stack multiple parameters into a single multi-column array.

```python
from ssms.basic_simulators.theta_transforms import ColumnStackParameters

# Stack v0, v1, v2 into v with shape (n, 3)
transform = ColumnStackParameters(
    source_params=["v0", "v1", "v2"],
    target_param="v",
    delete_sources=False  # Keep v0, v1, v2
)
```

### 4. `RenameParameter`
Rename a parameter, optionally applying a transformation.

```python
from ssms.basic_simulators.theta_transforms import RenameParameter
import numpy as np

# Rename 'A' to 'z' and expand dimensions
transform = RenameParameter(
    old_name="A",
    new_name="z",
    transform_func=lambda x: np.expand_dims(x, axis=1),
    delete_source=True
)
```

### 5. `DeleteParameters`
Remove parameters from theta.

```python
from ssms.basic_simulators.theta_transforms import DeleteParameters

# Remove temporary parameters
transform = DeleteParameters(["temp1", "temp2"])
```

### 6. `LambdaTransformation`
Apply arbitrary function to theta.

```python
from ssms.basic_simulators.theta_transforms import LambdaTransformation
import numpy as np

# Custom logic
transform = LambdaTransformation(
    func=lambda theta, cfg, n: theta.update({
        "custom": np.ones(n) * 42
    }) or theta,
    name="add_custom_param"
)
```

### 7. `ApplyMapping`
Apply a mapping function from model config.

```python
from ssms.basic_simulators.theta_transforms import ApplyMapping

# Map 'sz' parameter to 'z_dist' using model's mapping function
transform = ApplyMapping(
    source_param="sz",
    target_param="z_dist",
    mapping_key="z_dist"
)
```

---

## Advanced Usage

### Creating Custom Processor with Registry

```python
from ssms import Simulator
from ssms.basic_simulators.modular_theta_processor import ModularThetaProcessor
from ssms.basic_simulators.theta_transforms import (
    ThetaProcessorRegistry,
    SetDefaultValue,
    ExpandDimension,
)

# Create custom registry
registry = ThetaProcessorRegistry()

# Register custom model
registry.register_model("my_custom_model", [
    SetDefaultValue("alpha", 1.0),
    SetDefaultValue("beta", 2.0),
    ExpandDimension(["a", "z"]),
])

# Create processor with custom registry
processor = ModularThetaProcessor(registry=registry)

# Use with Simulator
sim = Simulator(
    model="my_custom_model",
    theta_processor=processor
)
```

### Using SimpleThetaProcessor (Legacy)

For backward compatibility, you can use the legacy processor:

```python
from ssms import Simulator
from ssms.basic_simulators.theta_processor import SimpleThetaProcessor

# Use legacy processor explicitly
sim = Simulator(
    "lba2",
    theta_processor=SimpleThetaProcessor()
)

# Note: theta_transforms parameter is ignored with SimpleThetaProcessor
```

### Family Registration (Pattern Matching)

Register transformations for model families:

```python
registry = ThetaProcessorRegistry()

# All race models with 2 choices
registry.register_family(
    family_name="race_2_choice",
    matcher=lambda model_name: model_name.startswith("race_") and model_name.endswith("_2"),
    transformations=[
        ColumnStackParameters(["v0", "v1"], "v"),
        ExpandDimension(["a", "z"]),
    ]
)
```

---

## API Reference

### Classes

#### `ModularThetaProcessor`
Main class for modular theta processing.

**Constructor:**
```python
ModularThetaProcessor(registry: ThetaProcessorRegistry | None = None)
```
- `registry`: Custom registry. If None, uses default with all 109 models.

**Methods:**
```python
process_theta(theta: dict, model_config: dict, n_trials: int) -> dict
```
Apply transformations to theta.

#### `ThetaProcessorRegistry`
Registry mapping models to transformation pipelines.

**Methods:**
```python
register_model(model_name: str, transformations: list[ThetaTransformation])
register_family(family_name: str, matcher: Callable, transformations: list)
get_processor(model_name: str) -> list[ThetaTransformation]
describe(model_name: str) -> str
```

### Base Classes

#### `ThetaTransformation` (ABC)
Base class for all transformations.

**Abstract Method:**
```python
apply(theta: dict, model_config: dict, n_trials: int) -> dict
```

### Transformation Classes

All transformation classes inherit from `ThetaTransformation` and implement `apply()`.

See [Built-in Transformations](#built-in-transformations) for usage examples.

---

## Best Practices

### 1. **Use Default Processor**
Unless you have specific needs, the default `ModularThetaProcessor` works well:
```python
sim = Simulator("ddm")  # Recommended
```

### 2. **Add, Don't Replace**
Use `theta_transforms` to add behavior, not replace model logic:
```python
sim = Simulator("ddm", theta_transforms=[...])  # Good
```

Avoid creating entirely custom processors unless necessary.

### 3. **Keep Transformations Simple**
Each transformation should do one thing:
```python
# Good: Single responsibility
SetDefaultValue("param", 42)

# Avoid: Complex multi-step lambda
LambdaTransformation(lambda theta, cfg, n: complicated_logic(...))
```

### 4. **Name Your Lambdas**
Always provide names for `LambdaTransformation`:
```python
LambdaTransformation(func, name="descriptive_name")
```

### 5. **Inspect Before Using**
Check what transformations a model uses:
```python
sim = Simulator("lba2")
print(sim.theta_processor.registry.describe("lba2"))
```

---

## Troubleshooting

### Problem: Custom transforms not applied

**Cause:** Using `SimpleThetaProcessor` instead of `ModularThetaProcessor`

**Solution:**
```python
# Don't do this:
sim = Simulator("ddm", theta_processor=SimpleThetaProcessor(), theta_transforms=[...])

# Do this:
sim = Simulator("ddm", theta_transforms=[...])  # Uses ModularThetaProcessor by default
```

### Problem: Parameters in wrong format

**Cause:** Transformations not registered for your model

**Solution:** Check if model has transformations:
```python
sim = Simulator("your_model")
print(sim.theta_processor.registry.describe("your_model"))
```

If no transformations, register them:
```python
from ssms.basic_simulators.theta_transforms import ThetaProcessorRegistry
registry = ThetaProcessorRegistry()
registry.register_model("your_model", [...])
```

### Problem: Can't see what's happening

**Solution:** Use introspection:
```python
# See processor type
print(type(sim.theta_processor))

# See transformations
print(sim.theta_processor.registry.describe(model_name))

# See processed theta (debug)
theta_before = {'v': 0.5, ...}
theta_after = sim.theta_processor.process_theta(theta_before, model_config, n_trials=1)
print(theta_after)
```

---

## Examples

### Example 1: Adding Noise Parameter

```python
from ssms import Simulator
from ssms.basic_simulators.theta_transforms import SetDefaultValue
import numpy as np

sim = Simulator(
    "ddm",
    theta_transforms=[
        SetDefaultValue("noise_scale", 0.1)
    ]
)

result = sim.simulate(
    theta={'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3},
    n_samples=1000
)
```

### Example 2: Parameter Constraints

```python
from ssms import Simulator
from ssms.basic_simulators.theta_transforms import LambdaTransformation
import numpy as np

# Ensure 'a' is always positive
ensure_positive = LambdaTransformation(
    func=lambda theta, cfg, n: theta.update({
        "a": np.abs(theta["a"])
    }) or theta,
    name="ensure_positive_a"
)

sim = Simulator("ddm", theta_transforms=[ensure_positive])
```

### Example 3: Computed Parameters

```python
from ssms import Simulator
from ssms.basic_simulators.theta_transforms import LambdaTransformation

# Compute relative start point from absolute
compute_relative_z = LambdaTransformation(
    func=lambda theta, cfg, n: theta.update({
        "z": theta["z_absolute"] / theta["a"]
    }) or theta,
    name="compute_relative_z"
)

sim = Simulator("ddm", theta_transforms=[compute_relative_z])

result = sim.simulate(
    theta={'v': 0.5, 'a': 1.0, 'z_absolute': 0.5, 't': 0.3},
    n_samples=1000
)
```

---

## See Also

- [Custom Theta Transforms Guide](custom_theta_transforms_guide.md) - Creating your own transformations
- [Migration Guide](migration_guide_theta_processor.md) - Migrating from SimpleThetaProcessor
- [Simulator Class Guide](simulator_class_guide.md) - Full Simulator documentation

---

## Summary

The modular theta processor system provides:
- ✅ **Automatic parameter transformations** for all 109 models
- ✅ **Customizable transformations** via `theta_transforms` parameter
- ✅ **Introspectable** via `describe()` and properties
- ✅ **Backward compatible** with `SimpleThetaProcessor`
- ✅ **Well-tested** with 100% model coverage

**Recommended usage:**
```python
from ssms import Simulator

# Use defaults (works for all models)
sim = Simulator("model_name")

# Add custom transforms when needed
sim = Simulator("model_name", theta_transforms=[...])
```

