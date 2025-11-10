# Simulator Class User Guide

The `Simulator` class provides a modern, object-oriented interface for running Sequential Sampling Model (SSM) simulations. It offers enhanced flexibility and extensibility compared to the legacy `simulator()` function while maintaining full backward compatibility.

## Table of Contents

1. [Why Use the Simulator Class?](#why-use-the-simulator-class)
2. [Basic Usage](#basic-usage)
3. [Custom Boundary Functions](#custom-boundary-functions)
4. [Custom Drift Functions](#custom-drift-functions)
5. [Fully Custom Simulators](#fully-custom-simulators)
6. [Configuration Management](#configuration-management)
7. [Migration Guide](#migration-guide)
8. [API Reference](#api-reference)

## Why Use the Simulator Class?

The `Simulator` class offers several advantages over the legacy `simulator()` function:

### State Management
- **Initialize once, simulate many times**: Configure your simulator once and reuse it for multiple simulations
- **Clean configuration access**: Inspect and modify configuration through the `config` property
- **Better encapsulation**: Keep related simulation settings together

### Extensibility
- **Custom boundary functions**: Easily integrate your own boundary dynamics
- **Custom drift functions**: Define time-varying drift rates
- **Custom simulators**: Use completely custom simulation logic (Python or Cython)

### Developer Experience
- **Type hints and docstrings**: Better IDE support and documentation
- **Validation**: Configuration is validated at initialization time
- **Clear error messages**: Helpful guidance when something goes wrong

## Basic Usage

### Using Pre-defined Models

The simplest way to use the `Simulator` class is with a pre-defined model name:

```python
from ssms import Simulator

# Initialize simulator with DDM model
sim = Simulator("ddm")

# Run simulation
results = sim.simulate(
    theta={'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3},
    n_samples=1000
)

# Access results
print(results['rts'])      # Reaction times
print(results['choices'])   # Choices
print(results['metadata'])  # Simulation metadata
```

### Available Models

The simulator supports all models from the legacy `simulator()` function:

```python
# Drift Diffusion Models
sim = Simulator("ddm")
sim = Simulator("angle")
sim = Simulator("weibull")

# Race Models
sim = Simulator("race_3")
sim = Simulator("lca_3")

# LBA Models
sim = Simulator("lba2")
sim = Simulator("lba3")

# And many more...
```

### Reproducible Simulations

Use the `random_state` parameter for reproducibility:

```python
sim = Simulator("ddm")

# These will produce identical results
results1 = sim.simulate(theta, n_samples=1000, random_state=42)
results2 = sim.simulate(theta, n_samples=1000, random_state=42)
```

### Configuration Overrides

Override specific configuration parameters at initialization:

```python
sim = Simulator(
    "ddm",
    param_bounds=[[-4.0, 0.3, 0.1, 0.0], [4.0, 3.0, 0.9, 2.0]],
    default_params=[0.0, 1.5, 0.5, 0.001]
)
```

## Custom Boundary Functions

Define time-varying boundary functions to model collapsing decision bounds or other dynamic processes.

### Using Pre-defined Boundaries

```python
# Use angle boundary (linear collapse)
sim = Simulator("ddm", boundary="angle")

# Simulate with boundary parameters
results = sim.simulate(
    theta={
        'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3,
        'theta': 0.5  # Boundary angle parameter
    },
    n_samples=1000
)
```

Available boundary functions:
- `"constant"`: Fixed boundaries (default for DDM)
- `"angle"`: Linear collapse
- `"weibull_cdf"`: Weibull-based collapse
- `"generalized_logistic"`: Logistic collapse
- `"conflict_gamma"`: Conflict task boundaries

### Custom Boundary Functions

Create your own boundary dynamics:

```python
import numpy as np

def exponential_boundary(t, rate, scale):
    """Exponentially collapsing boundary."""
    return scale * np.exp(-rate * t)

sim = Simulator(
    "ddm",
    boundary=exponential_boundary,
    boundary_params=["rate", "scale"]
)

results = sim.simulate(
    theta={
        'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3,
        'rate': 0.5, 'scale': 1.0
    },
    n_samples=1000
)
```

**Requirements for custom boundary functions:**
- First parameter must be `t` (time)
- Additional parameters passed via `boundary_params`
- Must return a numpy array or scalar
- For multiplicative boundaries: return values multiply the threshold
- For additive boundaries: return values add to the threshold

### Example: Hyperbolic Boundary

```python
def hyperbolic_boundary(t, k, b):
    """Hyperbolic boundary: b(t) = a / (1 + kt)"""
    return 1.0 / (1.0 + k * t) + b

sim = Simulator(
    "ddm",
    boundary=hyperbolic_boundary,
    boundary_params=["k", "b"],
    boundary_multiplicative=True  # Multiply threshold by boundary value
)
```

## Custom Drift Functions

Model time-varying evidence accumulation rates.

### Using Pre-defined Drift Functions

```python
sim = Simulator("gamma_drift", drift="gamma_drift")

results = sim.simulate(
    theta={
        'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3,
        'shape': 2.0, 'scale': 0.5, 'c': 1.5  # Drift parameters
    },
    n_samples=1000
)
```

Available drift functions:
- `"constant"`: No time variation
- `"gamma_drift"`: Gamma-shaped drift
- `"ds_support_analytic"`: Dynamical systems drift

### Custom Drift Functions

```python
def oscillating_drift(t, frequency, amplitude):
    """Sinusoidal drift modulation."""
    return amplitude * np.sin(2 * np.pi * frequency * t)

sim = Simulator(
    "gamma_drift",  # Base model that supports drift
    drift=oscillating_drift,
    drift_params=["frequency", "amplitude"]
)

results = sim.simulate(
    theta={
        'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3,
        'frequency': 2.0, 'amplitude': 0.3
    },
    n_samples=1000
)
```

**Requirements for custom drift functions:**
- First parameter must be `t` (time array)
- Additional parameters passed via `drift_params`
- Must return numpy array matching shape of `t`
- Drift adds to the base drift rate `v`

## Fully Custom Simulators

For complete control, provide your own simulator function.

### Simple Custom Simulator

```python
def simple_accumulator(v, a, z, t, max_t=20, n_samples=1000, n_trials=1,
                       delta_t=0.001, random_state=None, **kwargs):
    """Simple drift-diffusion simulator (Python implementation)."""
    import numpy as np
    
    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)
    
    # Initialize outputs
    rts = np.zeros((n_samples, n_trials, 1))
    choices = np.zeros((n_samples, n_trials, 1))
    
    for trial in range(n_trials):
        for sample in range(n_samples):
            # Simulate accumulation
            evidence = z[trial] * a[trial]
            time = 0.0
            
            while abs(evidence) < a[trial] and time < max_t:
                evidence += v[trial] * delta_t + np.random.randn() * np.sqrt(delta_t)
                time += delta_t
            
            rts[sample, trial, 0] = time + t[trial]
            choices[sample, trial, 0] = 1 if evidence > 0 else -1
    
    return {
        'rts': rts,
        'choices': choices,
        'metadata': {
            'model': 'simple_accumulator',
            'n_samples': n_samples,
            'possible_choices': [-1, 1]
        }
    }

# Use custom simulator
sim = Simulator(
    simulator_function=simple_accumulator,
    params=["v", "a", "z", "t"],
    nchoices=2
)

results = sim.simulate(
    theta={'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3},
    n_samples=1000
)
```

### Custom Simulator with Boundaries and Drift

```python
def flexible_accumulator(v, a, z, t,
                        boundary_fun=None, boundary_params=None,
                        boundary_multiplicative=True,
                        drift_fun=None, drift_params=None,
                        max_t=20, n_samples=1000, n_trials=1,
                        delta_t=0.001, random_state=None, **kwargs):
    """Accumulator supporting dynamic boundaries and drift."""
    import numpy as np
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Precompute time array
    n_steps = int(max_t / delta_t)
    t_array = np.linspace(0, max_t, n_steps)
    
    rts = np.zeros((n_samples, n_trials, 1))
    choices = np.zeros((n_samples, n_trials, 1))
    
    for trial in range(n_trials):
        # Compute boundary and drift if provided
        if boundary_fun is not None:
            b_params = {k: boundary_params[k][trial] for k in boundary_params}
            boundary = boundary_fun(t_array, **b_params)
            if boundary_multiplicative:
                boundary = a[trial] * boundary
            else:
                boundary = a[trial] + boundary
        else:
            boundary = a[trial] * np.ones(n_steps)
        
        if drift_fun is not None:
            d_params = {k: drift_params[k][trial] for k in drift_params}
            drift = v[trial] + drift_fun(t_array, **d_params)
        else:
            drift = v[trial] * np.ones(n_steps)
        
        for sample in range(n_samples):
            evidence = z[trial] * boundary[0]
            step = 0
            
            while abs(evidence) < boundary[min(step, n_steps-1)] and step < n_steps:
                evidence += drift[step] * delta_t + np.random.randn() * np.sqrt(delta_t)
                step += 1
            
            rts[sample, trial, 0] = step * delta_t + t[trial]
            choices[sample, trial, 0] = 1 if evidence > 0 else -1
    
    return {
        'rts': rts,
        'choices': choices,
        'metadata': {
            'model': 'flexible_accumulator',
            'n_samples': n_samples,
            'possible_choices': [-1, 1]
        }
    }

# Use with custom boundary
sim = Simulator(
    simulator_function=flexible_accumulator,
    params=["v", "a", "z", "t"],
    nchoices=2,
    boundary=exponential_boundary,
    boundary_params=["rate", "scale"]
)
```

**Simulator Function Requirements:**
- Accept all model parameters as keyword arguments (numpy arrays)
- Accept standard simulation parameters: `max_t`, `n_samples`, `n_trials`, `delta_t`, `random_state`
- Optionally accept: `boundary_fun`, `boundary_params`, `drift_fun`, `drift_params`
- Return dict with keys: `'rts'`, `'choices'`, `'metadata'`

## Configuration Management

### ConfigBuilder Utility

The `ConfigBuilder` class helps create and manage configurations:

```python
from ssms.config import ConfigBuilder

# Start from existing model
config = ConfigBuilder.from_model("ddm")

# Override specific fields
config = ConfigBuilder.from_model(
    "ddm",
    param_bounds=[[-4, 0.3, 0.1, 0], [4, 3.0, 0.9, 2.0]]
)

# Add custom boundary
config = ConfigBuilder.add_boundary(
    config,
    boundary=my_boundary_func,
    boundary_params=["param1", "param2"]
)

# Use configuration
sim = Simulator(config)
```

### Build from Scratch

For completely custom configurations:

```python
config = ConfigBuilder.from_scratch(
    name="my_model",
    params=["drift", "threshold", "bias", "ndt"],
    simulator_function=my_simulator,
    nchoices=2,
    param_bounds=[[-2, 0.5, 0.1, 0], [2, 2.0, 0.9, 1.0]],
    choices=[-1, 1]
)

sim = Simulator(config)
```

### Validate Configuration

```python
is_valid, errors = ConfigBuilder.validate_config(config)

if not is_valid:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
```

### Access and Inspect Configuration

```python
sim = Simulator("ddm")

# Get configuration (returns a copy)
config = sim.config

# Inspect configuration
print(f"Model: {config['name']}")
print(f"Parameters: {config['params']}")
print(f"Bounds: {config['param_bounds']}")
```

## Migration Guide

### From `simulator()` Function to `Simulator` Class

**Legacy code:**
```python
from ssms.basic_simulators.simulator import simulator

results = simulator(
    theta={'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3},
    model="ddm",
    n_samples=1000,
    random_state=42
)
```

**New code:**
```python
from ssms import Simulator

sim = Simulator("ddm")
results = sim.simulate(
    theta={'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3},
    n_samples=1000,
    random_state=42
)
```

### Key Differences

| Legacy `simulator()` | New `Simulator` |
|---------------------|-----------------|
| Function-based | Class-based |
| Configure every call | Initialize once, reuse |
| Limited extensibility | Support custom functions |
| Less validation | Comprehensive validation |
| Single interface | Multiple initialization modes |

### Benefits of Migration

1. **Performance**: Initialize once and reuse for multiple simulations
2. **Clarity**: Separate configuration from simulation
3. **Flexibility**: Easy to add custom boundaries, drifts, or simulators
4. **Maintainability**: Better code organization

### Backward Compatibility

The legacy `simulator()` function remains fully supported. You can mix both approaches:

```python
from ssms.basic_simulators.simulator import simulator
from ssms import Simulator

# Legacy function (still works)
results1 = simulator(theta, model="ddm", n_samples=1000)

# New class
sim = Simulator("ddm")
results2 = sim.simulate(theta, n_samples=1000)
```

## API Reference

### Simulator Class

#### `Simulator.__init__()`

```python
Simulator(
    model: str | dict | None = None,
    boundary: str | Callable | None = None,
    drift: str | Callable | None = None,
    simulator_function: Callable | None = None,
    **config_overrides
)
```

**Parameters:**
- `model`: Model name, config dict, or None (for custom simulators)
- `boundary`: Boundary function (name or callable)
- `drift`: Drift function (name or callable)
- `simulator_function`: Custom simulator function
- `**config_overrides`: Additional configuration parameters

#### `Simulator.simulate()`

```python
sim.simulate(
    theta: dict | np.ndarray | pd.DataFrame,
    n_samples: int = 1000,
    delta_t: float = 0.001,
    max_t: float = 20,
    no_noise: bool = False,
    sigma_noise: float | None = None,
    smooth_unif: bool = True,
    random_state: int | None = None,
    return_option: str = "full"
) -> dict
```

**Parameters:**
- `theta`: Model parameters
- `n_samples`: Number of samples per parameter set
- `delta_t`: Time step size
- `max_t`: Maximum simulation time
- `no_noise`: Disable noise (for visualization)
- `sigma_noise`: Noise standard deviation
- `smooth_unif`: Add uniform smoothing to RTs
- `random_state`: Random seed
- `return_option`: "full" or "minimal"

**Returns:**
Dictionary with keys:
- `'rts'`: Reaction times (numpy array)
- `'choices'`: Choices (numpy array)
- `'metadata'`: Simulation metadata (dict)

#### `Simulator.config`

Property that returns a deep copy of the configuration dictionary.

```python
config = sim.config
```

### ConfigBuilder Class

#### `ConfigBuilder.from_model()`

```python
ConfigBuilder.from_model(model_name: str, **overrides) -> dict
```

Create configuration from existing model.

#### `ConfigBuilder.from_scratch()`

```python
ConfigBuilder.from_scratch(
    name: str,
    params: list[str],
    simulator_function: Callable,
    nchoices: int,
    **config
) -> dict
```

Build configuration from scratch.

#### `ConfigBuilder.minimal_config()`

```python
ConfigBuilder.minimal_config(
    params: list[str],
    simulator_function: Callable,
    nchoices: int = 2,
    name: str = "custom"
) -> dict
```

Create minimal valid configuration.

#### `ConfigBuilder.validate_config()`

```python
ConfigBuilder.validate_config(
    config: dict,
    strict: bool = False
) -> tuple[bool, list[str]]
```

Validate configuration. Returns `(is_valid, errors)`.

#### `ConfigBuilder.add_boundary()`

```python
ConfigBuilder.add_boundary(
    config: dict,
    boundary: str | Callable,
    boundary_params: list[str] | None = None,
    multiplicative: bool = True
) -> dict
```

Add boundary function to configuration.

#### `ConfigBuilder.add_drift()`

```python
ConfigBuilder.add_drift(
    config: dict,
    drift: str | Callable,
    drift_params: list[str] | None = None
) -> dict
```

Add drift function to configuration.

## Examples

### Example 1: Collapsing Bound Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from ssms import Simulator

def weibull_boundary(t, alpha, beta):
    return np.exp(-np.power(t / beta, alpha))

# Different collapse rates
alphas = [0.5, 1.0, 2.0]
results_by_alpha = {}

for alpha in alphas:
    sim = Simulator(
        "ddm",
        boundary=weibull_boundary,
        boundary_params=["alpha", "beta"]
    )
    
    results = sim.simulate(
        theta={'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3,
               'alpha': alpha, 'beta': 2.0},
        n_samples=5000
    )
    results_by_alpha[alpha] = results

# Plot RT distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, alpha in enumerate(alphas):
    rts = results_by_alpha[alpha]['rts']
    axes[i].hist(rts, bins=50, density=True)
    axes[i].set_title(f'α = {alpha}')
    axes[i].set_xlabel('RT')
plt.tight_layout()
plt.show()
```

### Example 2: Time-varying Drift

```python
def attention_drift(t, peak_time, peak_strength):
    """Model attention waxing and waning."""
    return peak_strength * np.exp(-0.5 * ((t - peak_time) / 0.5)**2)

sim = Simulator(
    "gamma_drift",
    drift=attention_drift,
    drift_params=["peak_time", "peak_strength"]
)

results = sim.simulate(
    theta={'v': 0.2, 'a': 1.0, 'z': 0.5, 't': 0.3,
           'peak_time': 1.0, 'peak_strength': 0.5},
    n_samples=5000
)
```

### Example 3: Custom Three-Choice Model

```python
def three_choice_race(v1, v2, v3, a, t,
                      max_t=20, n_samples=1000, n_trials=1,
                      delta_t=0.001, random_state=None, **kwargs):
    """Three-accumulator race model."""
    import numpy as np
    
    if random_state is not None:
        np.random.seed(random_state)
    
    rts = np.zeros((n_samples, n_trials, 1))
    choices = np.zeros((n_samples, n_trials, 1))
    
    for trial in range(n_trials):
        for sample in range(n_samples):
            accum = np.zeros(3)
            time = 0.0
            
            while np.max(accum) < a[trial] and time < max_t:
                accum += np.array([v1[trial], v2[trial], v3[trial]]) * delta_t
                accum += np.random.randn(3) * np.sqrt(delta_t)
                time += delta_t
            
            rts[sample, trial, 0] = time + t[trial]
            choices[sample, trial, 0] = np.argmax(accum)
    
    return {
        'rts': rts,
        'choices': choices,
        'metadata': {
            'model': 'three_choice_race',
            'n_samples': n_samples,
            'possible_choices': [0, 1, 2]
        }
    }

sim = Simulator(
    simulator_function=three_choice_race,
    params=["v1", "v2", "v3", "a", "t"],
    nchoices=3,
    choices=[0, 1, 2]
)

results = sim.simulate(
    theta={'v1': 0.5, 'v2': 0.3, 'v3': 0.4, 'a': 1.0, 't': 0.3},
    n_samples=1000
)
```

## Troubleshooting

### Common Errors

**"Unknown model 'xyz'"**
- Check that the model name is valid
- Use `from ssms.config import model_config; print(model_config.keys())` to see available models

**"Must provide 'params' when using custom simulator"**
- When providing a custom `simulator_function` without a base model, you must specify `params`

**"Boundary function must accept 't' as first parameter"**
- Custom boundary functions must have `t` as the first positional parameter

**"Configuration missing required field"**
- Ensure your custom configuration includes all required fields: `params`, `nchoices`, `simulator`

### Getting Help

- Check the docstrings: `help(Simulator)`, `help(ConfigBuilder)`
- Review the test file: `tests/test_simulator_class.py`
- Examine example configurations: `from ssms.config import model_config`

