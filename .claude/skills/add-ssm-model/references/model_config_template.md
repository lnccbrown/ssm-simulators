# Model Config Template

Use this template when creating a new model config file at
`ssms/config/_modelconfig/{model_name}.py`.

## Simple model (constant boundary, no custom drift)

```python
"""<Model Name> model configuration."""

import cssm
from ssms.basic_simulators import boundary_functions as bf


def get_{model_name}_config():
    """Get the configuration for the <Model Name> model."""
    return {
        "name": "{model_name}",
        "params": ["v", "a", "z", "t"],
        "param_bounds": [[-3.0, 0.3, 0.1, 0.0], [3.0, 2.5, 0.9, 2.0]],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "boundary_params": [],
        "n_params": 4,
        "default_params": [0.0, 1.0, 0.5, 1e-3],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [],
        },
    }
```

## Model with collapsing boundary (e.g., angle)

```python
"""<Model Name> model configuration."""

import cssm
from ssms.basic_simulators import boundary_functions as bf


def get_{model_name}_config():
    """Get the configuration for the <Model Name> model."""
    return {
        "name": "{model_name}",
        "params": ["v", "a", "z", "t", "theta"],
        "param_bounds": [[-3.0, 0.3, 0.1, 1e-3, -0.1], [3.0, 3.0, 0.9, 2.0, 1.3]],
        "boundary_name": "angle",
        "boundary": bf.angle,
        "boundary_params": ["theta"],  # which params from "params" drive the boundary
        "n_params": 5,
        "default_params": [0.0, 1.0, 0.5, 1e-3, 0.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [],
        },
    }
```

## Model with custom drift

```python
"""<Model Name> model configuration."""

import cssm
from ssms.basic_simulators import boundary_functions as bf
from ssms.basic_simulators import drift_functions as df


def get_{model_name}_config():
    """Get the configuration for the <Model Name> model."""
    return {
        "name": "{model_name}",
        "params": ["v", "a", "z", "t", "shape", "scale", "c"],
        "param_bounds": [
            [-3.0, 0.3, 0.1, 1e-3, 1.0, 0.1, 0.0],
            [3.0, 3.0, 0.9, 2.0, 5.0, 2.0, 3.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "boundary_params": [],
        "drift_name": "gamma_drift",
        "drift_fun": df.gamma_drift,
        "drift_params": ["v", "shape", "scale", "c"],
        "n_params": 7,
        "default_params": [1.0, 1.0, 0.5, 1e-3, 2.0, 0.5, 1.0],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flex,  # ddm_flex supports custom drift
        "parameter_transforms": {
            "sampling": [],
            "simulation": [],
        },
    }
```

## Multi-choice race model

```python
"""<Model Name> race model configuration."""

import cssm
from ssms.basic_simulators import boundary_functions as bf


def get_{model_name}_3_config():
    """Get the 3-choice configuration for the <Model Name> model."""
    return {
        "name": "{model_name}_3",
        "params": ["v0", "v1", "v2", "a", "z0", "z1", "z2", "t"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
            [3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "boundary_params": [],
        "n_params": 8,
        "default_params": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1e-3],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "simulator": cssm.race,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [],
        },
    }
```
