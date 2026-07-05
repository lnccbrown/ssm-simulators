---
name: add-ssm-model
description: >
  Guided workflow for adding a new Sequential Sampling Model to ssm-simulators.
  Creates the model config, optional boundary/drift functions, registers the model,
  and validates the simulator runs correctly. Use when the user asks to "add a new
  model", "create a model config", "add a simulator", or "register a new SSM".
---

# Add a New SSM Model to ssm-simulators

This skill walks through adding a new model configuration to ssm-simulators.
It handles the config file, optional custom boundary/drift functions,
registration, and validation.

## Before Starting

Read these reference files for exact code patterns:
- [references/model_config_template.md](references/model_config_template.md)
- [references/validation_checklist.md](references/validation_checklist.md)

Also read an existing model config that's similar to what the user wants:
- Simple 2-choice with constant boundary: `ssms/config/_modelconfig/ddm.py`
- 2-choice with collapsing boundary: `ssms/config/_modelconfig/angle.py`
- Custom drift: `ssms/config/_modelconfig/gamma_drift.py`
- Multi-choice race: `ssms/config/_modelconfig/race.py`
- Multi-choice LBA: `ssms/config/_modelconfig/lba.py`

## Step 1: Gather Requirements

Ask the user for each of these. Do not proceed until all are answered:

1. **Model name** — lowercase, underscores only (e.g., `ddm`, `angle`, `levy`)
2. **Parameters** — ordered list of names (e.g., `["v", "a", "z", "t", "theta"]`)
   - Order matters: this becomes the contract downstream
3. **Parameter bounds** — `[[lower_1, ...], [upper_1, ...]]` matching param order
4. **Default parameter values** — one value per param, used for testing
5. **Number of choices and choice values** (e.g., `nchoices=2, choices=[-1, 1]`)
6. **Boundary type**:
   - Existing: `constant`, `angle`, `weibull_cdf`
     (see `ssms/basic_simulators/boundary_functions.py` for full list)
   - Or describe the custom boundary to implement
7. **Drift type** — standard constant drift, or custom?
   - Existing: `constant`, `gamma_drift`
     (see `ssms/basic_simulators/drift_functions.py` for full list)
   - Or describe the custom drift to implement
8. **Which simulator** to use:
   - `cssm.ddm_flexbound` — standard for 2-choice models with flexible boundaries
   - `cssm.ddm_flex` — for models needing custom drift functions
   - `cssm.race` — for race models (multiple choices, independent accumulators)
   - `cssm.lba` — for linear ballistic accumulator models
   - Check `src/cssm/` for other available simulators
9. **Number of particles** — usually 1 (for single-accumulator models), or
   matches nchoices for race/LBA models

## Step 2: Custom Boundary (if needed)

If the model needs a new boundary function not already in
`ssms/basic_simulators/boundary_functions.py`:

1. Add the function to `boundary_functions.py`:
   ```python
   def my_boundary(t: np.ndarray, **params) -> np.ndarray | float:
       """Describe the boundary."""
       # t is an array of time points
       # params are the boundary-specific parameters from the model config
       return result
   ```

2. Register in `ssms/config/_modelconfig/base.py` under `boundary_config`:
   ```python
   "my_boundary": {
       "fun": bf.my_boundary,
       "params": ["param_a", "param_b"],  # which model params drive this boundary
   }
   ```

3. Also register in `ssms/config/boundary_registry.py` if using the registry API.

## Step 3: Custom Drift (if needed)

Same pattern as boundaries. Add to `ssms/basic_simulators/drift_functions.py`,
register in `base.py` under `drift_config`, and optionally in `drift_registry.py`.

The model config must include `"drift_name"`, `"drift_fun"`, and `"drift_params"`
fields when using a custom drift.

## Step 4: Create Model Config

Create `ssms/config/_modelconfig/{model_name}.py`.

Use the template from [references/model_config_template.md](references/model_config_template.md).
The required fields are:

| Field | Type | Description |
|-------|------|-------------|
| `"name"` | str | Must match the config key in `__init__.py` |
| `"params"` | list[str] | Parameter names in order |
| `"param_bounds"` | list[list[float]] | `[[lowers], [uppers]]` |
| `"boundary_name"` | str | Matches a key in `base.py` `boundary_config` |
| `"boundary"` | Callable | The boundary function object |
| `"boundary_params"` | list[str] | Subset of `params` that drive the boundary |
| `"n_params"` | int | `len(params)` |
| `"default_params"` | list[float] | Default values for testing |
| `"nchoices"` | int | Number of discrete choices |
| `"choices"` | list[int] | Choice values |
| `"n_particles"` | int | Usually 1 or nchoices |
| `"simulator"` | Callable | Cython simulator function |
| `"parameter_transforms"` | dict | `{"sampling": [], "simulation": []}` |

Optional fields for models with custom drift:
- `"drift_name"` — string name
- `"drift_fun"` — drift function object
- `"drift_params"` — which params drive the drift

## Step 5: Register the Model

Edit `ssms/config/_modelconfig/__init__.py`:

1. **Import** — add near the top with the other imports:
   ```python
   from .{model_name} import get_{model_name}_config
   ```

2. **Register** — add to the `configs` dict inside `get_model_config()`:
   ```python
   "{model_name}": get_{model_name}_config(),
   ```

3. **Export** — add to `__all__`:
   ```python
   "get_{model_name}_config",
   ```

## Step 6: Validate

Run these checks. **All must pass.**

```python
from ssms.config._modelconfig import get_model_config
from ssms.basic_simulators import Simulator

# 1. Config loads without error
config = get_model_config()["{model_name}"]
print(f"Name: {config['name']}")
print(f"Params ({config['n_params']}): {config['params']}")
print(f"Bounds: {config['param_bounds']}")
print(f"Defaults: {config['default_params']}")
print(f"Choices ({config['nchoices']}): {config['choices']}")
print(f"Boundary: {config['boundary_name']}")
print(f"Simulator: {config['simulator']}")

# 2. Param count consistency
assert len(config["params"]) == config["n_params"]
assert len(config["default_params"]) == config["n_params"]
assert len(config["param_bounds"][0]) == config["n_params"]
assert len(config["param_bounds"][1]) == config["n_params"]

# 3. Bounds are valid (lower < upper for each param)
for i, (lo, hi) in enumerate(zip(config["param_bounds"][0], config["param_bounds"][1])):
    assert lo < hi, f"Param {config['params'][i]}: lower {lo} >= upper {hi}"

# 4. Defaults are within bounds
for i, (val, lo, hi) in enumerate(
    zip(config["default_params"], config["param_bounds"][0], config["param_bounds"][1])
):
    assert lo <= val <= hi, f"Param {config['params'][i]}: default {val} outside [{lo}, {hi}]"

# 5. Simulator runs with default params
sim = Simulator(model="{model_name}")
result = sim.simulate(theta=config["default_params"], n_samples=1000)
print(f"RT shape: {result['rts'].shape}")
print(f"Choices shape: {result['choices'].shape}")
unique_choices = set(int(c) for c in result["choices"].flatten() if c != -999.0)
print(f"Unique choices: {unique_choices}")
assert unique_choices.issubset(set(config["choices"])), (
    f"Simulator produced choices {unique_choices} not in config {config['choices']}"
)

# 6. Simulator runs with random params within bounds
import numpy as np
rng = np.random.default_rng(42)
for trial in range(5):
    random_params = [
        rng.uniform(lo, hi)
        for lo, hi in zip(config["param_bounds"][0], config["param_bounds"][1])
    ]
    result = sim.simulate(theta=random_params, n_samples=100)
    assert result["rts"].shape[0] == 100, f"Trial {trial}: wrong output shape"

print("All validation checks passed.")
```

## Step 7: Write Tests

Create `tests/test_{model_name}_config.py` with:

1. Test config structure — all required keys present
2. Test param count consistency (same checks as validation above)
3. Test simulator with default params — correct output shape, valid choices
4. Test simulator with random params within bounds — no crashes
5. Test simulator with boundary params — verify output changes with boundary

If the model supports the `_deadline` suffix, also test:
```python
sim_deadline = Simulator(model="{model_name}_deadline")
result = sim_deadline.simulate(theta=config["default_params"], n_samples=1000)
# Check that some trials have OMISSION_SENTINEL (-999.0) in rts
```

## Common Pitfalls

1. **Wrong param order** — the most common error. Double-check against the
   Cython simulator's expected parameter ordering.
2. **Boundary params not a subset of model params** — `"boundary_params"`
   must list parameter names that appear in `"params"`.
3. **Missing `param_bounds_dict`** — this is auto-generated by
   `_normalize_param_bounds()` in `__init__.py`. Don't add it manually.
4. **Simulator expects different param count** — some Cython simulators
   have fixed parameter expectations. Check `src/cssm/` source if unsure.
5. **Choices mismatch** — for 2-choice models use `[-1, 1]`, not `[0, 1]`.
   For n-choice race/LBA models use `[0, 1, ..., n-1]`.
