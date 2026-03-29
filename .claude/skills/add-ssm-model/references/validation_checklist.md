# Validation Checklist for New SSM Models

Run through every item. All must pass before the model is considered complete.

## Config Structure

- [ ] `"name"` matches the key used in `get_model_config()` dict
- [ ] `"params"` is a list of strings, no duplicates
- [ ] `len(params) == n_params`
- [ ] `len(default_params) == n_params`
- [ ] `len(param_bounds[0]) == n_params` (lower bounds)
- [ ] `len(param_bounds[1]) == n_params` (upper bounds)
- [ ] Every lower bound < corresponding upper bound
- [ ] Every default param is within its bounds
- [ ] `"boundary_name"` matches a key in `base.py` `boundary_config`
- [ ] `"boundary"` is the actual function object (not a string)
- [ ] `"boundary_params"` are all names that appear in `"params"`
- [ ] If custom drift: `"drift_params"` are all names that appear in `"params"`
- [ ] `"choices"` has length == `"nchoices"`
- [ ] `"simulator"` is a valid Cython function from `cssm`

## Registration

- [ ] Import added to `ssms/config/_modelconfig/__init__.py`
- [ ] Entry added to `get_model_config()` dict
- [ ] Entry added to `__all__` list
- [ ] Module-level `_validate_configs()` passes (runs on import)

## Simulator Execution

- [ ] `Simulator(model="{model_name}")` instantiates without error
- [ ] `.simulate(theta=default_params, n_samples=1000)` runs without error
- [ ] Output `rts` shape is `(n_samples, 1)` or `(n_samples,)`
- [ ] Output `choices` shape is `(n_samples, 1)` or `(n_samples,)`
- [ ] All output choices are in the set defined by `"choices"` (ignoring -999.0 sentinel)
- [ ] 5 runs with random params within bounds all succeed
- [ ] Multi-threaded simulation works: `Simulator(model="{model_name}").simulate(..., n_threads=2)`

## Boundary / Drift (if custom)

- [ ] Boundary function is registered in `base.py` `boundary_config`
- [ ] Boundary function returns correct shape (scalar or array matching `t`)
- [ ] Boundary function is positive for all valid parameter combinations
- [ ] Drift function is registered in `base.py` `drift_config` (if custom drift)
- [ ] Drift function returns correct shape

## Deadline Support

All models automatically support the `_deadline` suffix. Verify:

- [ ] `Simulator(model="{model_name}_deadline")` works
- [ ] Deadline simulation produces some -999.0 values in `rts` (timeouts)

## Tests

- [ ] Test file created at `tests/test_{model_name}_config.py`
- [ ] Tests cover: config structure, simulator execution, boundary cases
- [ ] `uv run pytest tests/test_{model_name}_config.py` passes

## Pre-commit

- [ ] `uv run pre-commit run --all-files` passes after all changes

## Downstream Awareness

After adding the model to ssm-simulators, the following downstream steps
are needed (handled by separate skills/processes):

- HSSM: create `src/hssm/modelconfig/{model_name}_config.py` + add to `SupportedModels`
- LANfactory/LAN_pipeline_minimal: create training YAML configs (if training networks)
- HuggingFace: upload trained ONNX networks (if applicable)
