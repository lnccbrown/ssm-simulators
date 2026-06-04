# RLSSM simulation (`ssms.rl`)

The `ssms.rl` namespace provides a compositional framework for simulating
reinforcement-learning sequential sampling models (RLSSMs). Combine a learning
process, SSM decision process, and task environment; simulate balanced panels
compatible with HSSM inference.

## Quick start

```python
import ssms.rl as rl

print(rl.preset.info("2AB_RW_Angle"))
config = rl.preset.get("2AB_RW_Angle")
sim = rl.Simulator(config)
data = sim.simulate(
    theta={
        "rl_alpha": 0.2,
        "scaler": 2.0,
        "a": 1.5,
        "z": 0.5,
        "t": 0.3,
        "theta": 0.2,
    },
    n_trials=200,
    n_participants=20,
    random_state=42,
)
```

See the [RLSSM Simulator tutorial](../core_tutorials/rlssm_simulator_demo.ipynb)
for presets, custom components, response mapping, and plots.

## Public API

| Export | Role |
|--------|------|
| `ModelConfig` | Structural model specification (no concrete `theta` values) |
| `Simulator` | Trial-wise generative simulation loop |
| `CompiledModel` | Validated executable form of a config (inference-oriented) |
| `resolve_model` | Resolve preset name or validate a `ModelConfig` |
| `env` | Task environments (`Bandit`, `TaskConfig`, …) |
| `learning` | Learning processes (`RescorlaWagnerDeltaRule`, …) |
| `preset` | Preset registry (`get`, `list`, `info`, `register`) |

Import style: `import ssms.rl as rl`.

## Model configuration

`ModelConfig` describes model **structure**, not parameter values. Pass concrete
values as `theta` to `Simulator.simulate()`.

Important fields:

- `decision_process` — SSM name (`"angle"`, `"ddm"`, …)
- `learning_process` — instance satisfying the `LearningProcess` protocol
- `task_environment` — bandit or other task environment (or `TaskConfig` shorthand)
- `response_mapping` — map SSM response labels to zero-based learning actions
- `learning_backend` / `gradient` — backend policy for simulation and HSSM export
- **`outcome_field`** — reward/outcome column name (default `"feedback"`); set
  to `None` for outcome-free learning processes
- `extra_fields` — extra data columns; defaults to `[outcome_field]` plus task extras

## Participant-wise parameters

`Simulator.simulate()` accepts scalar theta values shared by all participants
and one-dimensional participant-wise values. When any theta value is
participant-wise, all participant-wise values must have the same length. If
`n_participants` is omitted, that length is used as the participant count:

```python
data = sim.simulate(
    theta={
        "rl_alpha": [0.15, 0.25, 0.35],
        "scaler": 2.0,
        "a": [1.1, 1.4, 1.7],
        "z": 0.5,
        "t": 0.3,
        "theta": 0.2,
    },
    n_trials=200,
    random_state=42,
)
```

Passing `n_participants` explicitly is allowed, but it must match the
participant-wise theta length.

## Simulation modes

`Simulator.simulate()` supports two modes:

- `mode="generative"` — the default unconstrained simulation loop. The simulator
  samples responses, task outcomes, and learning updates end to end.
- `mode="ppc"` — observed-history-conditioned posterior predictive simulation.
  The simulator generates new RT/response values for each observed trial, copies
  the observed outcome column into the output, and updates learning state from
  the observed response/outcome history.

PPC mode uses the same data contract as inference validation (see below). The
observed panel must include `participant_id`, `trial_id`, all `config.response`
columns (default `rt` and `response`), and the configured outcome column (default
`feedback`):

## Data validation

Validate empirical or simulated panels before PPC or HSSM handoff:

```python
report = config.validate_data(data)
report.print()
report.raise_for_errors()
```

Required columns are derived from the model config:

- `participant_id`, `trial_id`
- every name in `config.response` (default `rt`, `response`)
- every name in `config.extra_fields` (includes `outcome_field`, default `feedback`)

The validator checks balanced panels, contiguous participant blocks, contiguous
zero-based `trial_id` within each participant, response labels compatible with
`config.choices` and `response_mapping`, missing values, and omission sentinels.
Errors include repair hints, for example renaming a reward column or setting
`ModelConfig(outcome_field="reward")`.

PPC mode example (observed data must satisfy the same contract):

```python
observed = sim.simulate(
    theta={
        "rl_alpha": 0.2,
        "scaler": 2.0,
        "a": 1.5,
        "z": 0.5,
        "t": 0.3,
        "theta": 0.2,
    },
    n_trials=200,
    n_participants=20,
    random_state=1,
)

ppc = sim.simulate(
    theta={
        "rl_alpha": 0.2,
        "scaler": 2.0,
        "a": 1.5,
        "z": 0.5,
        "t": 0.3,
        "theta": 0.2,
    },
    mode="ppc",
    observed_data=observed,
    random_state=2,
)
```

Within each participant, `trial_id` values must be contiguous and zero-based.
The observed response history is used only to condition learning state; PPC
output responses are newly simulated.

### Outcome column naming

By default, simulator output includes a `"feedback"` column and compiled
participant functions expect that column in participant history arrays. Use a
custom name when needed:

```python
config = rl.ModelConfig(
    ...,
    outcome_field="reward",
)
```

For learning processes that update from choices only:

```python
config = rl.ModelConfig(
    ...,
    outcome_field=None,
    extra_fields=[],
)
```

## Compiled model (inference integration)

Compile a config when you need validated metadata or participant-wise computed
parameter functions for downstream packages (for example HSSM):

```python
compiled = config.compile(backend="jax")

# Derived from config — no manual field lists for standard models
fields = compiled.participant_input_fields()
compute_params = compiled.compile_participant_fn()
```

`compile_participant_fn()` accepts optional overrides (`input_fields`,
`response_field`, `outcome_field`) for non-standard layouts.

Advanced resolution:

```python
config = rl.resolve_model("2AB_RW_Angle")  # str or ModelConfig
compiled = config.compile(backend="auto")
```

## HSSM bridge

The active HSSM handoff path is HSSM's bridge factory:

```python
import hssm
import ssms.rl as rl

ssms_config = rl.preset.get("2AB_RW_Angle")
hssm_config = hssm.rl.RLSSMConfig.from_ssms_model(ssms_config)
model = hssm.RLSSM(data=data, model_config=hssm_config)
```

`RLSSMConfig.from_ssms_model(...)` resolves the `ssms.rl` model, compiles it
with the JAX backend, checks gradient support, and wraps
`CompiledModel.compile_participant_fn(output="dict")` for HSSM's annotated
computed-parameter contract.

`ModelConfig.to_hssm_config_dict()` remains useful for structural inspection
and compatibility with lower-level HSSM config workflows. It exports shared
structural fields, plus:

- `learning_backend`, `gradient`, `learning_process_kind`
- `participant_contract` — derived trial input layout (`trial_params`,
  `response_field`, `outcome_field`, `input_fields`)

Inference-only placeholders in `to_hssm_config_dict()` (`ssm_logp_func`,
`learning_process`) are not a complete model by themselves. A higher-level
`hssm.RLSSM(data, model=...)` wrapper that consumes `ssms.rl` directly is
planned separately in HSSM.

## Module reference

::: ssms.rl.config.ModelConfig

::: ssms.rl.compiled.CompiledModel

::: ssms.rl.simulator.Simulator

::: ssms.rl.compiled.resolve_model

See also the full package reference on the [ssms](ssms.md) API page.
