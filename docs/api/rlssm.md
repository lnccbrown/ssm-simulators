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

## HSSM config bridge

`ModelConfig.to_hssm_config_dict()` exports structural fields for HSSM's
`RLSSMConfig.from_rlssm_dict()`, plus:

- `learning_backend`, `gradient`, `learning_process_kind`
- `participant_contract` — derived trial input layout (`trial_params`,
  `response_field`, `outcome_field`, `input_fields`)

Inference-only placeholders (`ssm_logp_func`, `learning_process`) remain for
HSSM to fill. A higher-level `hssm.RLSSM(data, model=...)` wrapper that consumes
`ssms.rl` directly is planned separately in HSSM.

## Module reference

::: ssms.rl.config.ModelConfig

::: ssms.rl.compiled.CompiledModel

::: ssms.rl.simulator.Simulator

::: ssms.rl.compiled.resolve_model

See also the full package reference on the [ssms](ssms.md) API page.
