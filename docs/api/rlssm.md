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

See the [RLSSM tutorial](../core_tutorials/rlssm_tutorial.ipynb)
for presets, building a model, simulating participants, validation, and plots.

## Public API

| Export | Role |
|--------|------|
| `ModelConfig` | Structural model specification (no concrete `theta` values) |
| `Simulator` | Trial-wise generative simulation loop |
| `AssembledModel` | Validated executable form of a config (inference-oriented) |
| `resolve_model` | Resolve preset name or validate a `ModelConfig` |
| `env` | Task environments (`Bandit`, `TaskConfig`, …) |
| `learning` | Learning processes (`RescorlaWagnerDrift`, `RescorlaWagnerSoftmax`, …) |
| `preset` | Preset registry (`get`, `list`, `info`, `register`) |

Import style: `import ssms.rl as rl`.

## Model configuration

`ModelConfig` describes model **structure**, not parameter values. Pass concrete
values as `theta` to `Simulator.simulate()`.

Important fields:

- `decision_process` — SSM name (`"angle"`, `"ddm"`, …)
- `learning_process` — instance satisfying the `LearningProcess` protocol
- `task_environment` — bandit or other task environment (or `TaskConfig` shorthand)
- `response_to_choice` — map SSM response labels to zero-based learning choices
- `learning_backend` / `gradient` — backend policy for simulation and HSSM export
- `context_fields` — observable per-trial context columns such as `"feedback"`,
  `"condition"`, `"block"`, or `"stimulus_id"`
- `include_choice` — optionally include the derived zero-based `choice` column in
  simulator output

### Derived decision-process config (`_ssm_config`)

`ModelConfig` builds an internal decision-process configuration in
`__post_init__` via `ModelConfigBuilder.from_model(decision_process)`. Users
never construct or pass this layer directly.

It supplies SSM parameter names, default bounds, default values, and choice
labels used to validate `choices`, derive `list_params` / `bounds`, and resolve
which SSM parameters are computed by the learning process versus fixed in
simulator `theta`. The assembled model and HSSM bridge consume the *derived*
public fields (`list_params`, `computed_params`, `response_to_choice`, …), not
`_ssm_config` itself.

## Task environment protocols

`TaskEnvironment` is the base protocol for per-trial context and post-decision
signals. Models that map SSM response labels to learning choices require a
`DiscreteChoiceEnvironment` (adds `n_choices` and `response_labels`). Built-in
bandits implement `DiscreteChoiceEnvironment`; `Bandit.n_arms` is an alias for
`n_choices`.

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
  samples responses, task context, and learning updates end to end.
- `mode="ppc"` — observed-history-conditioned resimulation for tutorials, smoke
  tests, and manual checks. Learning state is conditioned on observed trial history;
  RT/response are resimulated and observed context fields are copied into output.
  This is **not** a replacement for PyMC/HSSM posterior predictive checks after
  inference — use HSSM's inference workflow for canonical PPCs.

PPC mode uses the same data contract as inference validation (see below). The
observed panel must include `participant_id`, all `config.response` columns
(default `rt` and `response`), and every configured context field (default
`feedback` for the built-in bandit):

## Data validation

``ModelConfig.validate_data()`` validates **external** trial panels — empirical
data or simulated panels you plan to pass to PPC mode or HSSM. Generative
simulation does not self-validate its output; only ``mode="ppc"`` validates
user-supplied ``observed_data`` before conditioning on it.

Validate empirical or simulated panels before PPC or HSSM handoff:

```python
report = config.validate_data(data)
report.print()
report.raise_for_errors()
```

Required columns are derived from the model config:

- `participant_id`
- every name in `config.response` (default `rt`, `response`)
- every name in `config.context_fields` (default `feedback` for the built-in bandit)

The validator checks balanced panels, contiguous participant blocks, response labels
compatible with `config.choices` and `response_to_choice`, missing values, and
omission sentinels. Within each participant, rows are processed in their existing
order. `trial_id` is an ordinary data/context column, not a reserved reset or ordering
field. Errors include repair hints, for example renaming a reward column or adding it
to `ModelConfig(context_fields=[...])`.

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

The observed response history is used only to condition learning state; PPC output
responses are newly simulated.

### Context fields

Outcome-like values are ordinary context fields. By default, built-in bandits emit a
`"feedback"` column and built-in Rescorla-Wagner learners require `context["feedback"]`
for updates. Use a custom feedback field by configuring both the learner and the model
context:

```python
config = rl.ModelConfig(
    ...,
    learning_process=rl.learning.RescorlaWagnerDrift(feedback_field="reward"),
    context_fields=["reward"],
)
```

For learning processes that update from choices only, declare `required_context_fields`
with runtime fields such as `"choice"` and use no observable context fields:

```python
config = rl.ModelConfig(
    ...,
    learning_process=choice_only_learning,
    context_fields=[],
)
```

## Assembled model (inference integration)

Assemble a config when you need validated metadata or participant-wise computed
parameter functions for downstream packages (for example HSSM):

```python
assembled = config.assemble(backend="jax")

# Derived from config — no manual field lists for standard models
fields = assembled.get_participant_input_fields()
compute_params = assembled.assemble_participant_fn()
```

`assemble_participant_fn()` accepts optional overrides (`input_fields`,
`response_field`) for non-standard layouts. Runtime context fields such as `choice`
are derived internally from `response_to_choice`; observable context fields such as
`feedback` come from `config.context_fields`.

Advanced resolution:

```python
config = rl.resolve_model("2AB_RW_Angle")  # str or ModelConfig
assembled = config.assemble(backend="auto")
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

`RLSSMConfig.from_ssms_model(...)` resolves the `ssms.rl` model, assembles it
with the JAX backend, checks gradient support, and wraps
`AssembledModel.assemble_participant_fn(output="dict")` for HSSM's annotated
computed-parameter contract.

**Note:** HSSM's bridge factory still calls the pre-refactor `compile()` API
until the separate `hssm-rlssm-api` task updates it to `assemble()`.

`ModelConfig.to_hssm_config_dict()` remains useful for structural inspection
and compatibility with lower-level HSSM config workflows. It exports shared
structural fields, plus:

- `learning_backend`, `gradient`, `learning_process_kind`
- `participant_contract` — derived trial input layout (`trial_params`,
  `response_field`, `context_fields`, `input_fields`). Users never construct
  this directly; it is exported for bridge metadata and debugging.

Inference-only placeholders in `to_hssm_config_dict()` (`ssm_logp_func`,
`learning_process`) are not a complete model by themselves. A higher-level
`hssm.RLSSM(data, model=...)` wrapper that consumes `ssms.rl` directly is
planned separately in HSSM.

## Module reference

::: ssms.rl.config.ModelConfig

::: ssms.rl.assembled.AssembledModel

::: ssms.rl.simulator.Simulator

::: ssms.rl.assembled.resolve_model

See also the full package reference on the [ssms](ssms.md) API page.
