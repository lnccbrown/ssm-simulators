# RLSSM API (`ssms.rl`)

The `ssms.rl` namespace is the simulator-side API for reinforcement-learning
sequential sampling models. It defines model structure, task environments,
learning processes, simulation, validation, and the neutral assembled-model
contract consumed by inference packages.

This page is a reference roll-up. For procedures, start with
[Simulate your first RLSSM](../core_tutorials/rlssm_tutorial.ipynb), then use the
[advanced component guide](../core_tutorials/rlssm_advanced_tutorial.ipynb),
[choice-only guide](../core_tutorials/choice_only_rl_models.ipynb), or
[HSSM handoff guide](../core_tutorials/rlssm_simulation_hssm_handoff.ipynb).

## Namespace exports

Use `import ssms.rl as rl`. The root namespace exports:

| Export | Contract |
| --- | --- |
| `Simulator` | Trial-wise generative and observed-history-conditioned simulation |
| `ModelConfig` | Structural model specification; concrete parameter values are passed separately |
| `AssembledModel` | Validated, backend-resolved participant-function contract |
| `resolve_model` | Resolve a preset name or validate a `ModelConfig` |
| `env` | Task-environment protocols, implementations, and registry |
| `learning` | Learning-process protocol and built-in implementations |
| `preset` | Preset registry (`get`, `list`, `info`, `register`) |

## Model configuration contract

`ModelConfig` describes structure, not participant parameter values. Its main
fields are:

| Field | Meaning |
| --- | --- |
| `decision_process` | Registered SSM name such as `angle` or `ddm` |
| `learning_process` | Object satisfying `LearningProcess` |
| `task_environment` | Environment object or `TaskConfig` shorthand |
| `response_to_choice` | Mapping from SSM response labels to zero-based learning choices |
| `learning_backend`, `gradient` | Backend and differentiability policy |
| `context_fields` | Observable trial-context columns such as `feedback` |
| `include_choice` | Whether simulator output includes the derived zero-based choice |

The private derived `_ssm_config` is built from the registered decision process
and is not a constructor input. Public derived fields such as `list_params`,
`bounds`, `computed_params`, and `response_to_choice` are the supported
integration surface.

## Built-in learning processes

| Class | Computed output | Action count |
| --- | --- | --- |
| `RescorlaWagnerDeltaRule` | State/update only | 2 or more |
| `RescorlaWagnerDrift` | `v` | exactly 2 |
| `RescorlaWagnerSoftmax` | `q0`, `q1`, ... | 2 or more |
| `RescorlaWagnerRaceDrifts` | `v0`, `v1`, ... | 2 or more |
| `RescorlaWagnerDualAlphaRule` | State/update only | 2 or more |
| `RescorlaWagnerDualAlphaDrift` | `v` | exactly 2 |
| `RescorlaWagnerDualAlphaSoftmax` | `q0`, `q1`, ... | 2 or more |

Drift learners compute trial-wise drift from learned value differences.
Softmax learners expose Q-values and leave inverse-temperature application to
the decision process. The dual-alpha variants distinguish positive and
negative prediction errors.

## Preset registry

`ssms.rl.preset` is the source of truth for built-in RLSSM structures. Query the
runtime registry with `preset.list()` and inspect a resolved contract with
`preset.info(name)`.

| Preset | Decision process | Learning process | Response |
| --- | --- | --- | --- |
| `2AB_RW_DDM` | `ddm` | `RescorlaWagnerDrift` | `rt`, `response` |
| `2AB_RW_Angle` | `angle` | `RescorlaWagnerDrift` | `rt`, `response` |
| `2AB_RW_Weibull` | `weibull` | `RescorlaWagnerDrift` | `rt`, `response` |
| `2AB_RW_DualAlpha_Angle` | `angle` | `RescorlaWagnerDualAlphaDrift` | `rt`, `response` |
| `2AB_RW_InvTempSoftmax` | `inv_temp_softmax_2` | `RescorlaWagnerSoftmax` | `response` |
| `2AB_RW_DualAlpha_InvTempSoftmax` | `inv_temp_softmax_2` | `RescorlaWagnerDualAlphaSoftmax` | `response` |
| `3AB_RW_InvTempSoftmax` | `inv_temp_softmax_3` | `RescorlaWagnerSoftmax` | `response` |
| `4AB_RW_InvTempSoftmax` | `inv_temp_softmax_4` | `RescorlaWagnerSoftmax` | `response` |
| `4AB_RW_RaceNoBiasAngle` | `race_no_bias_angle_4` | `RescorlaWagnerRaceDrifts` | `rt`, `response` |

## Simulator contract

`Simulator.simulate()` accepts scalar parameter values shared by all
participants or one-dimensional participant-wise values. All participant-wise
arrays must have the same length; an explicit `n_participants` must match it.

The supported modes are:

| Mode | Contract |
| --- | --- |
| `generative` | Sample task context, response, and learning updates end to end |
| `ppc` | Condition learning on observed history while resimulating responses |

PPC input must satisfy the same panel contract as inference validation. The
observed response history conditions learning; returned responses are newly
simulated.

## Choice-only contract

The inverse-temperature softmax presets declare `response=["response"]` and do
not define an RT likelihood. Generative output retains `rt=-1.0` only as a
compatibility placeholder. That value is distinct from
`OMISSION_SENTINEL == -999.0`.

Validation, PPC, and HSSM handoff use a response-only table with the placeholder
column removed. Custom tasks may pair an `inv_temp_softmax_N` decision process
with a compatible learning process and environment.

## Task environments and registry

`TaskEnvironment` defines per-trial context and post-decision signals.
`DiscreteChoiceEnvironment` adds `n_choices` and ordered `response_labels`.
Built-in bandits satisfy the discrete protocol; `Bandit.n_arms` aliases
`n_choices`.

`TaskEnvironmentBuilder` is the callable type stored by the task registry.
`register_task()` adds a builder, `registered_tasks()` lists available names,
and `TaskConfig.build_environment()` resolves one. The built-in `bandit` task
supports Bernoulli and Gaussian rewards.

## Data-validation contract

`ModelConfig.validate_data()` and `validate_rlssm_data()` return a
`DataValidationReport` containing zero or more `DataValidationIssue` values.
Call `raise_for_errors()` when invalid panels must fail fast.

Required columns are derived from the model:

- `participant_id`;
- every configured response column;
- every observable `context_fields` entry.

Validation checks balanced panels, contiguous participant blocks, response
labels and mappings, missing values, RT validity, and omission sentinels. Rows
within each participant are processed in their existing order. `trial_id` is an
ordinary column, not a reserved ordering instruction.

## Assembled-model and HSSM contracts

`ModelConfig.assemble()` returns an `AssembledModel` with backend-resolved
participant input fields and computed-parameter functions. Runtime choice is
derived from `response_to_choice`; observable context comes from
`context_fields`.

HSSM owns inference and exposes `hssm.RLSSM(data, model=...)` as the normal
entry point for named ssms presets. An in-memory custom `ModelConfig` can use
the advanced `hssm.rl.RLSSMConfig.from_ssms_model(...)` path. The
[HSSM handoff guide](../core_tutorials/rlssm_simulation_hssm_handoff.ipynb)
owns that procedure, and HSSM's rendered
[RLSSM reference](https://lnccbrown.github.io/HSSM/api/rl/) owns inference-side
options.

`ModelConfig.to_hssm_config_dict()` remains an inspection and compatibility
surface. Its inference placeholders are not a complete HSSM model and should
not be assembled manually.

## Core objects

::: ssms.rl.config.ModelConfig

::: ssms.rl.assembled.AssembledModel

::: ssms.rl.simulator.Simulator

::: ssms.rl.assembled.resolve_model

## Preset functions

::: ssms.rl.preset.get

::: ssms.rl.preset.list

::: ssms.rl.preset.info

::: ssms.rl.preset.register

## Environment objects

::: ssms.rl.env.TaskEnvironment

::: ssms.rl.env.DiscreteChoiceEnvironment

::: ssms.rl.env.Bandit

::: ssms.rl.env.TaskConfig

::: ssms.rl.env.register_task

::: ssms.rl.env.registered_tasks

## Validation objects

::: ssms.rl.validation.DataValidationIssue

::: ssms.rl.validation.DataValidationReport

::: ssms.rl.validation.validate_rlssm_data

## Learning-process objects

::: ssms.rl.learning.LearningProcess

::: ssms.rl.learning.RescorlaWagnerDeltaRule

::: ssms.rl.learning.RescorlaWagnerDrift

::: ssms.rl.learning.RescorlaWagnerSoftmax

::: ssms.rl.learning.RescorlaWagnerRaceDrifts

::: ssms.rl.learning.RescorlaWagnerDualAlphaRule

::: ssms.rl.learning.RescorlaWagnerDualAlphaDrift

::: ssms.rl.learning.RescorlaWagnerDualAlphaSoftmax
