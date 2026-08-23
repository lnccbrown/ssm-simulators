# Why structured observation results exist

Sequential-sampling models do not all produce the same kind of observation. The familiar
case is reaction time plus a categorical choice, but a model may instead produce only a
choice, a bounded confidence rating, several continuous measurements, a circular angle,
or a fixed-width mixture of those fields.

Historically, ssm-simulators has exposed `rts` and `choices`. Those arrays remain the
default interface for existing simulators. They are heavily used by likelihood-estimation,
training-data, HSSM, RLSSM, plotting, and external integration code, and this new contract
does not silently migrate any of those consumers.

The native structured-observation contract gives *new or explicitly opted-in producers*
one general representation:

```text
observations[n_samples, n_trials, obs_dim]
```

The final axis is described by an ordered, versioned schema. Every stochastic scalar
observation—including reaction time, when present—is treated symmetrically. A sidecar
mask identifies complete omitted rows.

## Observations are not every numeric output

The schema contains only fields that are jointly observed and evaluated by a model's
likelihood. It does not absorb auxiliary state merely because that state is numeric.
Examples that remain outside the observation axis include:

- simulated trajectories and time-varying boundaries;
- participant and trial identifiers;
- task context, feedback, and block labels;
- latent reinforcement-learning values and computed trial parameters;
- training labels and likelihood-density metadata.

Those values remain in their existing result or open `metadata` locations. This boundary
keeps a fixed-width observation vector useful without pretending that it can encode every
kind of simulator output.

## Why the shape is never squeezed

Keeping all three axes makes singleton cases unambiguous. A response-only model with one
sample and one trial has shape `(1, 1, 1)`, while an RT-plus-confidence model has shape
`(1, 1, 2)`. Consumers do not need model-name rules or guesses about whether a singleton
axis represented samples, trials, or observation fields.

Legacy `rts` and `choices` may still use their historical squeezed shapes. Converting
those results requires the opt-in
[legacy-result normalizer](../how_to/normalize_simulator_results.md), explicit original
sample/trial counts, an explicit source mapping, and one projected source named as the
omission authority. The native validator and normalizer do not guess how legacy arrays
or their omission sentinels should be interpreted.

## Why schema entries are closed but metadata is open

Version 1 recognizes exactly three scalar-domain kinds: `categorical`, `continuous`, and
`circular`. Each entry accepts only the keys defined for that kind, so a misspelled bound
or an unsupported observation semantic fails immediately.

The surrounding `metadata` mapping has the opposite policy. Only
`observation_schema_version` and `observation_schema` are reserved. Producer-owned fields
such as simulator identity, possible choices, parameters, timing limits, boundaries, and
trajectories are accepted without interpretation and retain object identity in the
validated shallow view.

Adding unrelated producer metadata therefore does not require a schema-version change.
Adding a new stochastic-observation semantic that version 1 cannot express does.

## What version 1 deliberately does not represent

Version 1 covers dense, fixed-width, floating NumPy arrays of scalar numeric fields and
complete-row omissions. It does not claim support for ragged events, partial observation
vectors, array-valued fields, string categories, simplex or unit-vector joint support, or
continuous/mixed-response likelihood-estimation and LAN training.

See the [normative result contract](../reference/observation_result_contract.md) for the
exact schema, support, omission, and validation rules.
