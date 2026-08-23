# Normalize legacy simulator results

Use `normalize_simulator_result` when a consumer needs an opt-in structured view of an
existing `rts`/`choices` result. The adapter returns a shallow copy, retains every legacy
key, and does not change registered simulators or their default outputs.

The caller supplies the original sample and trial counts, ordered
[observation schema](../reference/observation_result_contract.md), exact source-to-field
projection, and projected source that authoritatively records omissions. The adapter
never infers these from values, names, or model metadata.

## Project RT and choice

Projection pairs use `(legacy source key, schema field name)` order and must cover the
schema exactly once in schema order. Deadline and RT-based RL simulators use
`omission_source="rts"`, because an omitted RT can retain a sampled or placeholder choice.

```python
--8<-- "docs/snippets/observation_results/legacy_rt_choice.py"
```

The expected counts reverse only the package's historical singleton squeezing:

| Samples | Trials | Required legacy shape | Canonical shape |
| --- | --- | --- | --- |
| 1 | 1 | `(1, 1)` | `(1, 1, obs_dim)` |
| many | 1 | `(n_samples, 1)` | `(n_samples, 1, obs_dim)` |
| 1 | many | `(n_trials, 1)` | `(1, n_trials, obs_dim)` |
| many | many | `(n_samples, n_trials, 1)` | `(n_samples, n_trials, obs_dim)` |

An equal-size array with another shape is rejected. This keeps `(N, 1)` deterministic:
the caller states whether `N` means samples or trials.

## Project a response-only result

For a choice-only RLSSM, map only `choices` and set `omission_source="choices"`. Its
compatibility `rts=-1` array remains available but is neither projected nor interpreted.

```python
--8<-- "docs/snippets/observation_results/legacy_response_only.py"
```

An integer-only response is promoted to `float64` so the canonical array can contain NaN
omissions; a floating response keeps its dtype. For two sources, NumPy selects their
combined floating dtype. Categorical labels and source values must remain exactly
representable after that promotion.

## Handle omissions and metadata deliberately

`omission_source` is required and must name a projected source. Its sentinel marks the
whole canonical row omitted, replaces that row with NaNs, and makes auxiliary projected
values irrelevant. This covers both deadline output with a valid retained choice and
legacy output where every projected source contains the sentinel.

A sentinel in an auxiliary source while the authority is available is contradictory and
raises an error. Unprojected sources never participate, so response-only normalization
ignores a dummy RT even if it contains the sentinel.

The result and metadata are shallow-copied; legacy arrays and producer-owned values such
as boundaries and trajectories retain identity. The adapter adds `observations` and
`omission_mask` to the result, and adds only `observation_schema_version` and
`observation_schema` to metadata. Identical reserved metadata is retained, conflicts are
rejected, and existing canonical result keys are never overwritten.

## Respect the compatibility boundary

The legacy adapter accepts one or two scalar fields. Wider new producers should emit the
native fixed-width contract and call `validate_observation_result` directly; see the
[executable native examples](../reference/observation_result_contract.md#executable-examples)
and [basic simulator API](../api/basic_simulators.md).

Normalization is additive and opt-in. It does not migrate dataset generation;
KDE/LAN/CPN/OPN paths; RL trial extraction or panel assembly; HSSM `simulate_data`;
ssms-gui; or external consumers of the existing keys and squeezed shapes. Such consumers
can construct a normalized view alongside the untouched legacy result when they
explicitly adopt the structured contract. Contributors should also review the
[integration guidance](../contributing/README.md).
