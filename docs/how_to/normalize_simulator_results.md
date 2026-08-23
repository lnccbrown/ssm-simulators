# Normalize legacy simulator results

Use `normalize_simulator_result` when a consumer already has an `rts`/`choices`
result and needs an opt-in view of the
[structured observation contract](../reference/observation_result_contract.md).
The adapter does not change `simulator`, `Simulator.simulate`, or any registered model.
It returns a new shallow dictionary and retains every legacy key.

The caller must know five things: the original sample count, the original trial count,
the ordered observation schema, the exact legacy-source-to-schema mapping, and which
projected legacy source authoritatively records omissions. The adapter never derives
them from array values, key names, schema field names, or model names.

## Project RT and choice

Pass projection pairs in `(legacy source key, schema field name)` order. They must cover
the schema exactly once and in schema order. Set `omission_source="rts"` because
deadline and RT-based RL simulators record omission in the RT source even when the
choice source retains a sampled or placeholder label.

```python
--8<-- "docs/snippets/observation_results/legacy_rt_choice.py"
```

The expected counts reverse only the package's historical singleton squeezing:

| Samples | Trials | Required legacy source shape | Canonical shape |
| --- | --- | --- | --- |
| 1 | 1 | `(1, 1)` | `(1, 1, obs_dim)` |
| many | 1 | `(n_samples, 1)` | `(n_samples, 1, obs_dim)` |
| 1 | many | `(n_trials, 1)` | `(1, n_trials, obs_dim)` |
| many | many | `(n_samples, n_trials, 1)` | `(n_samples, n_trials, obs_dim)` |

An array with the right element count but a different shape is rejected. This is what
keeps a squeezed `(N, 1)` source deterministic: the caller, not the adapter, says whether
`N` means samples or trials.

## Project a response-only result

For a choice-only RLSSM, map only `choices` and set `omission_source="choices"`. Its
compatibility `rts=-1` array is neither an observation nor an omission signal; it remains
available under the original key.

```python
--8<-- "docs/snippets/observation_results/legacy_response_only.py"
```

An integer-only response source is promoted to `float64`, which gives the canonical
array a floating dtype capable of representing NaN omissions. A floating response source
keeps its dtype. With two projected sources, NumPy promotion selects the combined
floating dtype. The validator then checks that categorical labels—and the integer source
values used for them—remain exactly representable in that dtype.

## Handle omissions and metadata deliberately

`omission_source` is required and must name exactly one key in `source_projection`. The
adapter marks a row omitted exactly when that source contains `OMISSION_SENTINEL`, then
converts every canonical field in the row to NaN. Auxiliary projected values on that
omitted row are ignored. This supports historical deadline output where `rts=-999` but
`choices` retains a valid label, as well as legacy rows where both sources contain the
sentinel.

If the authoritative source is non-sentinel, a sentinel in any auxiliary projected
source is contradictory and raises an error. Unprojected sources never participate, so
a response-only projection ignores its dummy RT even if that array happens to contain
the sentinel. The required keyword makes the policy explicit; the adapter does not infer
authority from field names, source keys, values, or model metadata.

The returned result and metadata are shallow copies. Legacy arrays and producer-owned
metadata values such as boundaries and trajectories keep their identity. The adapter
adds only `observation_schema_version` and `observation_schema`. An already-present
identical reserved value is retained, while a conflict is rejected. Pre-existing
`observations` or `omission_mask` keys are also rejected rather than overwritten.

## Validate native multi-field output directly

The legacy adapter is limited to one or two scalar fields. New producers with three or
more fields should emit native fixed-width output and use
`validate_observation_result` directly:

```python
--8<-- "docs/snippets/observation_results/native_multi_field.py"
```

See the [basic simulator API](../api/basic_simulators.md) for both public functions and
the [contributor guide](../contributing/README.md) for integration guidance.

## Know when to keep using legacy keys

This adapter is an additive consumer tool, not a package-wide migration. Keep consuming
`rts` and `choices` in paths that already depend on their legacy semantics. This change
does **not** migrate:

- dataset generation;
- KDE, LAN, CPN, or OPN likelihood/training paths;
- RL trial extraction and panel assembly;
- HSSM's public `simulate_data` path;
- ssms-gui;
- any external consumer that needs the existing keys or squeezed shapes.

Those paths retain their existing defaults and outputs. A consumer can construct the
normalized view alongside the untouched legacy result when it explicitly adopts the
structured contract.
