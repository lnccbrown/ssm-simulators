# Structured observation result contract

This page specifies observation schema version 1 and the behavior of
`validate_observation_result`. It is the normative package contract for native
fixed-width observation results.

## Required result fields

A native result is a mapping with three required keys:

| Key | Required value |
| --- | --- |
| `observations` | floating NumPy array with shape `(n_samples, n_trials, obs_dim)` |
| `omission_mask` | boolean NumPy array with shape `(n_samples, n_trials)` |
| `metadata` | mapping containing the two reserved schema keys |

The three observation axes are never squeezed. `obs_dim` must equal the number of ordered
schema entries and must be at least one. Integer, complex, object, and structured
observation dtypes are rejected; a floating dtype permits both numeric categorical labels
and NaN omission rows.

## Reserved metadata

`metadata` must contain:

```python
{
    "observation_schema_version": 1,
    "observation_schema": (
        # ordered schema entries
    ),
}
```

These are the only reserved metadata keys. Every other metadata key is producer-owned,
accepted without interpretation, and retained in the returned shallow mapping. This
includes fields such as `simulator`, `possible_choices`, parameter/configuration values,
`max_t`, boundary information, and trajectories.

The validator never inserts or overwrites reserved metadata. A future adapter that is
given schema arguments must reject a conflicting pre-existing reserved value rather than
silently replacing producer data.

Unknown schema versions fail explicitly.

## Version 1 schema entries

`observation_schema` is a non-empty tuple of mappings. Its order defines the final
observation axis. Every entry has a unique, non-empty string `name`, a supported `kind`,
and no keys beyond those allowed for that kind.

| Kind | Required keys | Optional keys | Support |
| --- | --- | --- | --- |
| `categorical` | `name`, `kind`, `values` | none | exact membership in finite, unique, integer-valued numeric labels |
| `continuous` | `name`, `kind` | `lower`, `upper`, `lower_inclusive`, `upper_inclusive` | unbounded or one/two-sided scalar interval |
| `circular` | `name`, `kind`, `lower`, `upper` | none | lower-inclusive, upper-exclusive interval `[lower, upper)` |

Categorical booleans, strings, non-integral labels, infinities, and duplicate labels are
invalid.

Continuous endpoints must be finite when present. The corresponding inclusion flag is
allowed only when its endpoint is present, must be boolean, and defaults to `True`.
When both endpoints exist, `lower < upper`.

Circular bounds must be finite and satisfy `lower < upper`. Their endpoint semantics are
fixed; configurable inclusion flags are not part of version 1.

Field names carry no hidden semantics. A field named `rt` is not required, moved, or
validated differently from another positive continuous field.

## Observation values

Every non-omitted value must be finite and valid for its schema entry:

- categorical values use exact numeric membership;
- continuous values respect each declared endpoint and its inclusion flag;
- circular values satisfy `lower <= value < upper`.

The validator does not coerce, clip, wrap, reorder, or otherwise repair producer output.
Schema order—not field-name convention—determines which domain applies to each column.

## Complete-row omissions

`omission_mask` is authoritative and must agree exactly with the observation array:

- `False`: every field in the row is finite and domain-valid;
- `True`: every field in the row is NaN;
- a partially NaN row is invalid;
- an all-NaN row with a false mask, or a finite row with a true mask, is invalid.

The mask is a sidecar, not an observation field. This contract does not enable fitting
missing observations or define a likelihood-level missing-data policy.

## Validation result and purity

```python
from ssms.basic_simulators import validate_observation_result

validated = validate_observation_result(result)
```

The function returns a plain dictionary. It shallow-copies the top-level result and its
metadata mapping, while retaining the original observation array, mask array, schema, and
producer-extension values. It never mutates the source.

## Executable examples

### RT plus bounded confidence

```python
--8<-- "docs/snippets/observation_results/rt_confidence.py"
```

### Response-only data with a complete omission

```python
--8<-- "docs/snippets/observation_results/response_only_omission.py"
```

The documentation test suite executes these exact files, including their assertions.

## Compatibility boundary

This validator does not adapt legacy `rts`/`choices`, attach schemas to registered
simulators, modify `Simulator.simulate`, or migrate dataset, KDE, LAN, RLSSM, HSSM, or GUI
consumers. Those changes require separate compatibility work. Existing simulator outputs
remain unchanged by version 1 validation support.
