"""Validate native observation results and explicitly project legacy results.

This module does not register, wrap, or change simulators.
"""

from collections.abc import Mapping, Sequence
import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from .simulator import OMISSION_SENTINEL

OBSERVATION_SCHEMA_VERSION = 1

_REQUIRED_RESULT_KEYS = frozenset({"observations", "omission_mask", "metadata"})
_RESERVED_METADATA_KEYS = frozenset(
    {"observation_schema_version", "observation_schema"}
)
_COMMON_SCHEMA_KEYS = frozenset({"name", "kind"})
_SCHEMA_KEYS_BY_KIND = {
    "categorical": _COMMON_SCHEMA_KEYS | {"values"},
    "continuous": _COMMON_SCHEMA_KEYS
    | {"lower", "upper", "lower_inclusive", "upper_inclusive"},
    "circular": _COMMON_SCHEMA_KEYS | {"lower", "upper"},
}


def normalize_simulator_result(
    result: Mapping[str, Any],
    *,
    expected_n_samples: int,
    expected_n_trials: int,
    observation_schema: tuple[Mapping[str, Any], ...],
    source_projection: tuple[tuple[str, str], ...],
    omission_source: str,
) -> dict[str, Any]:
    """Project an explicitly described legacy result into the native contract.

    Parameters
    ----------
    result
        Result containing metadata and each projected NumPy array.
    expected_n_samples, expected_n_trials
        Positive counts before historical singleton-axis squeezing.
    observation_schema
        Ordered native schema with one or two fields.
    source_projection
        Ordered ``(legacy key, schema name)`` pairs covering the schema exactly.
    omission_source
        Projected key whose ``OMISSION_SENTINEL`` authoritatively marks omissions.

    Notes
    -----
    Axes and semantics are never inferred. Inputs remain unchanged; the result and
    metadata are shallow-copied. Use :func:`validate_observation_result` directly for
    native results or schemas wider than two fields.
    """
    if not isinstance(result, Mapping):
        raise TypeError("legacy simulator result must be a mapping")

    preexisting_canonical_keys = {"observations", "omission_mask"}.intersection(result)
    if preexisting_canonical_keys:
        raise ValueError(
            "legacy simulator result already contains reserved canonical key(s): "
            f"{_format_keys(preexisting_canonical_keys)}"
        )

    n_samples = _validate_expected_count(expected_n_samples, "expected_n_samples")
    n_trials = _validate_expected_count(expected_n_trials, "expected_n_trials")

    if "metadata" not in result:
        raise ValueError("legacy simulator result is missing required key 'metadata'")
    metadata = result["metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError("legacy simulator result metadata must be a mapping")

    source_keys, omission_source_index = _validate_legacy_projection(
        observation_schema, source_projection, omission_source
    )
    expected_shape = _expected_legacy_shape(n_samples, n_trials)

    projected_values: list[np.ndarray] = []
    for source_key in source_keys:
        if source_key not in result:
            raise ValueError(
                f"legacy simulator result is missing projected source {source_key!r}"
            )
        source = result[source_key]
        if not isinstance(source, np.ndarray):
            raise TypeError(f"projected source {source_key!r} must be a NumPy array")
        if not (
            np.issubdtype(source.dtype, np.integer)
            or np.issubdtype(source.dtype, np.floating)
        ):
            raise TypeError(
                f"projected source {source_key!r} must have a real numeric NumPy dtype"
            )
        if source.shape != expected_shape:
            raise ValueError(
                f"projected source {source_key!r} must have historical legacy shape "
                f"{expected_shape} for expected_n_samples={n_samples} and "
                f"expected_n_trials={n_trials}; got {source.shape}"
            )

        values = source.reshape(n_samples, n_trials)
        projected_values.append(values)

    omission_mask = projected_values[omission_source_index] == OMISSION_SENTINEL
    contradictory_source_keys = tuple(
        source_key
        for source_key, values in zip(source_keys, projected_values, strict=True)
        if source_key != omission_source
        and np.any((values == OMISSION_SENTINEL) & ~omission_mask)
    )
    if contradictory_source_keys:
        raise ValueError(
            "legacy omission sentinel occurs in non-authoritative projected "
            f"source(s) {_format_keys(contradictory_source_keys)} while "
            f"omission_source {omission_source!r} is non-sentinel"
        )

    observation_dtype = _projected_observation_dtype(projected_values)
    _validate_categorical_source_precision(
        observation_schema,
        source_keys,
        projected_values,
        observation_dtype,
        omission_mask,
    )
    observations = np.stack(projected_values, axis=-1).astype(
        observation_dtype, copy=False
    )
    observations[omission_mask] = np.nan

    projected_metadata = dict(metadata)
    for key, value in (
        ("observation_schema_version", OBSERVATION_SCHEMA_VERSION),
        ("observation_schema", observation_schema),
    ):
        _insert_reserved_metadata(projected_metadata, key, value)

    projected_result = dict(result)
    projected_result.update(
        {
            "observations": observations,
            "omission_mask": omission_mask,
            "metadata": projected_metadata,
        }
    )
    return validate_observation_result(projected_result)


def validate_observation_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a native fixed-width simulator observation result.

    Parameters
    ----------
    result
        Mapping with ``observations``, ``omission_mask``, and ``metadata``. The
        observations must be a floating NumPy array with shape
        ``(n_samples, n_trials, obs_dim)``. Metadata must contain the versioned,
        ordered ``observation_schema``.

    Returns
    -------
    dict
        A shallow plain-dictionary view of the validated result. Observation and mask
        arrays are not copied. Metadata is shallow-copied, so producer-owned extension
        values retain identity.

    Raises
    ------
    TypeError
        If the result, arrays, metadata, or schema entries use invalid container or
        dtype types.
    ValueError
        If required fields, schema definitions, shapes, omission encodings, or
        observation values violate the contract.

    Notes
    -----
    Validation is pure: the source mapping, arrays, schema, and metadata values are
    never mutated. Metadata keys other than ``observation_schema_version`` and
    ``observation_schema`` are producer-owned extensions and are not interpreted.
    """
    if not isinstance(result, Mapping):
        raise TypeError("observation result must be a mapping")

    missing_result_keys = _REQUIRED_RESULT_KEYS.difference(result)
    if missing_result_keys:
        raise ValueError(
            "observation result is missing required key(s): "
            f"{_format_keys(missing_result_keys)}"
        )

    observations = result["observations"]
    omission_mask = result["omission_mask"]
    metadata = result["metadata"]

    if not isinstance(observations, np.ndarray):
        raise TypeError("observations must be a NumPy array")
    if observations.ndim != 3:
        raise ValueError(
            "observations must have exactly three axes (n_samples, n_trials, obs_dim)"
        )
    if not np.issubdtype(observations.dtype, np.floating):
        raise TypeError("observations must have a floating NumPy dtype")

    if not isinstance(omission_mask, np.ndarray):
        raise TypeError("omission_mask must be a NumPy array")
    if not np.issubdtype(omission_mask.dtype, np.bool_):
        raise TypeError("omission_mask must have a boolean NumPy dtype")
    if omission_mask.shape != observations.shape[:2]:
        raise ValueError(
            "omission_mask shape must equal the first two observation axes: "
            f"expected {observations.shape[:2]}, got {omission_mask.shape}"
        )

    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    missing_metadata_keys = _RESERVED_METADATA_KEYS.difference(metadata)
    if missing_metadata_keys:
        raise ValueError(
            "metadata is missing required reserved key(s): "
            f"{_format_keys(missing_metadata_keys)}"
        )

    version = metadata["observation_schema_version"]
    if type(version) is not int or version != OBSERVATION_SCHEMA_VERSION:
        raise ValueError(
            "observation_schema_version must be the supported integer version "
            f"{OBSERVATION_SCHEMA_VERSION}; got {version!r}"
        )

    schema = _validate_schema(metadata["observation_schema"], observations.dtype)
    if observations.shape[-1] != len(schema):
        raise ValueError(
            "observations width must equal observation_schema length: "
            f"expected {len(schema)}, got {observations.shape[-1]}"
        )

    _validate_omissions(observations, omission_mask)
    _validate_observation_values(observations, omission_mask, schema)

    validated = dict(result)
    validated["metadata"] = dict(metadata)
    return validated


def _validate_schema(
    schema: object, observation_dtype: np.dtype[Any]
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(schema, tuple):
        raise TypeError("observation_schema must be an ordered tuple of mappings")
    if not schema:
        raise ValueError("observation_schema must contain at least one field")

    names: list[str] = []
    for index, entry in enumerate(schema):
        if not isinstance(entry, Mapping):
            raise TypeError(f"observation_schema entry {index} must be a mapping")

        missing_common = _COMMON_SCHEMA_KEYS.difference(entry)
        if missing_common:
            raise ValueError(
                f"observation_schema entry {index} is missing required key(s): "
                f"{_format_keys(missing_common)}"
            )

        name = entry["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"observation_schema entry {index} name must be a non-empty string"
            )

        kind = entry["kind"]
        if kind not in _SCHEMA_KEYS_BY_KIND:
            raise ValueError(
                f"observation_schema entry {name!r} kind must be one of "
                f"{tuple(_SCHEMA_KEYS_BY_KIND)}; got {kind!r}"
            )

        allowed_keys = _SCHEMA_KEYS_BY_KIND[kind]
        unexpected_keys = set(entry).difference(allowed_keys)
        if unexpected_keys:
            raise ValueError(
                f"observation_schema entry {name!r} has unsupported key(s): "
                f"{_format_keys(unexpected_keys)}"
            )

        if kind == "categorical":
            _validate_categorical_schema(entry, name, observation_dtype)
        elif kind == "continuous":
            _validate_continuous_schema(entry, name)
        else:
            _validate_circular_schema(entry, name)

        names.append(name)

    if len(names) != len(set(names)):
        raise ValueError("observation_schema field names must be unique")
    return schema


def _validate_categorical_schema(
    entry: Mapping[str, Any], name: str, observation_dtype: np.dtype[Any]
) -> None:
    if "values" not in entry:
        raise ValueError(f"categorical field {name!r} requires values")

    values = entry["values"]
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(
            f"categorical field {name!r} values must be a non-empty sequence"
        )
    if not values:
        raise ValueError(f"categorical field {name!r} values must not be empty")

    validated_values: list[int] = []
    for value in values:
        integer_value = _as_finite_integer(value)
        if integer_value is None:
            raise ValueError(
                f"categorical field {name!r} values must be finite, "
                "integer-valued numeric labels"
            )
        if not _integer_is_exactly_representable(integer_value, observation_dtype):
            raise ValueError(
                f"categorical field {name!r} value {value!r} is not exactly "
                f"representable in observations dtype {observation_dtype.name}"
            )
        validated_values.append(integer_value)

    if len(validated_values) != len(set(validated_values)):
        raise ValueError(f"categorical field {name!r} values must be unique")


def _validate_continuous_schema(entry: Mapping[str, Any], name: str) -> None:
    for endpoint in ("lower", "upper"):
        inclusive = f"{endpoint}_inclusive"
        if endpoint not in entry:
            if inclusive in entry:
                raise ValueError(
                    f"continuous field {name!r} cannot define {inclusive} "
                    f"without {endpoint}"
                )
            continue

        if not _is_finite_real(entry[endpoint]):
            raise ValueError(
                f"continuous field {name!r} {endpoint} must be a finite number"
            )
        if inclusive in entry and type(entry[inclusive]) is not bool:
            raise ValueError(f"continuous field {name!r} {inclusive} must be boolean")

    if (
        "lower" in entry
        and "upper" in entry
        and float(entry["lower"]) >= float(entry["upper"])
    ):
        raise ValueError(f"continuous field {name!r} lower must be less than upper")


def _validate_circular_schema(entry: Mapping[str, Any], name: str) -> None:
    missing = {"lower", "upper"}.difference(entry)
    if missing:
        raise ValueError(
            f"circular field {name!r} is missing required key(s): "
            f"{_format_keys(missing)}"
        )
    if not _is_finite_real(entry["lower"]) or not _is_finite_real(entry["upper"]):
        raise ValueError(f"circular field {name!r} bounds must be finite numbers")
    if float(entry["lower"]) >= float(entry["upper"]):
        raise ValueError(f"circular field {name!r} lower must be less than upper")


def _validate_omissions(observations: np.ndarray, omission_mask: np.ndarray) -> None:
    nan_components = np.isnan(observations)
    any_nan = np.any(nan_components, axis=-1)
    all_nan = np.all(nan_components, axis=-1)
    if np.any(any_nan & ~all_nan):
        raise ValueError(
            "partial-NaN observation rows are invalid; omissions must cover every field"
        )
    if not np.array_equal(omission_mask, all_nan):
        raise ValueError(
            "omission_mask must exactly equal the rows whose observation fields are "
            "all NaN"
        )

    available = observations[~omission_mask]
    if not np.all(np.isfinite(available)):
        raise ValueError("non-omitted observation values must be finite")


def _validate_observation_values(
    observations: np.ndarray,
    omission_mask: np.ndarray,
    schema: tuple[Mapping[str, Any], ...],
) -> None:
    for index, entry in enumerate(schema):
        values = observations[..., index][~omission_mask]
        if values.size == 0:
            continue

        name = entry["name"]
        kind = entry["kind"]
        valid = np.ones(values.shape, dtype=bool)

        if kind == "categorical":
            categorical_values = np.asarray(entry["values"], dtype=observations.dtype)
            valid &= np.isin(values, categorical_values)
        elif kind == "continuous":
            if "lower" in entry:
                if entry.get("lower_inclusive", True):
                    valid &= values >= entry["lower"]
                else:
                    valid &= values > entry["lower"]
            if "upper" in entry:
                if entry.get("upper_inclusive", True):
                    valid &= values <= entry["upper"]
                else:
                    valid &= values < entry["upper"]
        else:
            valid &= values >= entry["lower"]
            valid &= values < entry["upper"]

        if not np.all(valid):
            invalid_values = np.unique(values[~valid])
            raise ValueError(
                f"observation field {name!r} contains value(s) outside its "
                f"{kind} domain: {invalid_values.tolist()}"
            )


def _is_finite_real(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _as_finite_integer(value: object) -> int | None:
    """Return the exact integer represented by a valid categorical label."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    if isinstance(value, Integral):
        return int(value)

    as_integer_ratio = getattr(value, "as_integer_ratio", None)
    if as_integer_ratio is not None:
        try:
            numerator, denominator = as_integer_ratio()
        except (OverflowError, TypeError, ValueError):
            return None
        if denominator != 1:
            return None
        return int(numerator)

    try:
        as_float = float(value)
        if not math.isfinite(as_float) or not as_float.is_integer():
            return None
        integer_value = int(as_float)
    except (OverflowError, TypeError, ValueError):
        return None
    return integer_value if value == integer_value else None


def _integer_is_exactly_representable(
    value: int, observation_dtype: np.dtype[Any]
) -> bool:
    """Return whether ``value`` survives an exact round trip through ``dtype``."""
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            cast_value = np.asarray(value, dtype=observation_dtype)[()]
    except (OverflowError, TypeError, ValueError):
        return False
    return bool(np.isfinite(cast_value)) and int(cast_value) == value


def _validate_expected_count(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    count = int(value)
    if count <= 0:
        raise ValueError(f"{name} must be positive")
    return count


def _validate_legacy_projection(
    observation_schema: object,
    source_projection: object,
    omission_source: object,
) -> tuple[tuple[str, ...], int]:
    if not isinstance(observation_schema, tuple):
        raise TypeError("observation_schema must be an ordered tuple of mappings")
    if not 1 <= len(observation_schema) <= 2:
        raise ValueError(
            "legacy result projection supports schemas with one or two fields"
        )

    names: list[str] = []
    for index, entry in enumerate(observation_schema):
        if not isinstance(entry, Mapping):
            raise TypeError(f"observation_schema entry {index} must be a mapping")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"observation_schema entry {index} name must be a non-empty string"
            )
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("observation_schema field names must be unique")

    if not isinstance(source_projection, tuple):
        raise TypeError("source_projection must be an ordered tuple of pairs")
    if len(source_projection) != len(names):
        raise ValueError("source_projection must cover the schema exactly")

    source_keys: list[str] = []
    target_names: list[str] = []
    for index, pair in enumerate(source_projection):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(f"source_projection entry {index} must be a pair")
        source_key, target_name = pair
        if not (
            isinstance(source_key, str)
            and source_key.strip()
            and isinstance(target_name, str)
            and target_name.strip()
        ):
            raise TypeError(
                f"source_projection entry {index} must contain non-empty strings"
            )
        source_keys.append(source_key)
        target_names.append(target_name)

    if len(source_keys) != len(set(source_keys)):
        raise ValueError("source_projection source keys must be unique")
    if target_names != names:
        raise ValueError(
            "source_projection target names must match every field in schema order"
        )

    if not isinstance(omission_source, str) or not omission_source.strip():
        raise TypeError(
            "omission_source must be a non-empty string naming a projected "
            "legacy source key"
        )
    try:
        omission_source_index = source_keys.index(omission_source)
    except ValueError as error:
        raise ValueError(
            "omission_source must name one projected legacy source key; "
            f"got {omission_source!r}"
        ) from error
    return tuple(source_keys), omission_source_index


def _expected_legacy_shape(n_samples: int, n_trials: int) -> tuple[int, ...]:
    if n_trials == 1:
        return (n_samples, 1)
    if n_samples == 1:
        return (n_trials, 1)
    return (n_samples, n_trials, 1)


def _projected_observation_dtype(sources: Sequence[np.ndarray]) -> np.dtype[Any]:
    promoted = np.result_type(*(source.dtype for source in sources))
    return np.dtype(promoted if np.issubdtype(promoted, np.floating) else np.float64)


def _validate_categorical_source_precision(
    observation_schema: tuple[Mapping[str, Any], ...],
    source_keys: tuple[str, ...],
    projected_values: Sequence[np.ndarray],
    observation_dtype: np.dtype[Any],
    omission_mask: np.ndarray,
) -> None:
    for schema_entry, source_key, values in zip(
        observation_schema, source_keys, projected_values, strict=True
    ):
        if schema_entry.get("kind") != "categorical" or not np.issubdtype(
            values.dtype, np.integer
        ):
            continue

        with np.errstate(over="ignore", invalid="ignore"):
            round_trip = values.astype(observation_dtype).astype(values.dtype)
        if not np.array_equal(round_trip[~omission_mask], values[~omission_mask]):
            raise ValueError(
                f"categorical projected source {source_key!r} contains value(s) not "
                f"exactly representable in observations dtype {observation_dtype.name}"
            )


def _insert_reserved_metadata(
    metadata: dict[str, Any],
    key: str,
    value: object,
) -> None:
    if key not in metadata:
        metadata[key] = value
        return
    existing = metadata[key]
    if existing is value:
        return
    try:
        equal = existing == value
    except (TypeError, ValueError):
        equal = False
    if not isinstance(equal, (bool, np.bool_)) or not bool(equal):
        raise ValueError(f"legacy metadata contains conflicting reserved key {key!r}")


def _format_keys(keys: Sequence[str] | set[str] | frozenset[str]) -> str:
    return ", ".join(repr(key) for key in sorted(keys))


__all__ = [
    "OBSERVATION_SCHEMA_VERSION",
    "normalize_simulator_result",
    "validate_observation_result",
]
