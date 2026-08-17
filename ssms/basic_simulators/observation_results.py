"""Validation for native fixed-width simulator observation results.

This module defines only the package-native result contract. It does not adapt legacy
``rts``/``choices`` results or attach schemas to registered simulators.
"""

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from typing import Any

import numpy as np

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

    schema = _validate_schema(metadata["observation_schema"])
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


def _validate_schema(schema: object) -> tuple[Mapping[str, Any], ...]:
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
            _validate_categorical_schema(entry, name)
        elif kind == "continuous":
            _validate_continuous_schema(entry, name)
        else:
            _validate_circular_schema(entry, name)

        names.append(name)

    if len(names) != len(set(names)):
        raise ValueError("observation_schema field names must be unique")
    return schema


def _validate_categorical_schema(entry: Mapping[str, Any], name: str) -> None:
    if "values" not in entry:
        raise ValueError(f"categorical field {name!r} requires values")

    values = entry["values"]
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(
            f"categorical field {name!r} values must be a non-empty sequence"
        )
    if not values:
        raise ValueError(f"categorical field {name!r} values must not be empty")

    validated_values: list[float] = []
    for value in values:
        if not _is_finite_real(value) or not float(value).is_integer():
            raise ValueError(
                f"categorical field {name!r} values must be finite, "
                "integer-valued numeric labels"
            )
        validated_values.append(float(value))

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
            valid &= np.isin(values, tuple(entry["values"]))
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
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, Real)
        and math.isfinite(float(value))
    )


def _format_keys(keys: Sequence[str] | set[str] | frozenset[str]) -> str:
    return ", ".join(repr(key) for key in sorted(keys))


__all__ = ["OBSERVATION_SCHEMA_VERSION", "validate_observation_result"]
