"""Validate and inspect explicit producer observation metadata."""

from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np

from .observation_results import (
    OBSERVATION_SCHEMA_VERSION,
    _validate_schema,
    _validate_schema_version,
)

_SCHEMA_ATTRIBUTE_NAMES = (
    "observation_schema_version",
    "observation_schema",
    "observation_schema_profile",
)
_LEGACY_RT_CHOICE_PROFILE = "legacy_rt_choice"


def validate_observation_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an explicit producer descriptor and return its canonical view.

    A descriptor must declare schema version 1 and exactly one of an ordered
    ``observation_schema`` or the named ``legacy_rt_choice`` profile. The profile
    additionally requires explicit ``choices``. An optional ``obs_dim`` is checked
    for consistency, never used to infer a schema.

    The returned plain dictionary contains only the version, a freshly copied schema,
    and the width derived from that schema. Input mappings are not mutated.
    """
    if not isinstance(metadata, Mapping):
        raise TypeError("observation metadata must be a mapping")
    if "observation_schema_version" not in metadata:
        raise ValueError("observation metadata is missing observation_schema_version")

    _validate_schema_version(metadata["observation_schema_version"])
    schema = _resolve_schema(metadata)
    validated_schema = _validate_schema(schema)
    copied_schema = tuple(_copy_schema_entry(entry) for entry in validated_schema)
    obs_dim = len(copied_schema)
    if "obs_dim" in metadata:
        _validate_declared_obs_dim(metadata["obs_dim"], obs_dim)

    return {
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_schema": copied_schema,
        "obs_dim": obs_dim,
    }


def get_observation_metadata(
    producer: Mapping[str, Any] | Callable[..., Any],
) -> dict[str, Any]:
    """Return explicit observation metadata for a mapping or callable producer.

    Callable producers are inspected through the explicit schema attributes only and
    are never executed. Their legacy ``obs_dim`` attribute is deliberately ignored:
    semantic width always comes from the declared schema.
    """
    if isinstance(producer, Mapping):
        return validate_observation_metadata(producer)
    if not callable(producer):
        raise TypeError("producer must be an observation metadata mapping or callable")

    metadata = _callable_metadata(producer)
    if not {
        "observation_schema",
        "observation_schema_profile",
    }.intersection(metadata):
        raise ValueError(
            "callable must declare explicit observation metadata through "
            "observation_schema or observation_schema_profile"
        )
    return validate_observation_metadata(metadata)


def _resolve_schema(metadata: Mapping[str, Any]) -> object:
    has_schema = "observation_schema" in metadata
    has_profile = "observation_schema_profile" in metadata
    if has_schema == has_profile:
        raise ValueError(
            "observation metadata must define exactly one of observation_schema "
            "or observation_schema_profile"
        )
    if has_schema:
        return metadata["observation_schema"]

    profile = metadata["observation_schema_profile"]
    if profile != _LEGACY_RT_CHOICE_PROFILE:
        raise ValueError(
            "observation_schema_profile must be the supported profile "
            f"{_LEGACY_RT_CHOICE_PROFILE!r}; got {profile!r}"
        )
    if "choices" not in metadata:
        raise ValueError("legacy_rt_choice profile requires explicit choices")

    choices = _profile_choices(metadata["choices"])
    return (
        {
            "name": "rt",
            "kind": "continuous",
            "lower": 0.0,
            "lower_inclusive": False,
        },
        {"name": "response", "kind": "categorical", "values": choices},
    )


def _profile_choices(choices: object) -> tuple[object, ...]:
    if isinstance(choices, np.ndarray):
        if choices.ndim != 1:
            raise TypeError(
                "legacy_rt_choice choices must be a one-dimensional sequence"
            )
        return tuple(choices.tolist())
    if isinstance(choices, (str, bytes)) or not isinstance(choices, Sequence):
        raise TypeError("legacy_rt_choice choices must be a non-empty sequence")
    return tuple(choices)


def _copy_schema_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(entry)
    if entry["kind"] == "categorical":
        copied["values"] = tuple(entry["values"])
    return copied


def _validate_declared_obs_dim(value: object, derived_obs_dim: int) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("obs_dim must be an integer when supplied")
    if int(value) != derived_obs_dim:
        raise ValueError(
            "obs_dim must equal the declared observation schema length: "
            f"expected {derived_obs_dim}, got {value!r}"
        )


def _callable_metadata(producer: object) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for name in _SCHEMA_ATTRIBUTE_NAMES:
        try:
            metadata[name] = getattr(producer, name)
        except AttributeError:
            continue
    if "observation_schema_profile" in metadata:
        try:
            metadata["choices"] = getattr(producer, "choices")
        except AttributeError:
            pass
    return metadata


__all__ = ["get_observation_metadata", "validate_observation_metadata"]
