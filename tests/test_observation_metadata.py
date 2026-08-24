"""Tests for explicit producer observation metadata."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest


RT = {
    "name": "rt",
    "kind": "continuous",
    "lower": 0.0,
    "lower_inclusive": False,
}
RESPONSE = {"name": "response", "kind": "categorical", "values": (-1, 1)}
CONFIDENCE = {
    "name": "confidence",
    "kind": "continuous",
    "lower": 0.0,
    "upper": 1.0,
}
ANGLE = {
    "name": "angle",
    "kind": "circular",
    "lower": -np.pi,
    "upper": np.pi,
}


def _validate(metadata: object) -> dict[str, Any]:
    from ssms.basic_simulators import validate_observation_metadata

    return validate_observation_metadata(metadata)


def _get(producer: object) -> dict[str, Any]:
    from ssms.basic_simulators import get_observation_metadata

    return get_observation_metadata(producer)


def _explicit(
    schema: tuple[Mapping[str, Any], ...], *, obs_dim: object | None = None
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "observation_schema_version": 1,
        "observation_schema": schema,
    }
    if obs_dim is not None:
        metadata["obs_dim"] = obs_dim
    return metadata


def test_validate_observation_metadata_expands_explicit_legacy_profile() -> None:
    source = MappingProxyType(
        {
            "observation_schema_version": 1,
            "observation_schema_profile": "legacy_rt_choice",
            "choices": [-1, 1],
            "obs_dim": 2,
            "producer_extension": object(),
        }
    )

    descriptor = _validate(source)

    assert descriptor == {
        "observation_schema_version": 1,
        "observation_schema": (
            RT,
            {"name": "response", "kind": "categorical", "values": (-1, 1)},
        ),
        "obs_dim": 2,
    }
    assert tuple(descriptor) == (
        "observation_schema_version",
        "observation_schema",
        "obs_dim",
    )


@pytest.mark.parametrize(
    "schema",
    [
        (RESPONSE,),
        ({"name": "latency", "kind": "continuous"},),
        (RT, CONFIDENCE, RESPONSE),
        (RT, CONFIDENCE, ANGLE, RESPONSE),
    ],
    ids=("response-only", "continuous-only", "mixed-three", "mixed-four"),
)
def test_validate_observation_metadata_accepts_explicit_generic_schemas(
    schema: tuple[Mapping[str, Any], ...],
) -> None:
    descriptor = _validate(_explicit(schema, obs_dim=len(schema)))

    assert descriptor["obs_dim"] == len(schema)
    assert tuple(field["name"] for field in descriptor["observation_schema"]) == tuple(
        field["name"] for field in schema
    )
    assert all(type(field) is dict for field in descriptor["observation_schema"])


@pytest.mark.parametrize("version", [0, 2, True, "1"])
def test_validate_observation_metadata_rejects_unsupported_versions(
    version: object,
) -> None:
    metadata = _explicit((RT,))
    metadata["observation_schema_version"] = version

    with pytest.raises(ValueError, match="observation_schema_version"):
        _validate(metadata)


@pytest.mark.parametrize(
    ("metadata", "error", "message"),
    [
        ([], TypeError, "mapping"),
        ({"observation_schema": (RT,)}, ValueError, "schema_version"),
        (
            {
                "observation_schema_version": 1,
                "observation_schema": (RT,),
                "observation_schema_profile": "legacy_rt_choice",
                "choices": (-1, 1),
            },
            ValueError,
            "exactly one",
        ),
        ({"observation_schema_version": 1}, ValueError, "exactly one"),
        (
            {"observation_schema_version": 1, "observation_schema": [RT]},
            TypeError,
            "ordered tuple",
        ),
        (_explicit((RT,), obs_dim=True), TypeError, "obs_dim"),
        (_explicit((RT,), obs_dim=0), ValueError, "obs_dim"),
        (_explicit((RT,), obs_dim=2), ValueError, "obs_dim"),
    ],
)
def test_validate_observation_metadata_rejects_malformed_descriptors(
    metadata: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _validate(metadata)


@pytest.mark.parametrize(
    ("profile", "choices", "error", "message"),
    [
        ("unknown", (-1, 1), ValueError, "profile"),
        ("legacy_rt_choice", None, ValueError, "choices"),
        ("legacy_rt_choice", (), ValueError, "values"),
        ("legacy_rt_choice", (0.5, 1), ValueError, "values"),
        ("legacy_rt_choice", (1, 1), ValueError, "unique"),
        ("legacy_rt_choice", "01", TypeError, "sequence"),
        (
            "legacy_rt_choice",
            np.zeros((1, 2), dtype=int),
            TypeError,
            "one-dimensional",
        ),
    ],
)
def test_validate_observation_metadata_rejects_malformed_profiles(
    profile: str,
    choices: object,
    error: type[Exception],
    message: str,
) -> None:
    metadata: dict[str, Any] = {
        "observation_schema_version": 1,
        "observation_schema_profile": profile,
    }
    if choices is not None:
        metadata["choices"] = choices

    with pytest.raises(error, match=message):
        _validate(metadata)


@pytest.mark.parametrize(
    "metadata",
    [
        {"observation_schema_version": 1, "choices": (-1, 1), "obs_dim": 2},
        {"observation_schema_version": 1, "nchoices": 2, "obs_dim": 2},
    ],
)
def test_validate_observation_metadata_never_infers_schema(
    metadata: Mapping[str, Any],
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        _validate(metadata)


def test_get_observation_metadata_reads_callable_attributes_without_execution() -> None:
    class Producer:
        observation_schema_version = 1
        observation_schema_profile = "legacy_rt_choice"
        choices = np.asarray([0, 1, 2])
        obs_dim = 99
        calls = 0

        def __call__(self) -> None:
            self.calls += 1
            raise AssertionError("metadata inspection must not execute the producer")

    producer = Producer()

    descriptor = _get(producer)

    assert producer.calls == 0
    assert descriptor["obs_dim"] == 2
    assert descriptor["observation_schema"][1]["values"] == (0, 1, 2)


def test_get_observation_metadata_reads_explicit_schema_callable() -> None:
    calls: list[bool] = []

    def producer() -> None:
        calls.append(True)
        raise AssertionError("metadata inspection must not execute the producer")

    setattr(producer, "observation_schema_version", 1)
    setattr(producer, "observation_schema", (RESPONSE,))

    descriptor = _get(producer)

    assert not calls
    assert descriptor["observation_schema"] == (RESPONSE,)
    assert descriptor["obs_dim"] == 1


def test_get_observation_metadata_requires_explicit_callable_declaration() -> None:
    class LegacyProducer:
        choices = (-1, 1)
        nchoices = 2
        obs_dim = 2
        calls = 0

        def __call__(self) -> None:
            self.calls += 1

    producer = LegacyProducer()

    with pytest.raises(ValueError, match="explicit observation metadata"):
        _get(producer)
    assert producer.calls == 0


def test_get_observation_metadata_resolves_every_registered_model_name() -> None:
    from ssms.config import get_model_registry

    registry = get_model_registry()

    for model_name in registry.list_models():
        assert _get(model_name) == _validate(registry.get(model_name)), model_name


def test_get_observation_metadata_resolves_derived_deadline_models() -> None:
    assert _get("ddm_deadline") == _get("ddm")


def test_get_observation_metadata_reads_live_registry(monkeypatch) -> None:
    from ssms.config import get_model_registry

    def simulator() -> None:
        raise AssertionError("metadata lookup must not execute the simulator")

    registry = get_model_registry()
    model_name = "test_live_observation_metadata"
    config = {
        "name": model_name,
        "simulator": simulator,
        **_explicit((RESPONSE,)),
    }
    monkeypatch.setitem(registry._configs, model_name, config)

    assert _get(model_name)["observation_schema"] == (RESPONSE,)


def test_get_observation_metadata_rejects_unknown_model_names() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        _get("not_a_registered_model")


def test_get_observation_metadata_uses_explicit_provider_hook() -> None:
    class Provider:
        calls = 0

        def get_observation_metadata(self) -> dict[str, Any]:
            self.calls += 1
            return _explicit((RESPONSE,))

    provider = Provider()

    assert _get(provider)["observation_schema"] == (RESPONSE,)
    assert provider.calls == 1


def test_get_observation_metadata_requires_profile_choices_on_callable() -> None:
    class Producer:
        observation_schema_version = 1
        observation_schema_profile = "legacy_rt_choice"

        def __call__(self) -> None:
            raise AssertionError("metadata inspection must not execute the producer")

    with pytest.raises(ValueError, match="requires explicit choices"):
        _get(Producer())


def test_get_observation_metadata_rejects_nonproducer_objects() -> None:
    with pytest.raises(TypeError, match="mapping or callable"):
        _get(object())


def test_observation_metadata_access_is_pure_and_returns_fresh_plain_values() -> None:
    values = [-1, 1]
    field = {"name": "response", "kind": "categorical", "values": values}
    source = _explicit((field,))

    first = _get(source)
    second = _validate(source)

    assert type(first) is dict
    assert first is not second
    assert first["observation_schema"] is not source["observation_schema"]
    assert first["observation_schema"] is not second["observation_schema"]
    assert first["observation_schema"][0] is not field
    assert first["observation_schema"][0] is not second["observation_schema"][0]
    assert first["observation_schema"][0]["values"] is not values
    first["observation_schema"][0]["name"] = "changed"
    assert field["name"] == "response"
    assert second["observation_schema"][0]["name"] == "response"


def test_result_validation_keeps_dtype_specific_categorical_precision_check() -> None:
    from ssms.basic_simulators import validate_observation_result

    label = 2**24 + 1
    result = {
        "observations": np.asarray([[[0.0]]], dtype=np.float32),
        "omission_mask": np.zeros((1, 1), dtype=bool),
        "metadata": _explicit(
            ({"name": "response", "kind": "categorical", "values": (label,)},)
        ),
    }
    descriptor = _validate(result["metadata"])

    assert descriptor["observation_schema"][0]["values"] == (label,)

    with pytest.raises(ValueError, match="not exactly representable.*float32"):
        validate_observation_result(result)
