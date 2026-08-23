"""Contract tests for native structured-observation simulator results."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

Schema = tuple[Mapping[str, Any], ...]


def _validate(result: Any) -> dict[str, Any]:
    from ssms.basic_simulators import validate_observation_result

    return validate_observation_result(result)


def _result(
    schema: Schema,
    observations: Sequence[Sequence[Sequence[float]]],
    *,
    dtype: npt.DTypeLike = np.float64,
    omission_mask: Sequence[Sequence[bool]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    observation_array = np.asarray(observations, dtype=dtype)
    if omission_mask is None:
        omission_array = np.zeros(observation_array.shape[:2], dtype=bool)
    else:
        omission_array = np.asarray(omission_mask, dtype=bool)

    result_metadata = dict(metadata or {})
    result_metadata.update(
        {
            "observation_schema_version": 1,
            "observation_schema": schema,
        }
    )
    return {
        "observations": observation_array,
        "omission_mask": omission_array,
        "metadata": result_metadata,
    }


RT = {
    "name": "rt",
    "kind": "continuous",
    "lower": 0.0,
    "lower_inclusive": False,
}
CHOICE = {"name": "response", "kind": "categorical", "values": (-1, 1)}
CONFIDENCE = {
    "name": "confidence",
    "kind": "continuous",
    "lower": 0.0,
    "upper": 1.0,
}
AZIMUTH = {
    "name": "azimuth",
    "kind": "circular",
    "lower": -np.pi,
    "upper": np.pi,
}


@pytest.mark.parametrize(
    ("schema", "observations"),
    [
        ((CHOICE,), [[[1.0], [-1.0]]]),
        ((RT, CHOICE), [[[0.3, -1.0], [0.8, 1.0]]]),
        (
            (
                {"name": "latency", "kind": "continuous"},
                {"name": "force", "kind": "continuous", "lower": 0.0},
                CONFIDENCE,
            ),
            [[[1.2, 3.4, 0.25]]],
        ),
        (
            (
                RT,
                {"name": "polar", "kind": "continuous", "lower": 0.0, "upper": np.pi},
                AZIMUTH,
                {"name": "endpoint", "kind": "continuous"},
            ),
            [[[0.4, 1.2, -2.2, 7.5]]],
        ),
    ],
    ids=(
        "response-only-categorical",
        "rt-and-categorical",
        "three-continuous-measurements",
        "four-mixed-observations",
    ),
)
def test_validate_observation_result_accepts_generic_one_to_four_field_schemas(
    schema: Schema,
    observations: Sequence[Sequence[Sequence[float]]],
) -> None:
    result = _result(schema, observations)

    validated = _validate(result)

    assert validated["observations"] is result["observations"]
    assert validated["omission_mask"] is result["omission_mask"]
    assert tuple(entry["name"] for entry in schema) == tuple(
        entry["name"] for entry in validated["metadata"]["observation_schema"]
    )
    assert validated["observations"].shape[-1] == len(schema)


@pytest.mark.parametrize(
    "schema",
    [
        (RT, {"name": "response", "kind": "categorical", "values": (0, 1, 2, 3)}),
        ({"name": "response", "kind": "categorical", "values": (0, 1, 2)},),
    ],
    ids=("rt-based-rlssm", "choice-only-rlssm"),
)
def test_validate_observation_result_accepts_existing_rlssm_observation_shapes(
    schema: Schema,
) -> None:
    observations = [[[0.5, 2.0]]] if len(schema) == 2 else [[[2.0]]]

    validated = _validate(_result(schema, observations))

    assert tuple(
        item["name"] for item in validated["metadata"]["observation_schema"]
    ) == tuple(item["name"] for item in schema)


@pytest.mark.parametrize(
    "missing_key",
    ["observation_schema_version", "observation_schema"],
)
def test_validate_observation_result_requires_reserved_metadata(
    missing_key: str,
) -> None:
    result = _result((RT,), [[[0.5]]])
    del result["metadata"][missing_key]

    with pytest.raises(ValueError, match=missing_key):
        _validate(result)


@pytest.mark.parametrize("version", [0, 2, True, "1"])
def test_validate_observation_result_rejects_unknown_or_non_integer_versions(
    version: object,
) -> None:
    result = _result((RT,), [[[0.5]]])
    result["metadata"]["observation_schema_version"] = version

    with pytest.raises(ValueError, match="observation_schema_version"):
        _validate(result)


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        ((), "at least one"),
        (({"name": "", "kind": "continuous"},), "name"),
        (
            (
                {"name": "same", "kind": "continuous"},
                {"name": "same", "kind": "continuous"},
            ),
            "unique",
        ),
        (({"name": "x"},), "kind"),
        (({"name": "x", "kind": "unknown"},), "kind"),
        (({"name": "x", "kind": "continuous", "typo": 1.0},), "typo"),
        (({"name": "x", "kind": "categorical"},), "values"),
        (({"name": "x", "kind": "circular", "lower": 0.0},), "upper"),
    ],
)
def test_validate_observation_result_rejects_invalid_schema_entries(
    schema: Schema,
    message: str,
) -> None:
    observations = np.zeros((1, 1, max(len(schema), 1)), dtype=np.float64)
    result = {
        "observations": observations,
        "omission_mask": np.zeros((1, 1), dtype=bool),
        "metadata": {
            "observation_schema_version": 1,
            "observation_schema": schema,
        },
    }

    with pytest.raises(ValueError, match=message):
        _validate(result)


@pytest.mark.parametrize(
    "entry",
    [
        {"name": "x", "kind": "continuous", "lower": np.inf},
        {"name": "x", "kind": "continuous", "lower_inclusive": False},
        {
            "name": "x",
            "kind": "continuous",
            "lower": 1.0,
            "upper": 1.0,
        },
        {
            "name": "x",
            "kind": "continuous",
            "lower": 0.0,
            "lower_inclusive": 1,
        },
    ],
)
def test_validate_observation_result_rejects_invalid_continuous_domains(
    entry: Mapping[str, Any],
) -> None:
    with pytest.raises(ValueError, match="continuous|lower|upper|inclusive"):
        _validate(_result((entry,), [[[0.5]]]))


@pytest.mark.parametrize(
    ("entry", "value", "valid"),
    [
        ({"name": "x", "kind": "continuous", "lower": 0.0}, 0.0, True),
        (
            {
                "name": "x",
                "kind": "continuous",
                "lower": 0.0,
                "lower_inclusive": False,
            },
            0.0,
            False,
        ),
        ({"name": "x", "kind": "continuous", "upper": 1.0}, 1.0, True),
        (
            {
                "name": "x",
                "kind": "continuous",
                "upper": 1.0,
                "upper_inclusive": False,
            },
            1.0,
            False,
        ),
    ],
)
def test_validate_observation_result_enforces_continuous_endpoint_inclusion(
    entry: Mapping[str, Any],
    value: float,
    valid: bool,
) -> None:
    result = _result((entry,), [[[value]]])

    if valid:
        assert _validate(result)["observations"] is result["observations"]
    else:
        with pytest.raises(ValueError, match="x"):
            _validate(result)


@pytest.mark.parametrize(
    "values",
    [(), (True, False), (0.5, 1.0), (0.0, np.inf), (0, 0)],
)
def test_validate_observation_result_rejects_invalid_categorical_labels(
    values: Sequence[object],
) -> None:
    entry = {"name": "choice", "kind": "categorical", "values": values}

    with pytest.raises(ValueError, match="values"):
        _validate(_result((entry,), [[[0.0]]]))


@pytest.mark.parametrize(
    ("dtype", "label"),
    [
        (np.float16, 2**11 + 1),
        (np.float32, 2**24 + 1),
        (np.float64, 2**53 + 1),
    ],
    ids=("float16", "float32", "float64"),
)
@pytest.mark.parametrize("batch", ["non-omitted", "all-omitted", "empty"])
def test_validate_observation_result_rejects_categorical_labels_not_exactly_representable(
    dtype: npt.DTypeLike,
    label: int,
    batch: str,
) -> None:
    if batch == "non-omitted":
        observations = np.asarray([[[label]]], dtype=dtype)
        omission_mask = np.zeros((1, 1), dtype=bool)
    elif batch == "all-omitted":
        observations = np.full((1, 1, 1), np.nan, dtype=dtype)
        omission_mask = np.ones((1, 1), dtype=bool)
    else:
        observations = np.empty((0, 1, 1), dtype=dtype)
        omission_mask = np.empty((0, 1), dtype=bool)

    result = {
        "observations": observations,
        "omission_mask": omission_mask,
        "metadata": {
            "observation_schema_version": 1,
            "observation_schema": (
                {"name": "choice", "kind": "categorical", "values": (label,)},
            ),
        },
    }

    with pytest.raises(
        ValueError,
        match=rf"choice.*not exactly representable.*{np.dtype(dtype).name}",
    ):
        _validate(result)


@pytest.mark.parametrize("label", [10**1000, -(10**1000)], ids=("positive", "negative"))
def test_validate_observation_result_rejects_categorical_labels_too_large_for_dtype(
    label: int,
) -> None:
    entry = {"name": "choice", "kind": "categorical", "values": (label,)}

    with pytest.raises(ValueError, match="not exactly representable.*float64"):
        _validate(_result((entry,), [[[0.0]]]))


@pytest.mark.parametrize(
    ("dtype", "label"),
    [
        (np.float16, 2**11 + 2),
        (np.float32, 2**24 + 2),
        (np.float64, 2**53 + 2),
    ],
    ids=("float16", "float32", "float64"),
)
def test_validate_observation_result_accepts_large_exactly_representable_categorical_labels(
    dtype: npt.DTypeLike,
    label: int,
) -> None:
    entry = {"name": "choice", "kind": "categorical", "values": (-label, label)}
    result = _result((entry,), [[[-label], [label]]], dtype=dtype)

    validated = _validate(result)

    np.testing.assert_array_equal(
        validated["observations"], np.asarray([[[-label], [label]]], dtype=dtype)
    )


def test_validate_observation_result_enforces_categorical_membership() -> None:
    with pytest.raises(ValueError, match="choice"):
        _validate(
            _result(
                ({"name": "choice", "kind": "categorical", "values": (-1, 1)},),
                [[[0.0]]],
            )
        )


@pytest.mark.parametrize("value", [-np.pi, 0.0, np.nextafter(np.pi, -np.inf)])
def test_validate_observation_result_accepts_circular_half_open_support(
    value: float,
) -> None:
    assert _validate(_result((AZIMUTH,), [[[value]]]))["observations"].shape == (
        1,
        1,
        1,
    )


@pytest.mark.parametrize("value", [np.pi, -np.pi - 0.1])
def test_validate_observation_result_rejects_values_outside_circular_support(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="azimuth"):
        _validate(_result((AZIMUTH,), [[[value]]]))


@pytest.mark.parametrize(
    ("observations", "mask", "message"),
    [
        ([[[np.nan, 1.0]]], [[True]], "partial"),
        ([[[np.nan, np.nan]]], [[False]], "omission_mask"),
        ([[[0.5, 1.0]]], [[True]], "omission_mask"),
        ([[[np.inf, 1.0]]], [[False]], "finite"),
    ],
)
def test_validate_observation_result_requires_exact_complete_row_omissions(
    observations: Sequence[Sequence[Sequence[float]]],
    mask: Sequence[Sequence[bool]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate(_result((RT, CHOICE), observations, omission_mask=mask))


def test_validate_observation_result_accepts_complete_row_omissions() -> None:
    result = _result(
        (RT, CHOICE),
        [[[0.4, 1.0], [np.nan, np.nan]]],
        omission_mask=[[False, True]],
    )

    validated = _validate(result)

    np.testing.assert_array_equal(validated["omission_mask"], [[False, True]])
    assert np.isnan(validated["observations"][0, 1]).all()


@pytest.mark.parametrize(
    ("observations", "mask", "message"),
    [
        (np.zeros((2, 1), dtype=np.float64), np.zeros((2,), dtype=bool), "three"),
        (np.zeros((1, 1, 1), dtype=np.int64), np.zeros((1, 1), dtype=bool), "floating"),
        (
            np.zeros((1, 1, 1), dtype=np.float64),
            np.zeros((1, 1), dtype=np.int64),
            "boolean",
        ),
        (np.zeros((1, 2, 1), dtype=np.float64), np.zeros((2, 1), dtype=bool), "shape"),
    ],
)
def test_validate_observation_result_rejects_invalid_array_contracts(
    observations: np.ndarray,
    mask: np.ndarray,
    message: str,
) -> None:
    result = {
        "observations": observations,
        "omission_mask": mask,
        "metadata": {
            "observation_schema_version": 1,
            "observation_schema": (RT,),
        },
    }

    with pytest.raises((TypeError, ValueError), match=message):
        _validate(result)


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ([], "mapping"),
        ({"metadata": {}}, "missing required"),
        (
            {
                "observations": [[[0.5]]],
                "omission_mask": np.zeros((1, 1), dtype=bool),
                "metadata": {
                    "observation_schema_version": 1,
                    "observation_schema": (RT,),
                },
            },
            "NumPy array",
        ),
        (
            {
                "observations": np.asarray([[[0.5]]]),
                "omission_mask": [[False]],
                "metadata": {
                    "observation_schema_version": 1,
                    "observation_schema": (RT,),
                },
            },
            "NumPy array",
        ),
        (
            {
                "observations": np.asarray([[[0.5]]]),
                "omission_mask": np.zeros((1, 1), dtype=bool),
                "metadata": [],
            },
            "metadata must be a mapping",
        ),
        (
            {
                "observations": np.asarray([[[0.5]]]),
                "omission_mask": np.zeros((1, 1), dtype=bool),
                "metadata": {
                    "observation_schema_version": 1,
                    "observation_schema": [RT],
                },
            },
            "ordered tuple",
        ),
        (
            {
                "observations": np.asarray([[[0.5]]]),
                "omission_mask": np.zeros((1, 1), dtype=bool),
                "metadata": {
                    "observation_schema_version": 1,
                    "observation_schema": ("rt",),
                },
            },
            "entry 0 must be a mapping",
        ),
        (
            {
                "observations": np.asarray([[[0.5, 0.6]]]),
                "omission_mask": np.zeros((1, 1), dtype=bool),
                "metadata": {
                    "observation_schema_version": 1,
                    "observation_schema": (RT,),
                },
            },
            "width",
        ),
        (
            {
                "observations": np.asarray([[[0.0]]]),
                "omission_mask": np.zeros((1, 1), dtype=bool),
                "metadata": {
                    "observation_schema_version": 1,
                    "observation_schema": (
                        {"name": "choice", "kind": "categorical", "values": "01"},
                    ),
                },
            },
            "non-empty sequence",
        ),
    ],
)
def test_validate_observation_result_rejects_invalid_container_contracts(
    result: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _validate(result)


@pytest.mark.parametrize(
    "entry",
    [
        {"name": "angle", "kind": "circular", "lower": -np.inf, "upper": np.pi},
        {"name": "angle", "kind": "circular", "lower": 1.0, "upper": 1.0},
    ],
)
def test_validate_observation_result_rejects_invalid_circular_domains(
    entry: Mapping[str, Any],
) -> None:
    with pytest.raises(ValueError, match="circular"):
        _validate(_result((entry,), [[[0.0]]]))


def test_validate_observation_result_accepts_an_all_omitted_batch() -> None:
    result = _result(
        (RT, CHOICE),
        [[[np.nan, np.nan], [np.nan, np.nan]]],
        omission_mask=[[True, True]],
    )

    validated = _validate(result)

    assert validated["omission_mask"].all()


def test_validate_observation_result_preserves_open_metadata_without_mutation() -> None:
    boundary = np.linspace(0.0, 1.0, 1000)
    trajectory = {"x": np.arange(10.0)}
    metadata = {
        "simulator": "future_model",
        "possible_choices": (-1, 1),
        "boundary": boundary,
        "trajectory": trajectory,
    }
    result = _result((RT, CHOICE), [[[0.4, 1.0]]], metadata=metadata)
    source_keys = tuple(result["metadata"])

    validated = _validate(result)

    assert type(validated) is dict
    assert type(validated["metadata"]) is dict
    assert validated is not result
    assert validated["metadata"] is not result["metadata"]
    assert tuple(result["metadata"]) == source_keys
    assert validated["metadata"]["boundary"] is boundary
    assert validated["metadata"]["trajectory"] is trajectory
    assert validated["metadata"]["possible_choices"] == (-1, 1)
