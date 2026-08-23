"""Contract tests for explicit projection of legacy simulator results."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from ssms.basic_simulators import OMISSION_SENTINEL

Schema = tuple[Mapping[str, Any], ...]
Projection = tuple[tuple[str, str], ...]

RT = {
    "name": "rt",
    "kind": "continuous",
    "lower": 0.0,
    "lower_inclusive": False,
}
BINARY_RESPONSE = {
    "name": "response",
    "kind": "categorical",
    "values": (-1, 1),
}
FOUR_WAY_RESPONSE = {
    "name": "response",
    "kind": "categorical",
    "values": (0, 1, 2, 3),
}
RT_RESPONSE_PROJECTION = (("rts", "rt"), ("choices", "response"))
RESPONSE_ONLY_PROJECTION = (("choices", "response"),)


def _normalize(
    result: Any,
    *,
    expected_n_samples: int = 2,
    expected_n_trials: int = 3,
    observation_schema: Schema = (RT, BINARY_RESPONSE),
    source_projection: Projection = RT_RESPONSE_PROJECTION,
) -> dict[str, Any]:
    from ssms.basic_simulators import normalize_simulator_result

    return normalize_simulator_result(
        result,
        expected_n_samples=expected_n_samples,
        expected_n_trials=expected_n_trials,
        observation_schema=observation_schema,
        source_projection=source_projection,
    )


def _legacy_shape(values: npt.ArrayLike, n_samples: int, n_trials: int) -> np.ndarray:
    """Return the exact historical squeeze shape for a scalar legacy source."""
    array = np.asarray(values).reshape(n_samples, n_trials, 1)
    if n_trials == 1:
        return np.squeeze(array, axis=1)
    if n_samples == 1:
        return np.squeeze(array, axis=0)
    return array


def _legacy_result(
    n_samples: int = 2,
    n_trials: int = 3,
    *,
    rt_dtype: npt.DTypeLike = np.float32,
    choice_dtype: npt.DTypeLike = np.int16,
) -> dict[str, Any]:
    size = n_samples * n_trials
    rts = _legacy_shape(
        np.arange(size, dtype=np.float64) / 10.0 + 0.2,
        n_samples,
        n_trials,
    ).astype(rt_dtype)
    choices = _legacy_shape(
        np.where(np.arange(size) % 2 == 0, -1, 1),
        n_samples,
        n_trials,
    ).astype(choice_dtype)
    return {
        "rts": rts,
        "choices": choices,
        "metadata": {
            "model": "ddm",
            "possible_choices": (-1, 1),
        },
    }


@pytest.mark.parametrize(
    ("n_samples", "n_trials"),
    [(1, 1), (4, 1), (1, 5), (3, 4)],
    ids=("one-sample-one-trial", "many-samples", "many-trials", "full-grid"),
)
def test_normalize_simulator_result_projects_rt_and_response_for_every_legacy_shape(
    n_samples: int,
    n_trials: int,
) -> None:
    result = _legacy_result(n_samples, n_trials)

    normalized = _normalize(
        result,
        expected_n_samples=n_samples,
        expected_n_trials=n_trials,
    )

    assert normalized["observations"].shape == (n_samples, n_trials, 2)
    assert normalized["observations"].dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        normalized["observations"][..., 0],
        result["rts"].reshape(n_samples, n_trials),
    )
    np.testing.assert_array_equal(
        normalized["observations"][..., 1],
        result["choices"].reshape(n_samples, n_trials),
    )
    assert not normalized["omission_mask"].any()


@pytest.mark.parametrize(
    ("n_samples", "n_trials"),
    [(1, 1), (4, 1), (1, 5), (3, 4)],
    ids=("one-sample-one-trial", "many-samples", "many-trials", "full-grid"),
)
def test_normalize_simulator_result_projects_response_only_for_every_legacy_shape(
    n_samples: int,
    n_trials: int,
) -> None:
    result = _legacy_result(n_samples, n_trials)
    result["rts"].fill(-1.0)

    normalized = _normalize(
        result,
        expected_n_samples=n_samples,
        expected_n_trials=n_trials,
        observation_schema=(BINARY_RESPONSE,),
        source_projection=RESPONSE_ONLY_PROJECTION,
    )

    assert normalized["observations"].shape == (n_samples, n_trials, 1)
    assert normalized["observations"].dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(
        normalized["observations"][..., 0],
        result["choices"].reshape(n_samples, n_trials),
    )
    assert not normalized["omission_mask"].any()
    assert np.all(result["rts"] == -1.0)


def test_normalize_simulator_result_projects_existing_rt_rlssm_response() -> None:
    result = _legacy_result(n_samples=2, n_trials=2)
    result["choices"] = _legacy_shape([0, 1, 2, 3], 2, 2).astype(np.int8)

    normalized = _normalize(
        result,
        expected_n_samples=2,
        expected_n_trials=2,
        observation_schema=(RT, FOUR_WAY_RESPONSE),
    )

    np.testing.assert_array_equal(
        normalized["observations"][..., 1], [[0.0, 1.0], [2.0, 3.0]]
    )


def test_normalize_simulator_result_projects_existing_choice_only_rlssm_response() -> (
    None
):
    result = _legacy_result(n_samples=1, n_trials=4)
    result["rts"].fill(-1.0)
    result["choices"] = _legacy_shape([0, 1, 2, 3], 1, 4).astype(np.int8)

    normalized = _normalize(
        result,
        expected_n_samples=1,
        expected_n_trials=4,
        observation_schema=(FOUR_WAY_RESPONSE,),
        source_projection=RESPONSE_ONLY_PROJECTION,
    )

    np.testing.assert_array_equal(
        normalized["observations"], [[[0.0], [1.0], [2.0], [3.0]]]
    )
    assert normalized["observations"].dtype == np.dtype(np.float64)


def test_normalize_simulator_result_converts_consistent_rt_response_omission() -> None:
    result = _legacy_result(n_samples=2, n_trials=2)
    result["rts"][1, 0] = OMISSION_SENTINEL
    result["choices"][1, 0] = OMISSION_SENTINEL

    normalized = _normalize(
        result,
        expected_n_samples=2,
        expected_n_trials=2,
    )

    np.testing.assert_array_equal(
        normalized["omission_mask"], [[False, False], [True, False]]
    )
    assert np.isnan(normalized["observations"][1, 0]).all()


def test_normalize_simulator_result_converts_choice_only_omission_and_ignores_dummy_rt() -> (
    None
):
    result = _legacy_result(n_samples=1, n_trials=3)
    result["rts"].fill(-1.0)
    result["choices"] = _legacy_shape([0, OMISSION_SENTINEL, 2], 1, 3).astype(np.int16)

    normalized = _normalize(
        result,
        expected_n_samples=1,
        expected_n_trials=3,
        observation_schema=(FOUR_WAY_RESPONSE,),
        source_projection=RESPONSE_ONLY_PROJECTION,
    )

    np.testing.assert_array_equal(normalized["omission_mask"], [[False, True, False]])
    assert np.isnan(normalized["observations"][0, 1, 0])
    np.testing.assert_array_equal(result["rts"], np.full((3, 1), -1.0, np.float32))


@pytest.mark.parametrize("sentinel_source", ["rts", "choices"])
def test_normalize_simulator_result_rejects_partial_legacy_omissions(
    sentinel_source: str,
) -> None:
    result = _legacy_result(n_samples=2, n_trials=2)
    result[sentinel_source][0, 1] = OMISSION_SENTINEL

    with pytest.raises(ValueError, match="sentinel.*every projected source"):
        _normalize(
            result,
            expected_n_samples=2,
            expected_n_trials=2,
        )


@pytest.mark.parametrize(
    ("n_samples", "n_trials", "wrong_shape"),
    [
        (1, 1, (1,)),
        (4, 1, (1, 4)),
        (1, 5, (1, 5)),
        (3, 4, (12, 1)),
    ],
    ids=("one-sample-one-trial", "many-samples", "many-trials", "full-grid"),
)
@pytest.mark.parametrize("source", ["rts", "choices"])
def test_normalize_simulator_result_rejects_equal_size_nonhistorical_shapes(
    n_samples: int,
    n_trials: int,
    wrong_shape: tuple[int, ...],
    source: str,
) -> None:
    result = _legacy_result(n_samples, n_trials)
    result[source] = result[source].reshape(wrong_shape)

    with pytest.raises(ValueError, match=rf"{source}.*historical legacy shape"):
        _normalize(
            result,
            expected_n_samples=n_samples,
            expected_n_trials=n_trials,
        )


@pytest.mark.parametrize(
    ("source_projection", "message"),
    [
        (("choices", "response"), "cover the schema exactly"),
        (
            (("choices", "response"), ("rts", "rt")),
            "schema order",
        ),
        (
            (("rts", "rt"), ("rts", "response")),
            "source keys must be unique",
        ),
        (
            (("rts", "rt"), ("choices", "rt")),
            "schema order",
        ),
        (
            (("latencies", "rt"), ("choices", "response")),
            "latencies",
        ),
        (
            (("rts", "rt"), ("choices", "response"), ("extra", "extra")),
            "cover the schema exactly",
        ),
    ],
)
def test_normalize_simulator_result_requires_explicit_one_to_one_schema_ordered_projection(
    source_projection: Projection,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _normalize(_legacy_result(), source_projection=source_projection)


def test_normalize_simulator_result_does_not_infer_source_names() -> None:
    result = _legacy_result()
    result["reaction_times"] = result.pop("rts")

    with pytest.raises(ValueError, match="rts"):
        _normalize(result)


def test_normalize_simulator_result_limits_projection_to_two_fields() -> None:
    result = _legacy_result()
    result["confidence"] = result["choices"].astype(np.float32)
    schema = (
        RT,
        BINARY_RESPONSE,
        {"name": "confidence", "kind": "continuous"},
    )

    with pytest.raises(ValueError, match="one or two fields"):
        _normalize(
            result,
            observation_schema=schema,
            source_projection=(
                ("rts", "rt"),
                ("choices", "response"),
                ("confidence", "confidence"),
            ),
        )


@pytest.mark.parametrize("reserved_key", ["observations", "omission_mask"])
def test_normalize_simulator_result_rejects_preexisting_canonical_arrays(
    reserved_key: str,
) -> None:
    result = _legacy_result()
    result[reserved_key] = np.empty((0,))

    with pytest.raises(ValueError, match=reserved_key):
        _normalize(result)


def test_normalize_simulator_result_shallow_copies_without_mutating_legacy_result() -> (
    None
):
    extension = {"boundary": object(), "trajectory": np.arange(4)}
    result = _legacy_result()
    result["metadata"]["extension"] = extension
    top_level_extension = object()
    result["diagnostics"] = top_level_extension
    original_keys = tuple(result)
    original_metadata_keys = tuple(result["metadata"])

    normalized = _normalize(result)

    assert normalized is not result
    assert normalized["rts"] is result["rts"]
    assert normalized["choices"] is result["choices"]
    assert normalized["diagnostics"] is top_level_extension
    assert normalized["metadata"] is not result["metadata"]
    assert normalized["metadata"]["extension"] is extension
    assert normalized["metadata"]["extension"]["boundary"] is extension["boundary"]
    assert normalized["metadata"]["extension"]["trajectory"] is extension["trajectory"]
    assert tuple(result) == original_keys
    assert tuple(result["metadata"]) == original_metadata_keys
    assert "observations" not in result
    assert "omission_mask" not in result
    assert "observation_schema" not in result["metadata"]


def test_normalize_simulator_result_accepts_identical_reserved_metadata() -> None:
    schema = (RT, BINARY_RESPONSE)
    result = _legacy_result()
    result["metadata"].update(
        {
            "observation_schema_version": 1,
            "observation_schema": tuple(dict(entry) for entry in schema),
        }
    )

    normalized = _normalize(result, observation_schema=schema)

    assert normalized["metadata"]["observation_schema_version"] == 1
    assert normalized["metadata"]["observation_schema"] == schema


@pytest.mark.parametrize(
    ("reserved_key", "reserved_value"),
    [
        ("observation_schema_version", 2),
        ("observation_schema", (BINARY_RESPONSE, RT)),
    ],
)
def test_normalize_simulator_result_rejects_conflicting_reserved_metadata(
    reserved_key: str,
    reserved_value: object,
) -> None:
    result = _legacy_result()
    result["metadata"][reserved_key] = reserved_value

    with pytest.raises(ValueError, match=rf"conflicting.*{reserved_key}"):
        _normalize(result)


@pytest.mark.parametrize(
    ("rt_dtype", "choice_dtype", "expected_dtype"),
    [
        (np.float16, np.int16, np.float32),
        (np.float32, np.int8, np.float32),
        (np.float32, np.int32, np.float64),
        (np.float64, np.int64, np.float64),
    ],
)
def test_normalize_simulator_result_uses_numpy_promotion_for_mixed_sources(
    rt_dtype: npt.DTypeLike,
    choice_dtype: npt.DTypeLike,
    expected_dtype: npt.DTypeLike,
) -> None:
    result = _legacy_result(
        rt_dtype=rt_dtype,
        choice_dtype=choice_dtype,
    )

    normalized = _normalize(result)

    assert normalized["observations"].dtype == np.dtype(expected_dtype)


def test_normalize_simulator_result_revalidates_labels_after_dtype_promotion() -> None:
    unrepresentable_label = 2**53 + 1
    result = _legacy_result(rt_dtype=np.float32, choice_dtype=np.int64)
    result["choices"].fill(unrepresentable_label)
    response = {
        "name": "response",
        "kind": "categorical",
        "values": (unrepresentable_label,),
    }

    with pytest.raises(ValueError, match="not exactly representable.*float64"):
        _normalize(result, observation_schema=(RT, response))


def test_normalize_simulator_result_accepts_representable_label_after_promotion() -> (
    None
):
    representable_label = 2**53 + 2
    result = _legacy_result(rt_dtype=np.float32, choice_dtype=np.int64)
    result["choices"].fill(representable_label)
    response = {
        "name": "response",
        "kind": "categorical",
        "values": (representable_label,),
    }

    normalized = _normalize(result, observation_schema=(RT, response))

    assert normalized["observations"].dtype == np.dtype(np.float64)
    assert np.all(normalized["observations"][..., 1] == representable_label)


@pytest.mark.parametrize(
    ("expected_n_samples", "expected_n_trials", "error", "message"),
    [
        (True, 3, TypeError, "expected_n_samples"),
        (2.0, 3, TypeError, "expected_n_samples"),
        (0, 3, ValueError, "expected_n_samples"),
        (2, False, TypeError, "expected_n_trials"),
        (2, 1.5, TypeError, "expected_n_trials"),
        (2, -1, ValueError, "expected_n_trials"),
    ],
)
def test_normalize_simulator_result_requires_positive_integer_counts(
    expected_n_samples: Any,
    expected_n_trials: Any,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _normalize(
            _legacy_result(),
            expected_n_samples=expected_n_samples,
            expected_n_trials=expected_n_trials,
        )


@pytest.mark.parametrize(
    ("mutate", "error", "message"),
    [
        (lambda result: result.pop("metadata"), ValueError, "metadata"),
        (lambda result: result.__setitem__("metadata", []), TypeError, "metadata"),
        (lambda result: result.__setitem__("rts", []), TypeError, "rts"),
        (
            lambda result: result.__setitem__(
                "choices", result["choices"].astype(np.complex64)
            ),
            TypeError,
            "choices.*real numeric",
        ),
        (
            lambda result: result.__setitem__(
                "choices", result["choices"].astype(np.bool_)
            ),
            TypeError,
            "choices.*real numeric",
        ),
    ],
    ids=(
        "missing-metadata",
        "nonmapping-metadata",
        "nonnumpy-source",
        "complex-source",
        "boolean-source",
    ),
)
def test_normalize_simulator_result_rejects_invalid_legacy_containers(
    mutate: Any,
    error: type[Exception],
    message: str,
) -> None:
    result = _legacy_result()
    mutate(result)

    with pytest.raises(error, match=message):
        _normalize(result)


def test_normalize_simulator_result_requires_a_mapping() -> None:
    with pytest.raises(TypeError, match="mapping"):
        _normalize([])


@pytest.mark.parametrize(
    ("observation_schema", "source_projection", "error", "message"),
    [
        ([RT, BINARY_RESPONSE], RT_RESPONSE_PROJECTION, TypeError, "ordered tuple"),
        ((RT,), [RESPONSE_ONLY_PROJECTION[0]], TypeError, "ordered tuple"),
        ((RT,), (("rts",),), TypeError, "pair"),
        ((RT,), ((1, "rt"),), TypeError, "non-empty strings"),
    ],
)
def test_normalize_simulator_result_requires_tuple_schema_and_projection_contracts(
    observation_schema: Any,
    source_projection: Any,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _normalize(
            _legacy_result(),
            observation_schema=observation_schema,
            source_projection=source_projection,
        )
