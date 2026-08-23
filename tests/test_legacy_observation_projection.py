"""Contract tests for explicit projection of legacy simulator results."""

from collections.abc import Callable, Mapping
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
ZERO_ONE_RESPONSE = {
    "name": "response",
    "kind": "categorical",
    "values": (0, 1),
}
RT_RESPONSE_PROJECTION = (("rts", "rt"), ("choices", "response"))
RESPONSE_ONLY_PROJECTION = (("choices", "response"),)
LEGACY_CASES = (
    pytest.param((1, 1, (1,)), id="one-sample-one-trial"),
    pytest.param((4, 1, (1, 4)), id="many-samples"),
    pytest.param((1, 5, (1, 5)), id="many-trials"),
    pytest.param((3, 4, (12, 1)), id="full-grid"),
)


def _normalize(
    result: Any,
    *,
    expected_n_samples: int = 2,
    expected_n_trials: int = 3,
    observation_schema: Schema = (RT, BINARY_RESPONSE),
    source_projection: Projection = RT_RESPONSE_PROJECTION,
    omission_source: str = "rts",
) -> dict[str, Any]:
    from ssms.basic_simulators import normalize_simulator_result

    return normalize_simulator_result(
        result,
        expected_n_samples=expected_n_samples,
        expected_n_trials=expected_n_trials,
        observation_schema=observation_schema,
        source_projection=source_projection,
        omission_source=omission_source,
    )


def _legacy_shape(values: npt.ArrayLike, n_samples: int, n_trials: int) -> np.ndarray:
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
    return {
        "rts": _legacy_shape(np.arange(size) / 10.0 + 0.2, n_samples, n_trials).astype(
            rt_dtype
        ),
        "choices": _legacy_shape(
            np.where(np.arange(size) % 2 == 0, -1, 1), n_samples, n_trials
        ).astype(choice_dtype),
        "metadata": {"model": "ddm", "possible_choices": (-1, 1)},
    }


@pytest.mark.parametrize("case", LEGACY_CASES)
@pytest.mark.parametrize("response_only", [False, True], ids=("rt-choice", "response"))
def test_projects_every_historical_legacy_shape(
    case: tuple[int, int, tuple[int, ...]],
    response_only: bool,
) -> None:
    n_samples, n_trials, _ = case
    result = _legacy_result(n_samples, n_trials)
    kwargs: dict[str, Any] = {}
    expected_width = 2
    if response_only:
        result["rts"].fill(-1.0)
        kwargs = {
            "observation_schema": (BINARY_RESPONSE,),
            "source_projection": RESPONSE_ONLY_PROJECTION,
            "omission_source": "choices",
        }
        expected_width = 1

    normalized = _normalize(
        result,
        expected_n_samples=n_samples,
        expected_n_trials=n_trials,
        **kwargs,
    )

    assert normalized["observations"].shape == (n_samples, n_trials, expected_width)
    np.testing.assert_array_equal(
        normalized["observations"][..., -1],
        result["choices"].reshape(n_samples, n_trials),
    )
    assert normalized["observations"].dtype == (
        np.dtype(np.float64) if response_only else np.dtype(np.float32)
    )
    assert not normalized["omission_mask"].any()


@pytest.mark.parametrize("case", LEGACY_CASES)
@pytest.mark.parametrize("response_only", [False, True], ids=("rt-choice", "response"))
def test_rejects_equal_size_nonhistorical_shapes(
    case: tuple[int, int, tuple[int, ...]],
    response_only: bool,
) -> None:
    n_samples, n_trials, wrong_shape = case
    result = _legacy_result(n_samples, n_trials)
    malformed_source = "choices" if response_only else "rts"
    result[malformed_source] = result[malformed_source].reshape(wrong_shape)
    kwargs = (
        {
            "observation_schema": (BINARY_RESPONSE,),
            "source_projection": RESPONSE_ONLY_PROJECTION,
            "omission_source": "choices",
        }
        if response_only
        else {}
    )

    with pytest.raises(
        ValueError, match=rf"{malformed_source}.*historical legacy shape"
    ):
        _normalize(
            result,
            expected_n_samples=n_samples,
            expected_n_trials=n_trials,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("model", "response_schema"),
    [("ddm_deadline", BINARY_RESPONSE), ("lba2_deadline", ZERO_ONE_RESPONSE)],
)
def test_registered_deadline_models_use_rt_as_omission_authority(
    model: str,
    response_schema: Mapping[str, Any],
) -> None:
    from ssms import Simulator

    simulator = Simulator(model)
    theta = dict(
        zip(simulator.config["params"], simulator.config["default_params"], strict=True)
    )
    theta["deadline"] = 0.001
    result = simulator.simulate(theta, n_samples=8, random_state=42)

    assert np.all(result["rts"] == OMISSION_SENTINEL)
    assert np.all(result["choices"] != OMISSION_SENTINEL)

    normalized = _normalize(
        result,
        expected_n_samples=8,
        expected_n_trials=1,
        observation_schema=(RT, response_schema),
    )

    assert normalized["omission_mask"].tolist() == [[True]] * 8
    assert np.isnan(normalized["observations"]).all()


@pytest.mark.parametrize(("auth", "aux"), [("rts", "choices"), ("choices", "rts")])
def test_rejects_auxiliary_sentinel(auth: str, aux: str) -> None:
    result = _legacy_result()
    result[aux][0, 1, 0] = OMISSION_SENTINEL

    with pytest.raises(ValueError, match=rf"non-authoritative.*{aux}"):
        _normalize(result, omission_source=auth)


def test_omission_authority_controls_projected_and_unprojected_values() -> None:
    rt_response = _legacy_result(2, 2)
    rt_response["rts"][1, 0] = OMISSION_SENTINEL
    rt_response["choices"][1, 0] = OMISSION_SENTINEL
    original = {
        key: value.copy() for key, value in rt_response.items() if key != "metadata"
    }

    normalized = _normalize(rt_response, expected_n_samples=2, expected_n_trials=2)

    np.testing.assert_array_equal(
        normalized["omission_mask"], [[False, False], [True, False]]
    )
    assert np.isnan(normalized["observations"][1, 0]).all()
    np.testing.assert_array_equal(rt_response["rts"], original["rts"])
    np.testing.assert_array_equal(rt_response["choices"], original["choices"])

    response_only = _legacy_result(1, 3)
    response_only["rts"].fill(OMISSION_SENTINEL)
    response_only["choices"] = _legacy_shape([0, OMISSION_SENTINEL, 2], 1, 3).astype(
        np.int16
    )
    normalized = _normalize(
        response_only,
        expected_n_samples=1,
        expected_n_trials=3,
        observation_schema=(FOUR_WAY_RESPONSE,),
        source_projection=RESPONSE_ONLY_PROJECTION,
        omission_source="choices",
    )

    np.testing.assert_array_equal(normalized["omission_mask"], [[False, True, False]])
    assert np.isnan(normalized["observations"][0, 1]).all()
    assert np.all(response_only["rts"] == OMISSION_SENTINEL)


@pytest.mark.parametrize(
    ("source_projection", "message"),
    [
        (("choices", "response"), "pair"),
        ((("choices", "response"),), "cover the schema exactly"),
        ((("choices", "response"), ("rts", "rt")), "schema order"),
        ((("rts", "rt"), ("rts", "response")), "source keys must be unique"),
        ((("rts", "rt"), ("missing", "response")), "missing projected source"),
    ],
)
def test_requires_explicit_ordered_schema_projection(
    source_projection: Any,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _normalize(
            _legacy_result(),
            source_projection=source_projection,
            omission_source="rts",
        )


def test_limits_legacy_projection_to_two_scalar_fields() -> None:
    result = _legacy_result()
    result["confidence"] = result["choices"].astype(np.float32)

    with pytest.raises(ValueError, match="one or two fields"):
        _normalize(
            result,
            observation_schema=(
                RT,
                BINARY_RESPONSE,
                {"name": "confidence", "kind": "continuous"},
            ),
            source_projection=(
                ("rts", "rt"),
                ("choices", "response"),
                ("confidence", "confidence"),
            ),
        )


def test_shallow_copies_without_mutating_legacy_result() -> None:
    extension = {"boundary": object(), "trajectory": np.arange(4)}
    result = _legacy_result()
    result["metadata"]["extension"] = extension
    diagnostic = object()
    result["diagnostics"] = diagnostic
    original_keys = tuple(result)
    original_metadata_keys = tuple(result["metadata"])

    normalized = _normalize(result)

    assert normalized is not result
    assert normalized["rts"] is result["rts"]
    assert normalized["choices"] is result["choices"]
    assert normalized["diagnostics"] is diagnostic
    assert normalized["metadata"] is not result["metadata"]
    assert normalized["metadata"]["extension"] is extension
    assert tuple(result) == original_keys
    assert tuple(result["metadata"]) == original_metadata_keys


def test_retains_identical_reserved_metadata() -> None:
    schema = (RT, BINARY_RESPONSE)
    result = _legacy_result()
    result["metadata"].update(
        {"observation_schema_version": 1, "observation_schema": schema}
    )

    normalized = _normalize(result, observation_schema=schema)

    assert normalized["metadata"]["observation_schema"] is schema


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("observation_schema_version", 2),
        ("observation_schema", (BINARY_RESPONSE, RT)),
    ],
)
def test_rejects_conflicting_reserved_metadata(key: str, value: object) -> None:
    result = _legacy_result()
    result["metadata"][key] = value

    with pytest.raises(ValueError, match=rf"conflicting.*{key}"):
        _normalize(result)


@pytest.mark.parametrize("key", ["observations", "omission_mask"])
def test_rejects_preexisting_canonical_arrays(key: str) -> None:
    result = _legacy_result()
    result[key] = np.empty((0,))

    with pytest.raises(ValueError, match=key):
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
def test_uses_numpy_promotion_for_projected_sources(
    rt_dtype: npt.DTypeLike,
    choice_dtype: npt.DTypeLike,
    expected_dtype: npt.DTypeLike,
) -> None:
    normalized = _normalize(
        _legacy_result(rt_dtype=rt_dtype, choice_dtype=choice_dtype)
    )

    assert normalized["observations"].dtype == np.dtype(expected_dtype)


@pytest.mark.parametrize(
    ("allowed_label", "raw_label"),
    [(2**53 + 1, 2**53 + 1), (2**53, 2**53 + 1)],
    ids=("unrepresentable-schema-label", "rounding-collision"),
)
def test_rejects_categorical_precision_loss(
    allowed_label: int,
    raw_label: int,
) -> None:
    result = _legacy_result(rt_dtype=np.float64, choice_dtype=np.int64)
    result["choices"].fill(raw_label)
    response = {
        "name": "response",
        "kind": "categorical",
        "values": (allowed_label,),
    }

    with pytest.raises(ValueError, match="not exactly representable.*float64"):
        _normalize(result, observation_schema=(RT, response))


def test_accepts_large_exactly_representable_label_after_promotion() -> None:
    label = 2**53 + 2
    result = _legacy_result(rt_dtype=np.float32, choice_dtype=np.int64)
    result["choices"].fill(label)
    response = {"name": "response", "kind": "categorical", "values": (label,)}

    normalized = _normalize(result, observation_schema=(RT, response))

    assert np.all(normalized["observations"][..., 1] == label)


def test_final_native_validation_rejects_out_of_domain_response() -> None:
    result = _legacy_result()
    result["choices"][0, 0, 0] = 3

    with pytest.raises(ValueError, match="response.*outside its categorical domain"):
        _normalize(result)


@pytest.mark.parametrize(
    ("n_samples", "n_trials", "error", "message"),
    [
        (True, 3, TypeError, "expected_n_samples"),
        (0, 3, ValueError, "expected_n_samples"),
        (2, 1.5, TypeError, "expected_n_trials"),
        (2, -1, ValueError, "expected_n_trials"),
    ],
)
def test_requires_positive_integer_counts(
    n_samples: Any,
    n_trials: Any,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _normalize(
            _legacy_result(),
            expected_n_samples=n_samples,
            expected_n_trials=n_trials,
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
)
def test_rejects_representative_invalid_containers(
    mutate: Callable[[dict[str, Any]], Any],
    error: type[Exception],
    message: str,
) -> None:
    result = _legacy_result()
    mutate(result)

    with pytest.raises(error, match=message):
        _normalize(result)


def test_requires_mapping_and_projected_omission_source() -> None:
    with pytest.raises(TypeError, match="mapping"):
        _normalize([])

    with pytest.raises(ValueError, match="omission_source.*projected"):
        _normalize(
            _legacy_result(),
            observation_schema=(BINARY_RESPONSE,),
            source_projection=RESPONSE_ONLY_PROJECTION,
            omission_source="rts",
        )
