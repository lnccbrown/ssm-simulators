"""RLSSM producer observation-metadata contracts."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ssms.basic_simulators import get_observation_metadata
import ssms.rl as rl


def _rt_field() -> dict[str, object]:
    return {
        "name": "rt",
        "kind": "continuous",
        "lower": 0.0,
        "lower_inclusive": False,
    }


def _response_field(values: tuple[int, ...]) -> dict[str, object]:
    return {"name": "response", "kind": "categorical", "values": values}


def _descriptor(*fields: dict[str, object]) -> dict[str, object]:
    return {
        "observation_schema_version": 1,
        "observation_schema": fields,
        "obs_dim": len(fields),
    }


def test_rt_response_preset_exposes_exact_metadata_through_public_accessor():
    config = rl.preset.get("2AB_RW_Angle")
    assembled = config.assemble(backend="python")
    expected = _descriptor(_rt_field(), _response_field((-1, 1)))

    assert get_observation_metadata(config) == expected
    assert get_observation_metadata(assembled) == expected


@pytest.mark.parametrize(
    ("preset_name", "choices"),
    [
        ("2AB_RW_InvTempSoftmax", (0, 1)),
        ("3AB_RW_InvTempSoftmax", (0, 1, 2)),
        ("4AB_RW_InvTempSoftmax", (0, 1, 2, 3)),
    ],
)
def test_response_only_presets_expose_exact_metadata(preset_name, choices):
    config = rl.preset.get(preset_name)
    assembled = config.assemble(backend="python")
    expected = _descriptor(_response_field(choices))

    assert get_observation_metadata(config) == expected
    assert get_observation_metadata(assembled) == expected


def test_metadata_access_preserves_seeded_choice_only_simulation_and_input_order():
    config = rl.preset.get("2AB_RW_InvTempSoftmax")
    assembled = config.assemble(backend="python")
    simulator = rl.Simulator(config)
    theta = {"rl_alpha": 0.2, "beta": 2.0}

    input_fields_before = assembled.get_participant_input_fields()
    simulated_before = simulator.simulate(
        theta=theta,
        n_trials=8,
        n_participants=2,
        random_state=37,
    )

    metadata = get_observation_metadata(config)
    get_observation_metadata(assembled)

    input_fields_after = assembled.get_participant_input_fields()
    simulated_after = simulator.simulate(
        theta=theta,
        n_trials=8,
        n_participants=2,
        random_state=37,
    )

    assert (
        input_fields_before
        == input_fields_after
        == [
            "rl_alpha",
            "response",
            "feedback",
        ]
    )
    schema_names = tuple(field["name"] for field in metadata["observation_schema"])
    assert schema_names == ("response",)
    assert config.context_fields == ["feedback"]
    assert set(config.context_fields).isdisjoint(schema_names)
    assert np.all(simulated_before["rt"] == -1.0)
    pd.testing.assert_frame_equal(simulated_after, simulated_before)


def test_schema_choice_order_does_not_follow_response_to_choice_mapping():
    config = replace(
        rl.preset.get("2AB_RW_Angle"),
        response_to_choice={1: 0, -1: 1},
    )

    metadata = get_observation_metadata(config)

    assert tuple(config.resolved_response_to_choice) == (1, -1)
    assert config.choices == (-1, 1)
    assert metadata["observation_schema"][1]["values"] == (-1, 1)


def test_nonstandard_response_layout_requires_an_explicit_v1_ordered_schema():
    response = ["rt", "confidence", "response"]
    schema = (
        _rt_field(),
        {
            "name": "confidence",
            "kind": "continuous",
            "lower": 0.0,
            "upper": 1.0,
        },
        _response_field((-1, 1)),
    )
    config = replace(rl.preset.get("2AB_RW_Angle"), response=response)

    with pytest.raises(ValueError, match="require an explicit observation_schema"):
        get_observation_metadata(config)

    explicit = replace(
        config,
        observation_schema_version=1,
        observation_schema=schema,
    )
    assert get_observation_metadata(explicit) == _descriptor(*schema)

    unsupported_version = replace(explicit, observation_schema_version=2)
    with pytest.raises(ValueError, match="supported integer version 1"):
        get_observation_metadata(unsupported_version)


@pytest.mark.parametrize(
    ("schema", "match"),
    [
        (
            (_response_field((-1, 1)), _rt_field()),
            "names and order must exactly match response",
        ),
        (
            (_rt_field(), _response_field((1, -1))),
            "values equal to choices in order",
        ),
        (
            (_rt_field(), {"name": "response", "kind": "continuous"}),
            "must be categorical",
        ),
    ],
    ids=("field-order", "categorical-values", "categorical-kind"),
)
def test_explicit_schema_must_match_the_rl_response_contract(schema, match):
    config = replace(
        rl.preset.get("2AB_RW_Angle"),
        observation_schema=schema,
    )

    with pytest.raises(ValueError, match=match):
        get_observation_metadata(config)


def test_assembled_metadata_is_an_independent_fresh_snapshot():
    raw_values = [-1, 1]
    config = replace(
        rl.preset.get("2AB_RW_Angle"),
        observation_schema=(
            _rt_field(),
            {"name": "response", "kind": "categorical", "values": raw_values},
        ),
    )
    assembled = config.assemble(backend="python")

    raw_values[0] = 99
    first = get_observation_metadata(assembled)
    first["observation_schema"][1]["values"] = (99, 100)
    second = get_observation_metadata(assembled)

    assert second == _descriptor(_rt_field(), _response_field((-1, 1)))
    assert first is not second
    assert first["observation_schema"] is not second["observation_schema"]
    assert first["observation_schema"][0] is not second["observation_schema"][0]


def test_metadata_access_does_not_change_or_extend_the_hssm_config_dict():
    config = rl.preset.get("2AB_RW_Angle")
    before = config.to_hssm_config_dict()

    get_observation_metadata(config)
    get_observation_metadata(config.assemble(backend="python"))

    after = config.to_hssm_config_dict()
    assert after == before
    assert config.observation_schema is None
    assert {
        "observation_schema_version",
        "observation_schema",
        "obs_dim",
    }.isdisjoint(after)
