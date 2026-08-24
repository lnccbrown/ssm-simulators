"""Focused facts for built-in producer observation metadata."""

from collections.abc import Mapping
from typing import Any

import pytest

from ssms.basic_simulators import get_observation_metadata
from ssms.config import ModelConfigBuilder
from ssms.config._modelconfig import get_model_config


CHOICE_ONLY_MODELS = {
    "inv_temp_softmax_2",
    "inv_temp_softmax_3",
    "inv_temp_softmax_4",
}
ALIASES_WITH_INTERNAL_NAMES = {
    "ddm_mic2_ornstein_no_bias_no_lowdim_noise": "ddm_mic2_ornstein_no_bias",
    "ddm_mic2_ornstein_conflict_gamma_no_bias_no_lowdim_noise": (
        "ddm_mic2_ornstein_conflict_gamma_no_bias"
    ),
    "ddm_mic2_ornstein_angle_no_bias_no_lowdim_noise": (
        "ddm_mic2_ornstein_angle_no_bias"
    ),
    "ddm_mic2_ornstein_weibull_no_bias_no_lowdim_noise": (
        "ddm_mic2_ornstein_weibull_no_bias"
    ),
    "ddm_mic2_leak_no_bias_no_lowdim_noise": "ddm_mic2_leak_no_bias",
    "ddm_mic2_leak_conflict_gamma_no_bias_no_lowdim_noise": (
        "ddm_mic2_leak_conflict_gamma_no_bias"
    ),
    "ddm_mic2_leak_angle_no_bias_no_lowdim_noise": ("ddm_mic2_leak_angle_no_bias"),
    "ddm_mic2_leak_weibull_no_bias_no_lowdim_noise": ("ddm_mic2_leak_weibull_no_bias"),
    "shrink_spot_extended": "shrink_spot",
    "weibull_cdf": "weibull",
    "full_ddm2": "full_ddm",
}
RT = {
    "name": "rt",
    "kind": "continuous",
    "lower": 0.0,
    "lower_inclusive": False,
}


def _descriptor(*fields: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "observation_schema_version": 1,
        "observation_schema": fields,
        "obs_dim": len(fields),
    }


def _response(values: tuple[int, ...]) -> dict[str, Any]:
    return {"name": "response", "kind": "categorical", "values": values}


def test_all_raw_builtin_configs_are_explicitly_classified() -> None:
    configs = get_model_config()
    explicit = {
        name for name, config in configs.items() if "observation_schema" in config
    }
    profiled = {
        name
        for name, config in configs.items()
        if "observation_schema_profile" in config
    }

    assert explicit == CHOICE_ONLY_MODELS
    assert len(profiled) == 110
    assert explicit.isdisjoint(profiled)
    assert explicit | profiled == set(configs)
    assert {config["observation_schema_version"] for config in configs.values()} == {1}
    assert {configs[name]["observation_schema_profile"] for name in profiled} == {
        "legacy_rt_choice"
    }


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("ddm", _descriptor(RT, _response((-1, 1)))),
        ("race_3", _descriptor(RT, _response((0, 1, 2)))),
        ("lba4", _descriptor(RT, _response((0, 1, 2, 3)))),
        ("inv_temp_softmax_2", _descriptor(_response((0, 1)))),
        ("inv_temp_softmax_3", _descriptor(_response((0, 1, 2)))),
        ("inv_temp_softmax_4", _descriptor(_response((0, 1, 2, 3)))),
    ],
)
def test_representative_builtin_semantics_are_exact(
    model_name: str, expected: dict[str, Any]
) -> None:
    config = ModelConfigBuilder.from_model(model_name)

    assert get_observation_metadata(config) == expected


def test_registry_aliases_are_classified_despite_internal_names() -> None:
    configs = get_model_config()
    aliases = {
        name: config["name"]
        for name, config in configs.items()
        if name != config["name"]
    }

    assert aliases == ALIASES_WITH_INTERNAL_NAMES
    for registry_name, internal_name in aliases.items():
        config = ModelConfigBuilder.from_model(registry_name)
        assert config["name"] == internal_name
        assert get_observation_metadata(config) == _descriptor(
            RT, _response(tuple(config["choices"]))
        )


def test_deadline_builder_inherits_observation_classification() -> None:
    base = ModelConfigBuilder.from_model("ddm")
    deadline = ModelConfigBuilder.from_model("ddm_deadline")

    assert deadline["observation_schema_version"] == base["observation_schema_version"]
    assert deadline["observation_schema_profile"] == base["observation_schema_profile"]
    assert get_observation_metadata(deadline) == _descriptor(RT, _response((-1, 1)))


def test_custom_config_fails_closed_without_explicit_classification() -> None:
    def custom_simulator() -> None:
        return None

    config = ModelConfigBuilder.minimal_config(
        params=[], simulator_function=custom_simulator, nchoices=4
    )
    config["observation_schema_version"] = 1
    config["obs_dim"] = 2

    assert config["choices"] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="exactly one"):
        get_observation_metadata(config)
