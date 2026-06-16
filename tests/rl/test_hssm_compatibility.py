"""Contract tests verifying output is consumable by HSSM."""

import pandas as pd
import pytest

import ssms.rl as rl
from ssms.rl.config import _HSSM_SHARED_FIELDS


@pytest.fixture()
def sim_data():
    config = rl.ModelConfig(
        model_name="test_compat",
        description="Compatibility test",
        decision_process="angle",
        learning_process=rl.learning.RescorlaWagnerDeltaRule(),
        task_environment=rl.env.Bandit.bernoulli(
            probabilities=[0.7, 0.3], response_labels=[-1, 1]
        ),
    )
    sim = rl.Simulator(config)
    data = sim.simulate(
        theta={
            "rl_alpha": 0.2,
            "scaler": 2.0,
            "a": 1.5,
            "z": 0.5,
            "t": 0.3,
            "theta": 0.2,
        },
        n_trials=20,
        n_participants=3,
        random_state=42,
    )
    return data, config


class TestOutputDtypes:
    def test_participant_id_int(self, sim_data):
        data, _ = sim_data
        assert pd.api.types.is_integer_dtype(data["participant_id"])

    def test_trial_id_int(self, sim_data):
        data, _ = sim_data
        assert pd.api.types.is_integer_dtype(data["trial_id"])

    def test_rt_float(self, sim_data):
        data, _ = sim_data
        assert pd.api.types.is_float_dtype(data["rt"])

    def test_response_int(self, sim_data):
        data, _ = sim_data
        assert pd.api.types.is_integer_dtype(data["response"])

    def test_feedback_float(self, sim_data):
        data, _ = sim_data
        assert pd.api.types.is_float_dtype(data["feedback"])


class TestOutputQuality:
    def test_no_nans(self, sim_data):
        data, _ = sim_data
        assert not data.isna().any().any()


class TestToHssmConfigDictSchema:
    def test_all_required_fields(self, sim_data):
        _, config = sim_data
        d = config.to_hssm_config_dict()
        for field_name in _HSSM_SHARED_FIELDS:
            assert field_name in d
            assert d[field_name] is not None

    def test_field_types(self, sim_data):
        _, config = sim_data
        d = config.to_hssm_config_dict()
        assert isinstance(d["model_name"], str)
        assert isinstance(d["description"], str)
        assert isinstance(d["decision_process"], str)
        assert isinstance(d["list_params"], list)
        assert isinstance(d["bounds"], dict)
        assert isinstance(d["params_default"], list)
        assert isinstance(d["choices"], tuple)
        assert isinstance(d["response"], list)
        assert isinstance(d["response_to_choice"], dict)
        assert isinstance(d["context_fields"], list)

    def test_inference_placeholders_present(self, sim_data):
        _, config = sim_data
        d = config.to_hssm_config_dict()
        assert "ssm_logp_func" in d
        assert "learning_process" in d
        assert "learning_process_kind" in d
        assert "learning_process_loglik_kind" not in d


class TestRegistry:
    def test_list_presets(self):
        presets = rl.preset.list()
        assert "2AB_RW_Angle" in presets
        assert "rlssm1" not in presets

    def test_get_two_arm_rw_angle_preset(self):
        config = rl.preset.get("2AB_RW_Angle")
        assert isinstance(config, rl.ModelConfig)
        assert config.model_name == "2AB_RW_Angle"
        assert config.decision_process == "angle"

    def test_two_arm_rw_angle_preset_simulates(self):
        config = rl.preset.get("2AB_RW_Angle")
        sim = rl.Simulator(config)
        data = sim.simulate(
            theta={
                "rl_alpha": 0.2,
                "scaler": 2.0,
                "a": 1.5,
                "z": 0.5,
                "t": 0.3,
                "theta": 0.2,
            },
            n_trials=10,
            n_participants=2,
            random_state=42,
        )
        assert len(data) == 20

    def test_old_rlssm1_preset_name_is_not_public(self):
        with pytest.raises(KeyError, match="Unknown RLSSM preset"):
            rl.preset.get("rlssm1")

    def test_preset_info_contains_readable_metadata(self):
        info = rl.preset.info("2AB_RW_Angle")

        assert info["name"] == "2AB_RW_Angle"
        assert info["task"] == "two-armed Bernoulli bandit"
        assert info["learning_process"] == "RescorlaWagnerDeltaRule"
        assert info["decision_process"] == "angle"
        assert info["required_parameters"] == [
            "rl_alpha",
            "scaler",
            "a",
            "z",
            "t",
            "theta",
        ]
        assert info["default_parameters"]["rl_alpha"] == 0.2
        assert info["bounds"]["rl_alpha"] == (0.0, 1.0)
        assert info["response_labels"] == (-1, 1)
        assert info["response_to_choice"] == {-1: 0, 1: 1}
        assert info["context_fields"] == ["feedback"]
        assert info["hssm_compatibility"]["participant_contract"] is True
        assert info["learning_backend"] in {"python", "jax"}
        assert info["gradient"] in {"available", "unavailable"}

        rendered = str(info)
        assert "2AB_RW_Angle" in rendered
        assert "required parameters" in rendered.lower()

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError, match="Unknown RLSSM preset"):
            rl.preset.get("nonexistent")

    def test_register_custom_preset(self):
        def factory():
            return rl.ModelConfig(
                model_name="custom",
                description="Custom preset",
                decision_process="angle",
                learning_process=rl.learning.RescorlaWagnerDeltaRule(),
                task_environment=rl.env.Bandit.bernoulli(response_labels=[-1, 1]),
            )

        rl.preset.register("custom_test", factory)
        assert "custom_test" in rl.preset.list()
        assert rl.preset.get("custom_test").model_name == "custom"


class TestPublicApiSurface:
    def test_public_exports_are_small(self):
        assert rl.__all__ == [
            "Simulator",
            "ModelConfig",
            "AssembledModel",
            "resolve_model",
            "env",
            "learning",
            "preset",
        ]

    @pytest.mark.parametrize(
        "name",
        [
            "RLSSMSimulator",
            "RLSSMModelConfig",
            "get_rlssm_preset",
            "list_rlssm_presets",
            "register_rlssm_preset",
        ],
    )
    def test_old_developmental_names_are_not_exported(self, name):
        assert not hasattr(rl, name)
