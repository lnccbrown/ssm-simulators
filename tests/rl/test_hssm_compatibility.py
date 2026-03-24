"""Contract tests verifying output is consumable by HSSM."""

import numpy as np
import pandas as pd
import pytest

from ssms.rl import (
    RLSSMModelConfig,
    RLSSMSimulator,
    RescorlaWagnerDeltaRule,
    TwoArmedBandit,
    get_rlssm_preset,
    list_rlssm_presets,
)
from ssms.rl.rl_config import _HSSM_SHARED_FIELDS


@pytest.fixture()
def sim_data():
    config = RLSSMModelConfig(
        model_name="test_compat",
        description="Compatibility test",
        decision_process="angle",
        learning_process=RescorlaWagnerDeltaRule(),
        task_environment=TwoArmedBandit(
            reward_probabilities=[0.7, 0.3], choices=[-1, 1]
        ),
    )
    sim = RLSSMSimulator(config)
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
        assert isinstance(d["extra_fields"], list)

    def test_inference_placeholders_present(self, sim_data):
        _, config = sim_data
        d = config.to_hssm_config_dict()
        assert "ssm_logp_func" in d
        assert "learning_process" in d


class TestRegistry:
    def test_list_presets(self):
        presets = list_rlssm_presets()
        assert "rlssm1" in presets

    def test_get_rlssm1_preset(self):
        config = get_rlssm_preset("rlssm1")
        assert config.model_name == "rlssm1"
        assert config.decision_process == "angle"

    def test_rlssm1_preset_simulates(self):
        config = get_rlssm_preset("rlssm1")
        sim = RLSSMSimulator(config)
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

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError, match="Unknown RLSSM preset"):
            get_rlssm_preset("nonexistent")
