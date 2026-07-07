"""Choice-only inverse-temperature softmax RL presets."""

import numpy as np
import pandas as pd
import pytest

from ssms.basic_simulators.simulator import simulator as ssm_simulator
from ssms.config import ModelConfigBuilder
import ssms.rl as rl


def test_inv_temp_softmax_model_configs_are_registered():
    cfg2 = ModelConfigBuilder.from_model("inv_temp_softmax_2")
    cfg3 = ModelConfigBuilder.from_model("inv_temp_softmax_3")

    assert cfg2["params"] == ["beta", "q0", "q1"]
    assert cfg2["choices"] == [0, 1]
    assert cfg2["nchoices"] == 2
    assert cfg3["params"] == ["beta", "q0", "q1", "q2"]
    assert cfg3["choices"] == [0, 1, 2]
    assert cfg3["nchoices"] == 3


def test_inv_temp_softmax_simulator_emits_choices_and_placeholder_rt():
    result = ssm_simulator(
        theta={"beta": 5.0, "q0": 0.0, "q1": 1.0},
        model="inv_temp_softmax_2",
        n_samples=20,
        random_state=123,
    )

    assert set(np.unique(result["choices"])).issubset({0, 1})
    assert np.all(result["rts"] == -1.0)
    assert result["metadata"]["possible_choices"] == [0, 1]


def test_inv_temp_softmax_beta_controls_choice_concentration():
    low_beta = ssm_simulator(
        theta={"beta": 0.0, "q0": 0.0, "q1": 1.0},
        model="inv_temp_softmax_2",
        n_samples=1000,
        random_state=11,
    )
    high_beta = ssm_simulator(
        theta={"beta": 8.0, "q0": 0.0, "q1": 1.0},
        model="inv_temp_softmax_2",
        n_samples=1000,
        random_state=11,
    )

    low_p_choice_1 = np.mean(low_beta["choices"].reshape(-1) == 1)
    high_p_choice_1 = np.mean(high_beta["choices"].reshape(-1) == 1)

    assert 0.4 < low_p_choice_1 < 0.6
    assert high_p_choice_1 > 0.95


@pytest.mark.parametrize(
    ("preset_name", "n_choices", "theta"),
    [
        ("2AB_RW_InvTempSoftmax", 2, {"rl_alpha": 0.2, "beta": 2.0}),
        ("3AB_RW_InvTempSoftmax", 3, {"rl_alpha": 0.2, "beta": 2.0}),
    ],
)
def test_choice_only_presets_validate_assemble_and_simulate(
    preset_name, n_choices, theta
):
    config = rl.preset.get(preset_name)

    config.validate()
    assembled = config.assemble(backend="jax")
    data = rl.Simulator(config).simulate(
        theta=theta,
        n_trials=12,
        n_participants=2,
        random_state=4,
    )

    assert config.response == ["response"]
    assert config.decision_process == f"inv_temp_softmax_{n_choices}"
    assert config.list_params == ["rl_alpha", "beta"]
    assert config._computed_ssm_params == [f"q{i}" for i in range(n_choices)]
    assert assembled.gradient == "available"
    assert assembled.response == ["response"]
    assert assembled.computed_params == [f"q{i}" for i in range(n_choices)]
    assert set(data["response"].unique()).issubset(set(range(n_choices)))
    assert np.all(data["rt"] == -1.0)
    assert config.validate_data(data.drop(columns=["rt"])).ok


def test_choice_only_preset_info_documents_hssm_contract():
    info = rl.preset.info("2AB_RW_InvTempSoftmax")

    assert info["learning_process"] == "RescorlaWagnerDeltaRule_CO"
    assert info["decision_process"] == "inv_temp_softmax_2"
    assert info["required_parameters"] == ["rl_alpha", "beta"]
    assert info["response_labels"] == (0, 1)
    assert info["context_fields"] == ["feedback"]
    assert info["hssm_compatibility"]["participant_contract"] is True
    assert info["gradient"] == "available"


def test_response_only_validation_rejects_invalid_response_without_rt():
    config = rl.preset.get("2AB_RW_InvTempSoftmax")
    data = pd.DataFrame(
        {
            "participant_id": [0, 0],
            "response": [0, 2],
            "feedback": [1.0, 0.0],
        }
    )

    report = config.validate_data(data)

    assert not report.ok
    assert any(issue.code == "invalid_response_labels" for issue in report.issues)


def test_response_only_validation_rejects_fractional_response_without_rt():
    config = rl.preset.get("2AB_RW_InvTempSoftmax")
    data = pd.DataFrame(
        {
            "participant_id": [0, 0],
            "response": [0.0, 0.5],
            "feedback": [1.0, 0.0],
        }
    )

    report = config.validate_data(data)

    assert not report.ok
    assert any(issue.code == "invalid_response_labels" for issue in report.issues)


def test_response_only_ppc_uses_observed_responses_without_observed_rt():
    config = rl.preset.get("2AB_RW_InvTempSoftmax")
    sim = rl.Simulator(config)
    observed = pd.DataFrame(
        {
            "participant_id": [0, 0, 0],
            "trial_id": [0, 1, 2],
            "response": [0, 1, 1],
            "feedback": [1.0, 0.0, 1.0],
        }
    )

    data = sim.simulate(
        theta={"rl_alpha": 0.5, "beta": 2.0},
        mode="ppc",
        observed_data=observed,
        random_state=8,
    )

    assert data["trial_id"].tolist() == [0, 1, 2]
    assert set(data["response"].unique()).issubset({0, 1})
    assert np.all(data["rt"] == -1.0)
    assert data["feedback"].tolist() == [1.0, 0.0, 1.0]
