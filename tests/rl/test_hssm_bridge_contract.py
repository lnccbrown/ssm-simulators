"""Tests for the public ssms.rl surfaces consumed by HSSM."""

from pathlib import Path

import numpy as np
import pytest

import ssms.rl as rl


def test_preset_assembles_to_hssm_ready_jax_contract():
    pytest.importorskip("jax")

    config = rl.resolve_model("2AB_RW_Angle")
    assembled = config.assemble(backend="jax")

    assert assembled.model_name == "2AB_RW_Angle"
    assert assembled.decision_process == "angle"
    assert assembled.list_params == ["rl_alpha", "scaler", "a", "z", "t", "theta"]
    assert assembled.response == ["rt", "response"]
    assert assembled.choices == (-1, 1)
    assert assembled.context_fields == ["feedback"]
    assert assembled.computed_params == ["v"]
    assert assembled.response_to_choice == {-1: 0, 1: 1}
    assert assembled.learning_backend == "jax"
    assert assembled.gradient == "available"
    assert assembled.participant_input_fields() == [
        "rl_alpha",
        "scaler",
        "response",
        "feedback",
    ]


def test_assembled_output_dict_matches_response_to_choice_for_hssm_wrapper():
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    config = rl.ModelConfig(
        model_name="reversed_response_bridge",
        description="Bridge contract with non-default response mapping",
        decision_process="angle",
        learning_process=rl.learning.RescorlaWagnerDeltaRule(),
        task_environment=rl.env.Bandit.bernoulli(
            probabilities=[0.7, 0.3], response_labels=[-1, 1]
        ),
        response_to_choice={-1: 1, 1: 0},
        learning_backend="jax",
    )
    assembled = config.assemble(backend="jax")
    compute = assembled.assemble_participant_fn(output="dict")
    trials = jnp.asarray(
        [
            [0.5, 2.0, -1.0, 1.0],
            [0.5, 2.0, 1.0, 1.0],
        ]
    )

    values = compute(trials)

    assert set(values) == {"v"}
    np.testing.assert_allclose(np.asarray(values["v"]), [0.0, 0.5])


def test_simulated_preset_data_validates_for_hssm_handoff():
    config = rl.preset.get("2AB_RW_Angle")
    data = rl.Simulator(config).simulate(
        theta={
            "rl_alpha": 0.2,
            "scaler": 2.0,
            "a": 1.5,
            "z": 0.5,
            "t": 0.3,
            "theta": 0.2,
        },
        n_trials=12,
        n_participants=2,
        random_state=4,
    )

    report = config.validate_data(data)

    assert report.ok
    assert report.n_participants == 2
    assert report.n_trials == 12
    report.raise_for_errors()


def test_rlssm_docs_name_active_hssm_bridge_factory():
    docs = Path("docs/api/rlssm.md").read_text()

    assert "RLSSMConfig.from_ssms_model" in docs
    assert "hssm.RLSSM(data=data, model_config=hssm_config)" in docs
    assert "structural inspection" in docs
