"""Tests for RLSSMSimulator."""

import numpy as np
import pandas as pd
import pytest

from ssms import OMISSION_SENTINEL
from ssms.rl.learning_process import RescorlaWagnerDeltaRule
from ssms.rl.rl_config import RLSSMModelConfig
from ssms.rl.rl_simulator import RLSSMSimulator
from ssms.rl.task_environment import TwoArmedBandit


def _make_simulator(**config_overrides):
    """Create a simulator with sensible defaults for testing."""
    defaults = dict(
        model_name="test_rlssm",
        description="Test RLSSM",
        decision_process="angle",
        learning_process=RescorlaWagnerDeltaRule(n_choices=2, initial_q=0.5),
        # Use choices [-1, 1] to match angle model's SSM output
        task_environment=TwoArmedBandit(
            reward_probabilities=[0.7, 0.3], choices=[-1, 1]
        ),
    )
    defaults.update(config_overrides)
    config = RLSSMModelConfig(**defaults)
    return RLSSMSimulator(config)


# Default theta for angle model (v is computed by learning process)
THETA = {"rl_alpha": 0.2, "scaler": 2.0, "a": 1.5, "z": 0.5, "t": 0.3, "theta": 0.2}


class TestSimulateOutput:
    @pytest.fixture()
    def sim(self):
        return _make_simulator()

    @pytest.fixture()
    def data(self, sim):
        return sim.simulate(
            theta=THETA, n_trials=10, n_participants=3, random_state=42
        )

    def test_returns_dataframe(self, data):
        assert isinstance(data, pd.DataFrame)

    def test_balanced_panel(self, data):
        assert len(data) == 3 * 10
        for pid in range(3):
            assert len(data[data["participant_id"] == pid]) == 10

    def test_columns(self, data):
        expected = {"participant_id", "trial_id", "rt", "response", "feedback"}
        assert expected.issubset(set(data.columns))

    def test_sorted_order(self, data):
        assert (
            data.equals(
                data.sort_values(["participant_id", "trial_id"]).reset_index(
                    drop=True
                )
            )
        )

    def test_participant_ids(self, data):
        assert sorted(data["participant_id"].unique()) == [0, 1, 2]

    def test_trial_ids(self, data):
        for pid in range(3):
            trials = sorted(data[data["participant_id"] == pid]["trial_id"].tolist())
            assert trials == list(range(10))

    def test_rt_positive(self, data):
        non_omission = data[data["rt"] != OMISSION_SENTINEL]
        assert (non_omission["rt"] > 0).all()

    def test_response_in_choices(self, data):
        non_omission = data[data["response"] != -999]
        assert set(non_omission["response"].unique()).issubset({-1, 1})

    def test_feedback_binary(self, data):
        non_omission = data[data["rt"] != OMISSION_SENTINEL]
        assert set(non_omission["feedback"].unique()).issubset({0.0, 1.0})


class TestReproducibility:
    def test_same_seed_same_result(self):
        sim = _make_simulator()
        df1 = sim.simulate(theta=THETA, n_trials=10, n_participants=2, random_state=42)
        df2 = sim.simulate(theta=THETA, n_trials=10, n_participants=2, random_state=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_different_result(self):
        sim = _make_simulator()
        df1 = sim.simulate(theta=THETA, n_trials=10, n_participants=2, random_state=1)
        df2 = sim.simulate(theta=THETA, n_trials=10, n_participants=2, random_state=2)
        assert not df1["rt"].equals(df2["rt"])


class TestValidation:
    def test_missing_theta_param(self):
        sim = _make_simulator()
        incomplete = {"rl_alpha": 0.2, "scaler": 2.0}  # missing SSM params
        with pytest.raises(ValueError, match="theta is missing"):
            sim.simulate(theta=incomplete, n_trials=5, n_participants=1)


class TestEdgeCases:
    def test_single_participant(self):
        sim = _make_simulator()
        df = sim.simulate(
            theta=THETA, n_trials=5, n_participants=1, random_state=42
        )
        assert len(df) == 5
        assert df["participant_id"].unique().tolist() == [0]

    def test_single_trial(self):
        sim = _make_simulator()
        df = sim.simulate(
            theta=THETA, n_trials=1, n_participants=2, random_state=42
        )
        assert len(df) == 2


class TestOmissionHandling:
    def test_omission_code_path(self):
        """Verify that the omission sentinel is handled correctly by patching
        the SSM simulator to return an omission on certain trials."""
        from unittest.mock import patch

        sim = _make_simulator()
        call_count = 0

        def mock_simulator(**kwargs):
            nonlocal call_count
            call_count += 1
            # Return omission on trials 2 and 4 (0-indexed within each subject)
            if call_count in (3, 5):
                return {
                    "rts": np.array([[OMISSION_SENTINEL]]),
                    "choices": np.array([[-1]]),
                }
            return {
                "rts": np.array([[0.5]]),
                "choices": np.array([[1]]),
            }

        with patch("ssms.rl.rl_simulator.ssm_simulator", side_effect=mock_simulator):
            df = sim.simulate(
                theta=THETA, n_trials=5, n_participants=1, random_state=42
            )

        assert len(df) == 5
        omissions = df[df["rt"] == OMISSION_SENTINEL]
        assert len(omissions) == 2
        assert (omissions["response"] == -999).all()
        assert (omissions["feedback"] == 0.0).all()
        # Non-omission rows should have valid data
        non_omission = df[df["rt"] != OMISSION_SENTINEL]
        assert (non_omission["rt"] == 0.5).all()
        assert set(non_omission["response"].unique()) == {1}
