"""Tests for rl.Simulator."""

import numpy as np
import pandas as pd
import pytest

from ssms import OMISSION_SENTINEL
import ssms.rl as rl
from ssms.rl.simulator import MISSING_RESPONSE_SENTINEL


def _make_simulator(**config_overrides):
    """Create a simulator with sensible defaults for testing."""
    defaults = dict(
        model_name="test_rlssm",
        description="Test RLSSM",
        decision_process="angle",
        learning_process=rl.learning.RescorlaWagnerDeltaRule(
            n_actions=2, initial_q=0.5
        ),
        # Use response labels [-1, 1] to match angle model's SSM output
        task_environment=rl.env.Bandit.bernoulli(
            probabilities=[0.7, 0.3], response_labels=[-1, 1]
        ),
    )
    defaults.update(config_overrides)
    config = rl.ModelConfig(**defaults)
    return rl.Simulator(config)


# Default theta for angle model (v is computed by learning process)
THETA = {"rl_alpha": 0.2, "scaler": 2.0, "a": 1.5, "z": 0.5, "t": 0.3, "theta": 0.2}
THETA_DUAL = {
    "rl_alpha": 0.2,
    "rl_alpha_neg": 0.1,
    "scaler": 2.0,
    "a": 1.5,
    "z": 0.5,
    "t": 0.3,
    "theta": 0.2,
}


class TestSimulateOutput:
    @pytest.fixture()
    def sim(self):
        return _make_simulator()

    @pytest.fixture()
    def data(self, sim):
        return sim.simulate(theta=THETA, n_trials=10, n_participants=3, random_state=42)

    def test_returns_dataframe(self, data):
        assert isinstance(data, pd.DataFrame)

    def test_balanced_panel(self, data):
        assert len(data) == 3 * 10
        for pid in range(3):
            assert len(data[data["participant_id"] == pid]) == 10

    def test_columns(self, data):
        expected = {"participant_id", "trial_id", "rt", "response", "feedback"}
        assert expected.issubset(set(data.columns))

    def test_default_excludes_action(self, data):
        assert "action" not in data.columns

    def test_sorted_order(self, data):
        assert data.equals(
            data.sort_values(["participant_id", "trial_id"]).reset_index(drop=True)
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

    def test_response_in_labels(self, data):
        non_omission = data[data["response"] != MISSING_RESPONSE_SENTINEL]
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
        df = sim.simulate(theta=THETA, n_trials=5, n_participants=1, random_state=42)
        assert len(df) == 5
        assert df["participant_id"].unique().tolist() == [0]

    def test_single_trial(self):
        sim = _make_simulator()
        df = sim.simulate(theta=THETA, n_trials=1, n_participants=2, random_state=42)
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

        with patch("ssms.rl.simulator.ssm_simulator", side_effect=mock_simulator):
            df = sim.simulate(
                theta=THETA, n_trials=5, n_participants=1, random_state=42
            )

        assert len(df) == 5
        omissions = df[df["rt"] == OMISSION_SENTINEL]
        assert len(omissions) == 2
        assert (omissions["response"] == MISSING_RESPONSE_SENTINEL).all()
        assert (omissions["feedback"] == 0.0).all()
        # Non-omission rows should have valid data
        non_omission = df[df["rt"] != OMISSION_SENTINEL]
        assert (non_omission["rt"] == 0.5).all()
        assert set(non_omission["response"].unique()) == {1}

    def test_omission_action_uses_missing_response_sentinel(self):
        from unittest.mock import patch

        sim = _make_simulator(include_action=True)

        def mock_simulator(**kwargs):
            return {
                "rts": np.array([[OMISSION_SENTINEL]]),
                "choices": np.array([[-1]]),
            }

        with patch("ssms.rl.simulator.ssm_simulator", side_effect=mock_simulator):
            df = sim.simulate(
                theta=THETA, n_trials=1, n_participants=1, random_state=42
            )

        assert df.loc[0, "response"] == MISSING_RESPONSE_SENTINEL
        assert df.loc[0, "action"] == MISSING_RESPONSE_SENTINEL


class TestResponseActionMapping:
    def test_response_labels_are_mapped_to_learning_action_indices(self):
        """Angle/DDM response labels [-1, 1] map to learning actions [0, 1]."""
        from unittest.mock import patch

        sim = _make_simulator(
            task_environment=rl.env.Bandit.bernoulli(
                probabilities=[1.0, 0.0], response_labels=[-1, 1]
            )
        )
        choices = iter([-1, 1])
        theta = {**THETA, "rl_alpha": 1.0}

        def mock_simulator(**kwargs):
            return {
                "rts": np.array([[0.5]]),
                "choices": np.array([[next(choices)]]),
            }

        with patch("ssms.rl.simulator.ssm_simulator", side_effect=mock_simulator):
            df = sim.simulate(theta=theta, n_trials=2, n_participants=1)

        assert df["response"].tolist() == [-1, 1]
        np.testing.assert_allclose(
            sim.config.learning_process.q_values, np.array([1.0, 0.0])
        )

    def test_unknown_response_label_raises(self):
        from unittest.mock import patch

        sim = _make_simulator()

        def mock_simulator(**kwargs):
            return {"rts": np.array([[0.5]]), "choices": np.array([[0]])}

        with patch("ssms.rl.simulator.ssm_simulator", side_effect=mock_simulator):
            with pytest.raises(ValueError, match="not in response_mapping"):
                sim.simulate(theta=THETA, n_trials=1, n_participants=1)

    def test_include_action_emits_derived_action(self):
        from unittest.mock import patch

        sim = _make_simulator(include_action=True)
        choices = iter([-1, 1])

        def mock_simulator(**kwargs):
            return {
                "rts": np.array([[0.5]]),
                "choices": np.array([[next(choices)]]),
            }

        with patch("ssms.rl.simulator.ssm_simulator", side_effect=mock_simulator):
            df = sim.simulate(theta=THETA, n_trials=2, n_participants=1)

        assert df["response"].tolist() == [-1, 1]
        assert df["action"].tolist() == [0, 1]

    def test_reversed_mapping_changes_learning_updates(self):
        from unittest.mock import patch

        sim = _make_simulator(
            task_environment=rl.env.Bandit.bernoulli(
                probabilities=[1.0, 0.0], response_labels=[-1, 1]
            ),
            response_mapping={-1: 1, 1: 0},
            include_action=True,
        )
        choices = iter([-1, 1])
        theta = {**THETA, "rl_alpha": 1.0}

        def mock_simulator(**kwargs):
            return {
                "rts": np.array([[0.5]]),
                "choices": np.array([[next(choices)]]),
            }

        with patch("ssms.rl.simulator.ssm_simulator", side_effect=mock_simulator):
            df = sim.simulate(theta=theta, n_trials=2, n_participants=1)

        assert df["response"].tolist() == [-1, 1]
        assert df["action"].tolist() == [1, 0]
        np.testing.assert_allclose(
            sim.config.learning_process.q_values, np.array([1.0, 0.0])
        )


class TestMilestone2Integration:
    def test_dual_alpha_learning_rule_simulates(self):
        sim = _make_simulator(
            learning_process=rl.learning.RescorlaWagnerDualAlphaRule()
        )

        df = sim.simulate(
            theta=THETA_DUAL,
            n_trials=10,
            n_participants=2,
            random_state=42,
        )

        assert len(df) == 20
        assert sim.config.list_params == [
            "rl_alpha",
            "rl_alpha_neg",
            "scaler",
            "a",
            "z",
            "t",
            "theta",
        ]

    def test_gaussian_bandit_simulates_continuous_feedback(self):
        sim = _make_simulator(
            task_environment=rl.env.Bandit.gaussian(
                means=[1.0, 0.0],
                sds=[0.2, 0.2],
                response_labels=[-1, 1],
            )
        )

        df = sim.simulate(theta=THETA, n_trials=30, n_participants=2, random_state=42)
        non_omission = df[df["rt"] != OMISSION_SENTINEL]

        assert len(df) == 60
        assert pd.api.types.is_float_dtype(df["feedback"])
        assert not set(non_omission["feedback"].unique()).issubset({0.0, 1.0})

    def test_gaussian_response_labels_map_to_learning_action_indices(self):
        from unittest.mock import patch

        sim = _make_simulator(
            task_environment=rl.env.Bandit.gaussian(
                means=[1.0, 0.0],
                sds=[1e-12, 1e-12],
                response_labels=[-1, 1],
            )
        )
        choices = iter([-1, 1])
        theta = {**THETA, "rl_alpha": 1.0}

        def mock_simulator(**kwargs):
            return {
                "rts": np.array([[0.5]]),
                "choices": np.array([[next(choices)]]),
            }

        with patch("ssms.rl.simulator.ssm_simulator", side_effect=mock_simulator):
            df = sim.simulate(theta=theta, n_trials=2, n_participants=1)

        assert df["response"].tolist() == [-1, 1]
        np.testing.assert_allclose(
            sim.config.learning_process.q_values,
            np.array([1.0, 0.0]),
            atol=1e-9,
        )
