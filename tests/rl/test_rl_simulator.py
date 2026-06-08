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

    def test_default_excludes_choice(self, data):
        assert "choice" not in data.columns

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


class TestSimulationMode:
    def test_generative_mode_matches_default_behavior(self):
        sim = _make_simulator()

        default = sim.simulate(
            theta=THETA, n_trials=5, n_participants=2, random_state=42
        )
        generative = sim.simulate(
            theta=THETA,
            n_trials=5,
            n_participants=2,
            random_state=42,
            mode="generative",
        )

        pd.testing.assert_frame_equal(default, generative)

    def test_unknown_mode_raises(self):
        sim = _make_simulator()

        with pytest.raises(ValueError, match="mode must be one of"):
            sim.simulate(theta=THETA, n_trials=1, n_participants=1, mode="other")


class TestPPCMode:
    def test_ppc_requires_observed_data(self):
        sim = _make_simulator()

        with pytest.raises(ValueError, match="observed_data is required"):
            sim.simulate(theta=THETA, mode="ppc")

    def test_ppc_requires_observed_columns(self):
        sim = _make_simulator()
        observed = pd.DataFrame(
            {
                "participant_id": [0],
                "trial_id": [0],
                "rt": [0.5],
                "response": [-1],
            }
        )

        with pytest.raises(ValueError, match="missing_column"):
            sim.simulate(theta=THETA, mode="ppc", observed_data=observed)

    def test_ppc_preserves_observed_row_order_with_arbitrary_trial_ids(self):
        from unittest.mock import patch

        sim = _make_simulator()
        observed = pd.DataFrame(
            {
                "participant_id": [0, 0],
                "trial_id": [2, 0],
                "rt": [0.5, 0.6],
                "response": [-1, 1],
                "feedback": [1.0, 0.0],
            }
        )

        def mock_simulator(**kwargs):
            return {"rts": np.array([[0.5]]), "choices": np.array([[1]])}

        with patch("ssms.rl.simulator.ssm_simulator", side_effect=mock_simulator):
            data = sim.simulate(theta=THETA, mode="ppc", observed_data=observed)

        assert data["trial_id"].tolist() == [2, 0]

    def test_ppc_participant_count_must_match_observed_data(self):
        sim = _make_simulator()
        observed = pd.DataFrame(
            {
                "participant_id": [0, 1],
                "trial_id": [0, 0],
                "rt": [0.5, 0.6],
                "response": [-1, 1],
                "feedback": [1.0, 0.0],
            }
        )

        with pytest.raises(ValueError, match="participant count 2"):
            sim.simulate(
                theta=THETA,
                mode="ppc",
                observed_data=observed,
                n_participants=3,
            )

    def test_ppc_copies_observed_outcome_and_updates_from_observed_response(self):
        from unittest.mock import patch

        sim = _make_simulator(
            task_environment=rl.env.Bandit.bernoulli(
                probabilities=[0.0, 0.0], response_labels=[-1, 1]
            )
        )
        observed = pd.DataFrame(
            {
                "participant_id": [0, 0],
                "trial_id": [0, 1],
                "rt": [0.5, 0.6],
                "response": [-1, 1],
                "feedback": [1.0, 0.0],
            }
        )
        simulated_choices = iter([1, -1])
        seen_v = []

        def mock_simulator(**kwargs):
            seen_v.append(kwargs["theta"]["v"])
            return {
                "rts": np.array([[0.5]]),
                "choices": np.array([[next(simulated_choices)]]),
            }

        with patch("ssms.rl.simulator.ssm_simulator", side_effect=mock_simulator):
            data = sim.simulate(
                theta={**THETA, "rl_alpha": 1.0},
                mode="ppc",
                observed_data=observed,
                random_state=42,
            )

        assert data["response"].tolist() == [1, -1]
        assert data["feedback"].tolist() == [1.0, 0.0]
        np.testing.assert_allclose(seen_v, [0.0, -1.0])
        np.testing.assert_allclose(
            sim.config.learning_process.q_values, np.array([1.0, 0.0])
        )

    def test_ppc_omission_emits_missing_choice_but_copies_observed_context(self):
        from unittest.mock import patch

        sim = _make_simulator(include_choice=True)
        observed = pd.DataFrame(
            {
                "participant_id": [0],
                "trial_id": [0],
                "rt": [0.5],
                "response": [1],
                "feedback": [1.0],
            }
        )

        def mock_simulator(**kwargs):
            return {
                "rts": np.array([[OMISSION_SENTINEL]]),
                "choices": np.array([[-1]]),
            }

        with patch("ssms.rl.simulator.ssm_simulator", side_effect=mock_simulator):
            data = sim.simulate(
                theta=THETA,
                mode="ppc",
                observed_data=observed,
                random_state=42,
            )

        assert data.loc[0, "rt"] == OMISSION_SENTINEL
        assert data.loc[0, "response"] == MISSING_RESPONSE_SENTINEL
        assert data.loc[0, "choice"] == MISSING_RESPONSE_SENTINEL
        assert data.loc[0, "feedback"] == 1.0


class TestValidation:
    def test_missing_theta_param(self):
        sim = _make_simulator()
        incomplete = {"rl_alpha": 0.2, "scaler": 2.0}  # missing SSM params
        with pytest.raises(ValueError, match="theta is missing"):
            sim.simulate(theta=incomplete, n_trials=5, n_participants=1)

    def test_unknown_theta_param(self):
        sim = _make_simulator()
        with pytest.raises(ValueError, match="theta contains unknown params"):
            sim.simulate(theta={**THETA, "unused": 1.0}, n_trials=5, n_participants=1)


class TestParticipantWiseTheta:
    def test_scalar_theta_is_shared_across_participants(self):
        from unittest.mock import patch

        sim = _make_simulator()
        seen_theta = []

        def mock_simulator(**kwargs):
            seen_theta.append(kwargs["theta"])
            return {"rts": np.array([[0.5]]), "choices": np.array([[1]])}

        with patch("ssms.rl.simulator.ssm_simulator", side_effect=mock_simulator):
            df = sim.simulate(theta=THETA, n_trials=1, n_participants=2)

        assert df["participant_id"].tolist() == [0, 1]
        assert [theta["a"] for theta in seen_theta] == [1.5, 1.5]

    def test_participant_wise_theta_values_are_applied_by_participant(self):
        from unittest.mock import patch

        sim = _make_simulator()
        seen_theta = []

        def mock_simulator(**kwargs):
            seen_theta.append(kwargs["theta"])
            return {"rts": np.array([[0.5]]), "choices": np.array([[1]])}

        theta = {**THETA, "a": [1.1, 1.7], "rl_alpha": np.array([0.1, 0.9])}
        with patch("ssms.rl.simulator.ssm_simulator", side_effect=mock_simulator):
            df = sim.simulate(theta=theta, n_trials=2, n_participants=2)

        assert df.groupby("participant_id").size().to_dict() == {0: 2, 1: 2}
        assert [call_theta["a"] for call_theta in seen_theta] == [1.1, 1.1, 1.7, 1.7]

    def test_participant_count_is_inferred_from_theta_length(self):
        from unittest.mock import patch

        sim = _make_simulator()

        def mock_simulator(**kwargs):
            return {"rts": np.array([[0.5]]), "choices": np.array([[1]])}

        theta = {**THETA, "a": [1.1, 1.5, 1.9]}
        with patch("ssms.rl.simulator.ssm_simulator", side_effect=mock_simulator):
            df = sim.simulate(theta=theta, n_trials=1)

        assert df["participant_id"].tolist() == [0, 1, 2]

    @pytest.mark.parametrize("n_participants", [0, -1, 1.5])
    def test_participant_count_must_be_positive_integer(self, n_participants):
        sim = _make_simulator()

        with pytest.raises(ValueError, match="positive integer"):
            sim.simulate(theta=THETA, n_trials=1, n_participants=n_participants)

    def test_participant_wise_theta_lengths_must_match(self):
        sim = _make_simulator()
        theta = {**THETA, "a": [1.1, 1.5], "z": [0.4, 0.5, 0.6]}

        with pytest.raises(ValueError, match="participant-wise theta values"):
            sim.simulate(theta=theta, n_trials=1)

    def test_participant_wise_theta_values_must_not_be_empty(self):
        sim = _make_simulator()
        theta = {**THETA, "a": []}

        with pytest.raises(ValueError, match="must not be empty"):
            sim.simulate(theta=theta, n_trials=1)

    def test_explicit_participant_count_must_match_theta_length(self):
        sim = _make_simulator()
        theta = {**THETA, "a": [1.1, 1.5]}

        with pytest.raises(ValueError, match="n_participants=3"):
            sim.simulate(theta=theta, n_trials=1, n_participants=3)

    def test_multidimensional_theta_values_are_rejected(self):
        sim = _make_simulator()
        theta = {**THETA, "a": np.array([[1.1, 1.5]])}

        with pytest.raises(ValueError, match="scalar or a one-dimensional"):
            sim.simulate(theta=theta, n_trials=1)

    def test_theta_values_must_be_numeric(self):
        sim = _make_simulator()
        theta = {**THETA, "a": "wide"}

        with pytest.raises(ValueError, match="must be numeric"):
            sim.simulate(theta=theta, n_trials=1)


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

    def test_omission_choice_uses_missing_response_sentinel(self):
        from unittest.mock import patch

        sim = _make_simulator(include_choice=True)

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
        assert df.loc[0, "choice"] == MISSING_RESPONSE_SENTINEL


class TestResponseChoiceMapping:
    def test_response_labels_are_mapped_to_learning_choice_indices(self):
        """Angle/DDM response labels [-1, 1] map to learning choices [0, 1]."""
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
            with pytest.raises(ValueError, match="not in response_to_choice"):
                sim.simulate(theta=THETA, n_trials=1, n_participants=1)

    def test_include_choice_emits_derived_choice(self):
        from unittest.mock import patch

        sim = _make_simulator(include_choice=True)
        choices = iter([-1, 1])

        def mock_simulator(**kwargs):
            return {
                "rts": np.array([[0.5]]),
                "choices": np.array([[next(choices)]]),
            }

        with patch("ssms.rl.simulator.ssm_simulator", side_effect=mock_simulator):
            df = sim.simulate(theta=THETA, n_trials=2, n_participants=1)

        assert df["response"].tolist() == [-1, 1]
        assert df["choice"].tolist() == [0, 1]

    def test_reversed_mapping_changes_learning_updates(self):
        from unittest.mock import patch

        sim = _make_simulator(
            task_environment=rl.env.Bandit.bernoulli(
                probabilities=[1.0, 0.0], response_labels=[-1, 1]
            ),
            response_to_choice={-1: 1, 1: 0},
            include_choice=True,
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
        assert df["choice"].tolist() == [1, 0]
        np.testing.assert_allclose(
            sim.config.learning_process.q_values, np.array([1.0, 0.0])
        )


class TestFunctionalLearningBackend:
    def test_simulator_uses_explicit_learning_state_api(self):
        from unittest.mock import patch

        class FunctionalLearning:
            computed_params = ["v"]
            free_params = ["alpha"]
            param_bounds = {"alpha": (0.0, 1.0)}
            default_params = {"alpha": 0.2}
            n_actions = 2
            available_backends = ("python",)
            supports_gradient = False

            def __init__(self):
                self.seen_states = []

            def init_state(self):
                return {"q_values": np.array([0.5, 0.5], dtype=np.float64)}

            def compute_python(self, state, params, context):
                self.seen_states.append(state["q_values"].copy())
                return {"v": float(state["q_values"][1] - state["q_values"][0])}

            def update_python(self, state, params, context):
                choice = int(context["choice"])
                feedback = float(context["feedback"])
                next_q = state["q_values"].copy()
                next_q[choice] += params["alpha"] * (feedback - next_q[choice])
                return {"q_values": next_q}

            def reset(self, **kwargs):
                raise AssertionError("mutable reset should not be called")

            def compute_ssm_params(self, trial_params):
                raise AssertionError("mutable compute should not be called")

            def update(self, action, reward, trial_params):
                raise AssertionError("mutable update should not be called")

        learning = FunctionalLearning()
        sim = _make_simulator(
            learning_process=learning,
            task_environment=rl.env.Bandit.bernoulli(
                probabilities=[1.0, 0.0], response_labels=[-1, 1]
            ),
        )
        choices = iter([-1, 1])

        def mock_simulator(**kwargs):
            return {
                "rts": np.array([[0.5]]),
                "choices": np.array([[next(choices)]]),
            }

        with patch("ssms.rl.simulator.ssm_simulator", side_effect=mock_simulator):
            sim.simulate(
                theta={"alpha": 1.0, "a": 1.5, "z": 0.5, "t": 0.3, "theta": 0.2},
                n_trials=2,
                n_participants=1,
            )

        np.testing.assert_allclose(learning.seen_states[0], [0.5, 0.5])
        np.testing.assert_allclose(learning.seen_states[1], [1.0, 0.5])

    def test_simulator_supports_legacy_mutable_learning_api(self):
        from unittest.mock import patch

        class MutableLearning:
            computed_params = ["v"]
            free_params = ["alpha"]
            param_bounds = {"alpha": (0.0, 1.0)}
            default_params = {"alpha": 0.2}
            n_actions = 2

            def __init__(self):
                self.q_values = None
                self.seen_v = []

            def reset(self, **kwargs):
                self.q_values = np.array([0.5, 0.5], dtype=np.float64)

            def compute_ssm_params(self, trial_params):
                v = float(self.q_values[1] - self.q_values[0])
                self.seen_v.append(v)
                return {"v": v}

            def update(self, action, reward, trial_params):
                self.q_values[action] += trial_params["alpha"] * (
                    reward - self.q_values[action]
                )

        learning = MutableLearning()
        sim = _make_simulator(
            learning_process=learning,
            task_environment=rl.env.Bandit.bernoulli(
                probabilities=[1.0, 0.0], response_labels=[-1, 1]
            ),
        )
        choices = iter([-1, 1])

        def mock_simulator(**kwargs):
            return {
                "rts": np.array([[0.5]]),
                "choices": np.array([[next(choices)]]),
            }

        with patch("ssms.rl.simulator.ssm_simulator", side_effect=mock_simulator):
            sim.simulate(
                theta={"alpha": 1.0, "a": 1.5, "z": 0.5, "t": 0.3, "theta": 0.2},
                n_trials=2,
                n_participants=1,
            )

        np.testing.assert_allclose(learning.seen_v, [0.0, -0.5])
        np.testing.assert_allclose(learning.q_values, [1.0, 0.0])


class TestMilestone4Generalization:
    def test_context_fields_support_condition_specific_learning_state(self):
        from unittest.mock import patch

        class AlternatingConditionBandit:
            n_arms = 2
            context_fields = ["condition", "feedback"]

            @property
            def response_labels(self):
                return [-1, 1]

            def reset(self, rng=None):
                pass

            def get_trial_context(self, trial_idx):
                return {"condition": float(trial_idx % 2)}

            def sample_context(self, context, trial_idx):
                feedback = int(context["condition"]) == int(context["choice"])
                return {"feedback": float(feedback)}

        class ConditionSpecificLearning:
            computed_params = ["v"]
            free_params = ["alpha", "scaler"]
            param_bounds = {"alpha": (0.0, 1.0), "scaler": (0.001, 10.0)}
            default_params = {"alpha": 0.2, "scaler": 2.0}
            n_actions = 2
            required_context_fields = ["condition", "choice", "feedback"]
            available_backends = ("python",)
            supports_gradient = False

            def init_state(self):
                return {"q_values": np.full((2, 2), 0.5, dtype=np.float64)}

            def compute_python(self, state, params, context):
                condition = int(context["condition"])
                q_values = state["q_values"][condition]
                return {"v": float(params["scaler"] * (q_values[1] - q_values[0]))}

            def update_python(self, state, params, context):
                condition = int(context["condition"])
                choice = int(context["choice"])
                feedback = float(context["feedback"])
                next_q = state["q_values"].copy()
                old_value = next_q[condition, choice]
                next_q[condition, choice] += params["alpha"] * (feedback - old_value)
                return {"q_values": next_q}

        sim = _make_simulator(
            learning_process=ConditionSpecificLearning(),
            task_environment=AlternatingConditionBandit(),
        )
        choices = iter([-1, 1, -1, 1])
        seen_v = []

        def mock_simulator(**kwargs):
            seen_v.append(kwargs["theta"]["v"])
            return {
                "rts": np.array([[0.5]]),
                "choices": np.array([[next(choices)]]),
            }

        with patch("ssms.rl.simulator.ssm_simulator", side_effect=mock_simulator):
            data = sim.simulate(
                theta={
                    "alpha": 1.0,
                    "scaler": 2.0,
                    "a": 1.5,
                    "z": 0.5,
                    "t": 0.3,
                    "theta": 0.2,
                },
                n_trials=4,
                n_participants=1,
            )

        np.testing.assert_allclose(seen_v, [0.0, 0.0, -1.0, 1.0])
        assert data["condition"].tolist() == [0.0, 1.0, 0.0, 1.0]
        assert data["feedback"].tolist() == [1.0, 1.0, 1.0, 1.0]

    def test_simulator_maps_multiple_learning_outputs_to_decision_parameters(self):
        from unittest.mock import patch

        class ControlLearning:
            computed_params = ["drift", "threshold"]
            free_params = ["gain"]
            param_bounds = {"gain": (0.0, 2.0)}
            default_params = {"gain": 0.5}
            n_actions = 2
            required_context_fields = ["choice"]
            available_backends = ("python",)
            supports_gradient = False

            def init_state(self):
                return {"step": 0}

            def compute_python(self, state, params, context):
                return {
                    "drift": float(params["gain"] * state["step"]),
                    "threshold": float(1.0 + 0.1 * state["step"]),
                }

            def update_python(self, state, params, context):
                return {"step": state["step"] + 1}

        sim = _make_simulator(
            learning_process=ControlLearning(),
            computed_param_mapping={"drift": "v", "threshold": "a"},
            task_environment=rl.env.Bandit.bernoulli(
                probabilities=[1.0, 0.0], response_labels=[-1, 1]
            ),
        )
        seen_theta = []

        def mock_simulator(**kwargs):
            seen_theta.append(kwargs["theta"])
            return {"rts": np.array([[0.5]]), "choices": np.array([[1]])}

        with patch("ssms.rl.simulator.ssm_simulator", side_effect=mock_simulator):
            sim.simulate(
                theta={"gain": 0.5, "z": 0.5, "t": 0.3, "theta": 0.2},
                n_trials=2,
                n_participants=1,
            )

        assert sim.config.list_params == ["gain", "z", "t", "theta"]
        np.testing.assert_allclose([theta["v"] for theta in seen_theta], [0.0, 0.5])
        np.testing.assert_allclose([theta["a"] for theta in seen_theta], [1.0, 1.1])


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

    def test_gaussian_response_labels_map_to_learning_choice_indices(self):
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
