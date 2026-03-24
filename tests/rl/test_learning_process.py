"""Tests for LearningProcess protocol and RescorlaWagnerDeltaRule."""

import numpy as np
import pytest

from ssms.rl.learning_process import LearningProcess, RescorlaWagnerDeltaRule


class TestRescorlaWagnerDeltaRule:
    def setup_method(self):
        self.rw = RescorlaWagnerDeltaRule(n_choices=2, initial_q=0.5)
        self.rw.reset()

    def test_initial_q_values(self):
        np.testing.assert_array_equal(self.rw.q_values, [0.5, 0.5])

    def test_compute_v_initial(self):
        """With equal Q-values, v should be 0.0 regardless of scaler."""
        for scaler in [0.5, 1.0, 2.0, 10.0]:
            result = self.rw.compute_ssm_params({"scaler": scaler})
            assert result["v"] == 0.0

    def test_compute_v_after_update(self):
        """After one update (action=0, reward=1.0, alpha=0.5):
        Q = [0.5 + 0.5*(1.0 - 0.5), 0.5] = [0.75, 0.5]
        v = (0.5 - 0.75) * 2.0 = -0.5
        """
        params = {"rl_alpha": 0.5, "scaler": 2.0}
        self.rw.update(action=0, reward=1.0, trial_params=params)
        np.testing.assert_allclose(self.rw.q_values, [0.75, 0.5])
        result = self.rw.compute_ssm_params(params)
        assert result["v"] == pytest.approx(-0.5)

    def test_drift_before_update_ordering(self):
        """compute_ssm_params returns drift BEFORE the update
        (matching HSSM's scan ordering)."""
        params = {"rl_alpha": 0.5, "scaler": 1.0}
        # Before any update, drift should be 0
        v_before = self.rw.compute_ssm_params(params)["v"]
        assert v_before == 0.0
        # Update action=1, reward=1.0 → Q=[0.5, 0.75]
        self.rw.update(action=1, reward=1.0, trial_params=params)
        # Now drift should reflect the updated Q-values
        v_after = self.rw.compute_ssm_params(params)["v"]
        assert v_after == pytest.approx(0.25)  # (0.75 - 0.5) * 1.0

    def test_multiple_updates_trajectory(self):
        """Run a fixed sequence and compare against hand-computed values."""
        params = {"rl_alpha": 0.3, "scaler": 2.0}
        # Trial sequence: (action, reward)
        sequence = [(0, 1.0), (1, 0.0), (0, 1.0), (1, 1.0)]

        expected_q = [
            # After (0, 1.0): Q[0] = 0.5 + 0.3*(1.0-0.5) = 0.65, Q[1] = 0.5
            [0.65, 0.5],
            # After (1, 0.0): Q[0] = 0.65, Q[1] = 0.5 + 0.3*(0.0-0.5) = 0.35
            [0.65, 0.35],
            # After (0, 1.0): Q[0] = 0.65 + 0.3*(1.0-0.65) = 0.755, Q[1] = 0.35
            [0.755, 0.35],
            # After (1, 1.0): Q[0] = 0.755, Q[1] = 0.35 + 0.3*(1.0-0.35) = 0.545
            [0.755, 0.545],
        ]

        # Drift BEFORE each update (computed from Q before the trial's update)
        expected_v_before = [
            (0.5 - 0.5) * 2.0,  # 0.0
            (0.5 - 0.65) * 2.0,  # -0.3
            (0.35 - 0.65) * 2.0,  # -0.6
            (0.35 - 0.755) * 2.0,  # -0.81
        ]

        for i, (action, reward) in enumerate(sequence):
            v = self.rw.compute_ssm_params(params)["v"]
            assert v == pytest.approx(expected_v_before[i], abs=1e-12)
            self.rw.update(action=action, reward=reward, trial_params=params)
            np.testing.assert_allclose(self.rw.q_values, expected_q[i], atol=1e-12)

    def test_numerical_equivalence_with_hssm(self):
        """Run same action/reward sequence through our RW and a NumPy
        reimplementation of HSSM's compute_v_trial_wise. Assert match."""
        alpha = 0.25
        scaler = 1.5
        params = {"rl_alpha": alpha, "scaler": scaler}

        # Fixed action/reward sequence
        actions = [0, 1, 1, 0, 1, 0, 0, 1, 0, 1]
        rewards = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]

        # Our implementation
        our_drifts = []
        for action, reward in zip(actions, rewards):
            v = self.rw.compute_ssm_params(params)["v"]
            our_drifts.append(v)
            self.rw.update(action=action, reward=reward, trial_params=params)

        # HSSM-equivalent NumPy reimplementation
        q_val = np.array([0.5, 0.5], dtype=np.float64)
        hssm_drifts = []
        for action, reward in zip(actions, rewards):
            computed_v = (q_val[1] - q_val[0]) * scaler
            hssm_drifts.append(float(computed_v))
            delta_rl = reward - q_val[action]
            q_val[action] = q_val[action] + alpha * delta_rl

        np.testing.assert_allclose(our_drifts, hssm_drifts, atol=1e-15)

    def test_protocol_compliance(self):
        assert isinstance(RescorlaWagnerDeltaRule(), LearningProcess)

    def test_reset_clears_state(self):
        params = {"rl_alpha": 0.5, "scaler": 1.0}
        self.rw.update(action=0, reward=1.0, trial_params=params)
        assert not np.array_equal(self.rw.q_values, [0.5, 0.5])
        self.rw.reset()
        np.testing.assert_array_equal(self.rw.q_values, [0.5, 0.5])

    def test_q_values_property_returns_copy(self):
        q = self.rw.q_values
        q[0] = 999.0
        np.testing.assert_array_equal(self.rw.q_values, [0.5, 0.5])

    def test_q_values_none_before_reset(self):
        rw = RescorlaWagnerDeltaRule()
        assert rw.q_values is None

    def test_free_params(self):
        assert self.rw.free_params == ["rl_alpha", "scaler"]

    def test_param_bounds(self):
        bounds = self.rw.param_bounds
        assert bounds["rl_alpha"] == (0.0, 1.0)
        assert bounds["scaler"] == (0.001, 10.0)

    def test_default_params(self):
        defaults = self.rw.default_params
        assert defaults == {"rl_alpha": 0.2, "scaler": 2.0}
