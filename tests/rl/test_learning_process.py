"""Tests for LearningProcess protocol and Rescorla-Wagner learning classes."""

import numpy as np
import pytest

import ssms.rl.learning as learning
from ssms.rl.learning import (
    LearningProcess,
    RescorlaWagnerDeltaRule,
    RescorlaWagnerDrift,
    RescorlaWagnerDualAlphaDrift,
    RescorlaWagnerDualAlphaRule,
    RescorlaWagnerDualAlphaSoftmax,
    RescorlaWagnerSoftmax,
)


class TestRescorlaWagnerDeltaRule:
    def setup_method(self):
        self.rw = RescorlaWagnerDeltaRule(n_actions=3, initial_q=0.5)
        self.rw.reset()

    def test_protocol_compliance(self):
        assert isinstance(RescorlaWagnerDeltaRule(), LearningProcess)

    def test_initial_q_values(self):
        np.testing.assert_array_equal(self.rw.q_values, [0.5, 0.5, 0.5])

    def test_core_metadata(self):
        assert self.rw.computed_params == []
        assert self.rw.free_params == ["rl_alpha"]
        assert self.rw.param_bounds == {"rl_alpha": (0.0, 1.0)}
        assert self.rw.default_params == {"rl_alpha": 0.2}
        assert self.rw.required_context_fields == ["choice", "feedback"]
        assert self.rw.available_backends == ("python", "jax")
        assert self.rw.supports_gradient is True

    def test_python_state_api_updates_without_mutating_input(self):
        params = {"rl_alpha": 0.5}
        state = self.rw.init_state()

        computed = self.rw.compute_python(state, params, context={})
        next_state = self.rw.update_python(
            state,
            params,
            context={"choice": 2, "feedback": 1.0},
        )

        assert computed == {}
        np.testing.assert_allclose(state["q_values"], [0.5, 0.5, 0.5])
        np.testing.assert_allclose(next_state["q_values"], [0.5, 0.5, 0.75])
        assert next_state is not state

    def test_mutable_api_delegates_to_python_state_api(self):
        params = {"rl_alpha": 0.25}

        assert self.rw.compute_ssm_params(params) == {}
        self.rw.update(action=1, reward=0.0, trial_params=params)

        np.testing.assert_allclose(self.rw.q_values, [0.5, 0.375, 0.5])

    def test_update_python_rejects_out_of_range_choice(self):
        state = self.rw.init_state()

        with pytest.raises(ValueError, match="choice 3 out of range"):
            self.rw.update_python(
                state,
                {"rl_alpha": 0.5},
                context={"choice": 3, "feedback": 1.0},
            )

    def test_update_python_rejects_non_int_choice(self):
        state = self.rw.init_state()

        with pytest.raises(TypeError, match="choice must be an int"):
            self.rw.update_python(
                state,
                {"rl_alpha": 0.5},
                context={"choice": 1.0, "feedback": 1.0},
            )

    def test_reset_clears_state(self):
        params = {"rl_alpha": 0.5}
        self.rw.update(action=0, reward=1.0, trial_params=params)
        assert not np.array_equal(self.rw.q_values, [0.5, 0.5, 0.5])
        self.rw.reset()
        np.testing.assert_array_equal(self.rw.q_values, [0.5, 0.5, 0.5])

    def test_q_values_property_returns_copy(self):
        q = self.rw.q_values
        q[0] = 999.0
        np.testing.assert_array_equal(self.rw.q_values, [0.5, 0.5, 0.5])

    def test_q_values_none_before_reset(self):
        rw = RescorlaWagnerDeltaRule()
        assert rw.q_values is None

    def test_compute_requires_reset(self):
        rw = RescorlaWagnerDeltaRule()

        with pytest.raises(RuntimeError, match="Call reset"):
            rw.compute_ssm_params({"rl_alpha": 0.2})

    def test_update_requires_reset(self):
        rw = RescorlaWagnerDeltaRule()

        with pytest.raises(RuntimeError, match="Call reset"):
            rw.update(action=0, reward=1.0, trial_params={"rl_alpha": 0.2})

    def test_invalid_n_actions(self):
        with pytest.raises(ValueError, match="at least 2"):
            RescorlaWagnerDeltaRule(n_actions=1)

    def test_non_int_n_actions(self):
        with pytest.raises(TypeError, match="n_actions must be an int"):
            RescorlaWagnerDeltaRule(n_actions=2.0)


class TestRescorlaWagnerDrift:
    def setup_method(self):
        self.rw = RescorlaWagnerDrift(n_actions=2, initial_q=0.5)
        self.rw.reset()

    def test_protocol_compliance(self):
        assert isinstance(RescorlaWagnerDrift(), LearningProcess)

    def test_metadata(self):
        assert self.rw.computed_params == ["v"]
        assert self.rw.free_params == ["rl_alpha", "scaler"]
        assert self.rw.param_bounds == {
            "rl_alpha": (0.0, 1.0),
            "scaler": (0.001, 10.0),
        }
        assert self.rw.default_params == {"rl_alpha": 0.2, "scaler": 2.0}
        assert self.rw.available_backends == ("python", "jax")
        assert self.rw.supports_gradient is True

    def test_compute_v_initial(self):
        for scaler in [0.5, 1.0, 2.0, 10.0]:
            result = self.rw.compute_ssm_params({"scaler": scaler})
            assert result["v"] == 0.0

    def test_compute_v_after_update(self):
        params = {"rl_alpha": 0.5, "scaler": 2.0}
        self.rw.update(action=0, reward=1.0, trial_params=params)
        np.testing.assert_allclose(self.rw.q_values, [0.75, 0.5])
        assert self.rw.compute_ssm_params(params)["v"] == pytest.approx(-0.5)

    def test_drift_before_update_ordering(self):
        params = {"rl_alpha": 0.5, "scaler": 1.0}
        assert self.rw.compute_ssm_params(params)["v"] == 0.0
        self.rw.update(action=1, reward=1.0, trial_params=params)
        assert self.rw.compute_ssm_params(params)["v"] == pytest.approx(0.25)

    def test_multiple_updates_trajectory(self):
        params = {"rl_alpha": 0.3, "scaler": 2.0}
        sequence = [(0, 1.0), (1, 0.0), (0, 1.0), (1, 1.0)]
        expected_q = [
            [0.65, 0.5],
            [0.65, 0.35],
            [0.755, 0.35],
            [0.755, 0.545],
        ]
        expected_v_before = [0.0, -0.3, -0.6, -0.81]

        for i, (action, reward) in enumerate(sequence):
            v = self.rw.compute_ssm_params(params)["v"]
            assert v == pytest.approx(expected_v_before[i], abs=1e-12)
            self.rw.update(action=action, reward=reward, trial_params=params)
            np.testing.assert_allclose(self.rw.q_values, expected_q[i], atol=1e-12)

    def test_numerical_equivalence_with_hssm_two_action_rw(self):
        alpha = 0.25
        scaler = 1.5
        params = {"rl_alpha": alpha, "scaler": scaler}
        actions = [0, 1, 1, 0, 1, 0, 0, 1, 0, 1]
        rewards = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]

        our_drifts = []
        for action, reward in zip(actions, rewards):
            our_drifts.append(self.rw.compute_ssm_params(params)["v"])
            self.rw.update(action=action, reward=reward, trial_params=params)

        q_val = np.array([0.5, 0.5], dtype=np.float64)
        hssm_drifts = []
        for action, reward in zip(actions, rewards):
            hssm_drifts.append(float((q_val[1] - q_val[0]) * scaler))
            delta_rl = reward - q_val[action]
            q_val[action] = q_val[action] + alpha * delta_rl

        np.testing.assert_allclose(our_drifts, hssm_drifts, atol=1e-15)

    def test_rejects_more_than_two_actions(self):
        with pytest.raises(ValueError, match="two-action"):
            RescorlaWagnerDrift(n_actions=3)

    def test_jax_backend_matches_python_backend_when_jax_is_installed(self):
        jnp = pytest.importorskip("jax.numpy")

        rw = RescorlaWagnerDrift(n_actions=2, initial_q=0.5)
        params = {"rl_alpha": 0.25, "scaler": 2.0}
        state = rw.init_state()
        jax_state = rw.init_jax_state()

        actions = [0, 1, 0, 1]
        rewards = [1.0, 0.0, 0.0, 1.0]
        python_drifts = []
        jax_drifts = []
        for action, reward in zip(actions, rewards):
            context = {"choice": action, "feedback": reward}
            jax_context = {
                "choice": jnp.asarray(action),
                "feedback": jnp.asarray(reward),
            }
            python_drifts.append(rw.compute_python(state, params, context)["v"])
            jax_drifts.append(
                float(rw.compute_jax(jax_state, params, jax_context)["v"])
            )
            state = rw.update_python(state, params, context)
            jax_state = rw.update_jax(jax_state, params, jax_context)

        np.testing.assert_allclose(python_drifts, jax_drifts, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(
            state["q_values"], np.asarray(jax_state["q_values"]), rtol=1e-6, atol=1e-7
        )

    def test_jax_update_is_differentiable(self):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")

        rw = RescorlaWagnerDrift(n_actions=2, initial_q=0.5)

        def drift_after_update(alpha):
            state = rw.init_jax_state()
            params = {"rl_alpha": alpha, "scaler": jnp.asarray(2.0)}
            context = {"choice": jnp.asarray(0), "feedback": jnp.asarray(1.0)}
            state = rw.update_jax(state, params, context)
            return rw.compute_jax(state, params, context)["v"]

        grad = jax.jit(jax.grad(drift_after_update))(jnp.asarray(0.6))

        assert grad == pytest.approx(-1.0)


class TestRescorlaWagnerSoftmax:
    def setup_method(self):
        self.rw = RescorlaWagnerSoftmax(n_actions=3, initial_q=0.5)
        self.rw.reset()

    def test_protocol_compliance(self):
        assert isinstance(RescorlaWagnerSoftmax(), LearningProcess)

    def test_metadata(self):
        assert self.rw.computed_params == ["q0", "q1", "q2"]
        assert self.rw.free_params == ["rl_alpha"]
        assert self.rw.param_bounds == {"rl_alpha": (0.0, 1.0)}
        assert self.rw.default_params == {"rl_alpha": 0.2}
        assert self.rw.required_context_fields == ["choice", "feedback"]
        assert self.rw.available_backends == ("python", "jax")
        assert self.rw.supports_gradient is True

    def test_compute_python_emits_pre_update_q_values(self):
        params = {"rl_alpha": 0.5}
        state = self.rw.init_state()

        computed = self.rw.compute_python(state, params, context={})
        next_state = self.rw.update_python(
            state,
            params,
            context={"choice": 2, "feedback": 1.0},
        )

        assert computed == {"q0": 0.5, "q1": 0.5, "q2": 0.5}
        np.testing.assert_allclose(state["q_values"], [0.5, 0.5, 0.5])
        np.testing.assert_allclose(next_state["q_values"], [0.5, 0.5, 0.75])

    def test_mutable_api_emits_q_values_and_updates_only_observed_action(self):
        params = {"rl_alpha": 0.25}

        assert self.rw.compute_ssm_params(params) == {
            "q0": 0.5,
            "q1": 0.5,
            "q2": 0.5,
        }
        self.rw.update(action=1, reward=0.0, trial_params=params)

        np.testing.assert_allclose(self.rw.q_values, [0.5, 0.375, 0.5])
        assert self.rw.compute_ssm_params(params) == {
            "q0": 0.5,
            "q1": 0.375,
            "q2": 0.5,
        }

    def test_update_python_rejects_out_of_range_choice(self):
        state = self.rw.init_state()

        with pytest.raises(ValueError, match="choice 3 out of range"):
            self.rw.update_python(
                state,
                {"rl_alpha": 0.5},
                context={"choice": 3, "feedback": 1.0},
            )

    def test_invalid_n_actions(self):
        with pytest.raises(ValueError, match="at least 2"):
            RescorlaWagnerSoftmax(n_actions=1)

    def test_jax_backend_matches_python_backend_when_jax_is_installed(self):
        jnp = pytest.importorskip("jax.numpy")

        rw = RescorlaWagnerSoftmax(n_actions=3, initial_q=0.5)
        params = {"rl_alpha": 0.25}
        state = rw.init_state()
        jax_state = rw.init_jax_state()

        actions = [0, 2, 1, 2]
        rewards = [1.0, 1.0, 0.0, 0.0]
        for action, reward in zip(actions, rewards):
            context = {"choice": action, "feedback": reward}
            jax_context = {
                "choice": jnp.asarray(action),
                "feedback": jnp.asarray(reward),
            }
            assert set(rw.compute_python(state, params, context)) == {
                "q0",
                "q1",
                "q2",
            }
            state = rw.update_python(state, params, context)
            jax_state = rw.update_jax(jax_state, params, jax_context)

        np.testing.assert_allclose(
            state["q_values"], np.asarray(jax_state["q_values"]), rtol=1e-6, atol=1e-7
        )

    def test_jax_update_is_differentiable(self):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")

        rw = RescorlaWagnerSoftmax(n_actions=3, initial_q=0.5)

        def q2_after_update(alpha):
            state = rw.init_jax_state()
            params = {"rl_alpha": alpha}
            context = {"choice": jnp.asarray(2), "feedback": jnp.asarray(1.0)}
            state = rw.update_jax(state, params, context)
            return rw.compute_jax(state, params, context)["q2"]

        grad = jax.jit(jax.grad(q2_after_update))(jnp.asarray(0.6))

        assert grad == pytest.approx(0.5)


class TestRescorlaWagnerRaceDrifts:
    def test_computes_scaled_q_values_as_race_drifts(self):
        learner = learning.RescorlaWagnerRaceDrifts(n_actions=4, initial_q=0.5)
        state = learner.init_state()

        computed = learner.compute_python(state, {"scaler": 2.0}, context={})

        assert learner.computed_params == ["v0", "v1", "v2", "v3"]
        assert learner.free_params == ["rl_alpha", "scaler"]
        assert computed == {"v0": 1.0, "v1": 1.0, "v2": 1.0, "v3": 1.0}

    def test_jax_matches_python_scaling_contract(self):
        learner = learning.RescorlaWagnerRaceDrifts(n_actions=4, initial_q=0.5)
        state = learner.init_state()
        state["q_values"] = np.asarray([0.1, 0.2, 0.3, 0.4])

        computed = learner.compute_jax(state, {"scaler": 3.0}, context={})

        np.testing.assert_allclose(
            np.asarray([computed[f"v{i}"] for i in range(4)]),
            [0.3, 0.6, 0.9, 1.2],
        )


class TestRescorlaWagnerDualAlphaRule:
    def setup_method(self):
        self.rw = RescorlaWagnerDualAlphaRule(n_actions=3, initial_q=0.5)
        self.rw.reset()

    def test_protocol_compliance(self):
        assert isinstance(RescorlaWagnerDualAlphaRule(), LearningProcess)

    def test_initial_q_values(self):
        np.testing.assert_array_equal(self.rw.q_values, [0.5, 0.5, 0.5])

    def test_core_metadata(self):
        assert self.rw.computed_params == []
        assert self.rw.free_params == ["rl_alpha", "rl_alpha_neg"]
        assert self.rw.param_bounds == {
            "rl_alpha": (0.0, 1.0),
            "rl_alpha_neg": (0.0, 1.0),
        }
        assert self.rw.default_params == {"rl_alpha": 0.2, "rl_alpha_neg": 0.2}

    def test_positive_prediction_error_uses_rl_alpha(self):
        params = {"rl_alpha": 0.6, "rl_alpha_neg": 0.1}
        self.rw.update(action=0, reward=1.0, trial_params=params)
        np.testing.assert_allclose(self.rw.q_values, [0.8, 0.5, 0.5])

    def test_negative_prediction_error_uses_rl_alpha_neg(self):
        params = {"rl_alpha": 0.6, "rl_alpha_neg": 0.1}
        self.rw.update(action=0, reward=0.0, trial_params=params)
        np.testing.assert_allclose(self.rw.q_values, [0.45, 0.5, 0.5])

    def test_python_state_api_uses_sign_dependent_learning_rates(self):
        params = {"rl_alpha": 0.6, "rl_alpha_neg": 0.1}
        state = self.rw.init_state()

        positive_state = self.rw.update_python(
            state, params, {"choice": 0, "feedback": 1.0}
        )
        negative_state = self.rw.update_python(
            positive_state, params, {"choice": 0, "feedback": 0.0}
        )

        np.testing.assert_allclose(state["q_values"], [0.5, 0.5, 0.5])
        np.testing.assert_allclose(positive_state["q_values"], [0.8, 0.5, 0.5])
        np.testing.assert_allclose(negative_state["q_values"], [0.72, 0.5, 0.5])

    def test_update_requires_reset(self):
        rw = RescorlaWagnerDualAlphaRule()

        with pytest.raises(RuntimeError, match="Call reset"):
            rw.update(
                action=0,
                reward=1.0,
                trial_params={"rl_alpha": 0.6, "rl_alpha_neg": 0.1},
            )


class TestRescorlaWagnerDualAlphaDrift:
    def setup_method(self):
        self.rw = RescorlaWagnerDualAlphaDrift(n_actions=2, initial_q=0.5)
        self.rw.reset()

    def test_protocol_compliance(self):
        assert isinstance(RescorlaWagnerDualAlphaDrift(), LearningProcess)

    def test_metadata(self):
        assert self.rw.computed_params == ["v"]
        assert self.rw.free_params == ["rl_alpha", "rl_alpha_neg", "scaler"]
        assert self.rw.param_bounds == {
            "rl_alpha": (0.0, 1.0),
            "rl_alpha_neg": (0.0, 1.0),
            "scaler": (0.001, 10.0),
        }
        assert self.rw.default_params == {
            "rl_alpha": 0.2,
            "rl_alpha_neg": 0.2,
            "scaler": 2.0,
        }

    def test_multiple_updates_trajectory(self):
        params = {"rl_alpha": 0.6, "rl_alpha_neg": 0.1, "scaler": 2.0}
        sequence = [(0, 1.0), (0, 0.0), (1, 1.0), (1, 0.0)]
        expected_v_before = [0.0, -0.6, -0.44, 0.16]
        expected_q = [
            [0.8, 0.5],
            [0.72, 0.5],
            [0.72, 0.8],
            [0.72, 0.72],
        ]

        for i, (action, reward) in enumerate(sequence):
            v = self.rw.compute_ssm_params(params)["v"]
            assert v == pytest.approx(expected_v_before[i], abs=1e-12)
            self.rw.update(action=action, reward=reward, trial_params=params)
            np.testing.assert_allclose(self.rw.q_values, expected_q[i], atol=1e-12)

    def test_rejects_more_than_two_actions(self):
        with pytest.raises(ValueError, match="two-action"):
            RescorlaWagnerDualAlphaDrift(n_actions=3)

    def test_jax_backend_matches_python_backend_when_jax_is_installed(self):
        jnp = pytest.importorskip("jax.numpy")

        rw = RescorlaWagnerDualAlphaDrift(n_actions=2, initial_q=0.5)
        params = {"rl_alpha": 0.6, "rl_alpha_neg": 0.1, "scaler": 2.0}
        state = rw.init_state()
        jax_state = rw.init_jax_state()

        actions = [0, 0, 1, 1]
        rewards = [1.0, 0.0, 1.0, 0.0]
        python_drifts = []
        jax_drifts = []
        for action, reward in zip(actions, rewards):
            context = {"choice": action, "feedback": reward}
            jax_context = {
                "choice": jnp.asarray(action),
                "feedback": jnp.asarray(reward),
            }
            python_drifts.append(rw.compute_python(state, params, context)["v"])
            jax_drifts.append(
                float(rw.compute_jax(jax_state, params, jax_context)["v"])
            )
            state = rw.update_python(state, params, context)
            jax_state = rw.update_jax(jax_state, params, jax_context)

        np.testing.assert_allclose(python_drifts, jax_drifts, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(
            state["q_values"], np.asarray(jax_state["q_values"]), rtol=1e-6, atol=1e-7
        )
        assert isinstance(jax_state["q_values"], jnp.ndarray)

    def test_jax_update_is_differentiable_through_negative_delta(self):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")

        rw = RescorlaWagnerDualAlphaDrift(n_actions=2, initial_q=0.5)

        def drift_after_negative_update(alpha_neg):
            state = rw.init_jax_state()
            params = {
                "rl_alpha": jnp.asarray(0.6),
                "rl_alpha_neg": alpha_neg,
                "scaler": jnp.asarray(2.0),
            }
            positive_context = {
                "choice": jnp.asarray(0),
                "feedback": jnp.asarray(1.0),
            }
            negative_context = {
                "choice": jnp.asarray(0),
                "feedback": jnp.asarray(0.0),
            }
            state = rw.update_jax(state, params, positive_context)
            state = rw.update_jax(state, params, negative_context)
            return rw.compute_jax(state, params, negative_context)["v"]

        grad = jax.jit(jax.grad(drift_after_negative_update))(jnp.asarray(0.1))

        assert grad == pytest.approx(1.6)


class TestRescorlaWagnerDualAlphaSoftmax:
    def setup_method(self):
        self.rw = RescorlaWagnerDualAlphaSoftmax(n_actions=3, initial_q=0.5)
        self.rw.reset()

    def test_protocol_compliance(self):
        assert isinstance(RescorlaWagnerDualAlphaSoftmax(), LearningProcess)

    def test_metadata(self):
        assert self.rw.computed_params == ["q0", "q1", "q2"]
        assert self.rw.free_params == ["rl_alpha", "rl_alpha_neg"]
        assert self.rw.param_bounds == {
            "rl_alpha": (0.0, 1.0),
            "rl_alpha_neg": (0.0, 1.0),
        }
        assert self.rw.default_params == {"rl_alpha": 0.2, "rl_alpha_neg": 0.2}

    def test_computes_q_values_and_uses_sign_dependent_learning_rates(self):
        params = {"rl_alpha": 0.6, "rl_alpha_neg": 0.1}

        assert self.rw.compute_ssm_params(params) == {
            "q0": 0.5,
            "q1": 0.5,
            "q2": 0.5,
        }
        self.rw.update(action=2, reward=1.0, trial_params=params)
        self.rw.update(action=2, reward=0.0, trial_params=params)

        np.testing.assert_allclose(self.rw.q_values, [0.5, 0.5, 0.72])
        assert self.rw.compute_ssm_params(params) == {
            "q0": 0.5,
            "q1": 0.5,
            "q2": 0.72,
        }

    def test_jax_backend_matches_python_backend_when_jax_is_installed(self):
        jnp = pytest.importorskip("jax.numpy")

        rw = RescorlaWagnerDualAlphaSoftmax(n_actions=3, initial_q=0.5)
        params = {"rl_alpha": 0.6, "rl_alpha_neg": 0.1}
        state = rw.init_state()
        jax_state = rw.init_jax_state()

        actions = [0, 2, 2, 1]
        rewards = [1.0, 1.0, 0.0, 0.0]
        for action, reward in zip(actions, rewards):
            context = {"choice": action, "feedback": reward}
            jax_context = {
                "choice": jnp.asarray(action),
                "feedback": jnp.asarray(reward),
            }
            state = rw.update_python(state, params, context)
            jax_state = rw.update_jax(jax_state, params, jax_context)

        np.testing.assert_allclose(
            state["q_values"], np.asarray(jax_state["q_values"]), rtol=1e-6, atol=1e-7
        )

    def test_jax_update_is_differentiable_through_negative_delta(self):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")

        rw = RescorlaWagnerDualAlphaSoftmax(n_actions=3, initial_q=0.5)

        def q2_after_negative_update(alpha_neg):
            state = rw.init_jax_state()
            params = {
                "rl_alpha": jnp.asarray(0.6),
                "rl_alpha_neg": alpha_neg,
            }
            positive_context = {
                "choice": jnp.asarray(2),
                "feedback": jnp.asarray(1.0),
            }
            negative_context = {
                "choice": jnp.asarray(2),
                "feedback": jnp.asarray(0.0),
            }
            state = rw.update_jax(state, params, positive_context)
            state = rw.update_jax(state, params, negative_context)
            return rw.compute_jax(state, params, negative_context)["q2"]

        grad = jax.jit(jax.grad(q2_after_negative_update))(jnp.asarray(0.1))

        assert grad == pytest.approx(-0.8)
